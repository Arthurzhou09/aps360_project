import argparse
import json
import pandas as pd
from torch_geometric.loader import DataLoader
import torch
import os
import numpy as np
import sys

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.tem_beta import Tem1BetaLactamaseDataset
from data.data_utils import load_cif_structure, parse_structure, load_dms
from data.split import split_by_structural_position, split_by_random
from model.gnn import Tem1BetaGNN
from train.train_utils import (EarlyStopping, standardize, load_config, set_seed,
                               safe_pearson, safe_spearman, build_scheduler)


def _checkpoint(model, label_stats):
    return {
        'model_state': model.state_dict(),
        'config': {
            'node_in_channels': model.node_in_channels,
            'edge_features_dim': model.edge_features_dim,
            'hidden_channels': model.hidden_channels,
            'reg_hidden_channels': model.reg_hidden_channels,
            'mp_layers': model.mp_layers,
            'head_layers': model.head_layers,
            'dropout': model.dropout,
            # without these a model trained with >1 round cannot be rebuilt at inference
            'encoder_rounds': model.encoder_rounds,
            'decoder_rounds': model.decoder_rounds,
        },
        # carried so inference can invert the label scaling without re-deriving it from
        # the split (which silently breaks if the split rule ever changes again)
        'label_stats': label_stats,
    }


def evaluate(model, loader, criterion, device):
    """

    returns:
        metrics: dict with loss, spearman/pearson overall, per group, and 'balanced'
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    predictions, targets, groups = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.node_features, batch.edge_index, batch.distance_features,
                         batch.batch, batch.mutation_idx).squeeze(-1)
            loss = criterion(pred, batch.fitness)
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            predictions.append(pred.cpu().numpy())
            targets.append(batch.fitness.cpu().numpy())
            groups.append(batch.is_single.cpu().numpy().reshape(-1))
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    groups = np.concatenate(groups)

    metrics = {
        "loss": total_loss / total_samples,
        "spearman": safe_spearman(predictions, targets),
        "pearson": safe_pearson(predictions, targets),
    }
    for name, flag in (("single", 1), ("double", 0)):
        sub = groups == flag
        # a group with <2 rows has no defined rank correlation; leave it nan rather than 0
        metrics[f"spearman_{name}"] = safe_spearman(predictions[sub], targets[sub]) if sub.sum() > 1 else float("nan")
        metrics[f"n_{name}"] = int(sub.sum())

    present = [metrics[f"spearman_{n}"] for n in ("single", "double") if metrics[f"n_{n}"] > 1]
    metrics["balanced"] = float(np.mean(present)) if present else float("nan")
    return metrics


def group_loss_weights(train_df, device):
    """
    loss weights that give single and double mutants equal total influence. (double has more samples than singles)
    returns:
        weights: tensor indexed by the is_single flag (index 0 = double, 1 = single)

    note: this function not realyl used now since we trian on single and infer on double
    """
    counts = {1: int((train_df['Single'] == 1).sum()), 0: int((train_df['Single'] == 0).sum())}
    total = sum(counts.values())
    # mean weight stays ~1 so the loss scale (and hence a sensible lr) is unchanged
    weights = torch.ones(2, dtype=torch.float, device=device)
    for flag, count in counts.items():
        weights[flag] = (total / (2.0 * count)) if count else 0.0
    print(f"group loss weights: single {weights[1]:.3f} (n={counts[1]}), double {weights[0]:.3f} (n={counts[0]})")
    return weights


def train(model, num_epochs, train_loader, val_loader, optimizer, criterion, device, output_dir,
          early_stopping=None, scheduler=None, grad_clip=None, label_stats=None, group_weights=None):
    """
    trainig loop. use "mean" on mse
    select on spearman, mean.
    """
    model.to(device)
    train_m = []
    val_m = []
    val_spearman_m = []
    val_balanced_m = []
    best_score = None
    best_state = None
    best_epoch = None

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for b_i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch.node_features, batch.edge_index, batch.distance_features, batch.batch, batch.mutation_idx)
            if group_weights is None:
                loss = criterion(output.squeeze(-1), batch.fitness)
            else:
                # per-sample loss, then a weighted mean so singles and doubles count equally
                per_sample = torch.nn.functional.mse_loss(output.squeeze(-1), batch.fitness, reduction='none')
                w = group_weights[batch.is_single.view(-1)]
                loss = (per_sample * w).sum() / w.sum()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
        mean_loss = total_loss / total_samples
        train_m.append(mean_loss)

        # validation
        metrics = evaluate(model, val_loader, criterion, device)
        val_m.append(metrics["loss"])
        val_spearman_m.append(metrics["spearman"])
        val_balanced_m.append(metrics["balanced"])

        if scheduler is not None:
            scheduler.step(metrics['loss']) # reduceOnplateua

        # if only one group is present, balanced == that group's Spearman, so mixed and
        # single-group runs both select on the same quantity without special-casing
        selection_score = metrics["balanced"] if not np.isnan(metrics["balanced"]) else metrics["spearman"]

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs}  train {mean_loss:.4f}  val {metrics['loss']:.4f}  "
              f"val_spearman {metrics['spearman']:.4f} (single {metrics['spearman_single']:.4f} "
              f"n={metrics['n_single']} | double {metrics['spearman_double']:.4f} n={metrics['n_double']} "
              f"| balanced {metrics['balanced']:.4f})  lr {lr_now:.2e}")

        # logging: keep the best-ranking epoch, and hold its weights so training can be
        # rolled back to it at the end instead of finishing on an overfit final epoch
        if best_score is None or selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(_checkpoint(model, label_stats), os.path.join(output_dir, "best_model.pt"))
            print(f"new model saved (balanced_spearman={selection_score:.6f})")

        if early_stopping is not None:
            early_stopping(-selection_score) # EarlyStopping minimizes, so feed it negated
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

    torch.save(_checkpoint(model, label_stats), os.path.join(output_dir, "final_model.pt"))
    np.save(os.path.join(output_dir, "train_loss.npy"), train_m)
    np.save(os.path.join(output_dir, "val_loss.npy"), val_m)
    np.save(os.path.join(output_dir, "val_spearman.npy"), val_spearman_m)
    np.save(os.path.join(output_dir, "val_balanced_spearman.npy"), val_balanced_m)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights from epoch {best_epoch} (balanced_spearman={best_score:.4f})")

    return best_score, best_epoch


if __name__ == "__main__":

    # node_in_channels = 41 + 2P   (61 for aaindex8, 83 for pca19)
    #   mutation mask (1) + mutant identity one-hot (20) + mutant properties (P)
    #   + property delta mutant-WT, zero off the mutated site(s) (P)
    #   + WT identity one-hot, zero off the mutated site(s) (20)
    #   + structural descriptors, same for every sample (4)
    #
    # edge_features_dim = 96, or 99 when dci_k is set with dci_add_spatial
    #   16 ordered CA/N/C/O atom pairs x 6 gaussian RBFs over 2-12A (96)
    #   + signed DCI coupling for the edge (1) + [from_dci, from_spatial] flags (2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the dms directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration JSON file.")

    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--directed", action="store_true", help="Whether to create directed edges in the graph. False if not specified.")
    parser.add_argument("--seed", type=int, default=1012, help="Random seed for reproducibility.")

    parser.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping (on by default).")
    parser.add_argument("--split", type=str, default="not_random", help="Split to use")
    args = parser.parse_args()
    cfg= load_config(args.config)

    set_seed(args.seed)

    ### Data Loading ###
    pdb_dir = os.path.dirname(args.processed_dms.rstrip("/\\"))
    wt_sequence, _ = parse_structure(
        load_cif_structure(os.path.join(pdb_dir, f"{args.pdb_id}.cif"), args.pdb_id))

    # load_dms drops WT-to-itself rows,
    dms = load_dms(args.processed_dms, wt_sequence)

    if args.split != "random":
        train_df, val_df, _ = split_by_structural_position(
            dms, wt_sequence, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size, seed=args.seed,
            n_blocks=getattr(cfg.data, "n_blocks", None))
    else:
        train_df, val_df, test_df = split_by_random(dms, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size, seed=args.seed)

    train_df, val_df, _, label_stats = standardize(train_df, val_df, None)
    print(f"train {len(train_df)} rows, val {len(val_df)} rows; "
          f"label mean/std {label_stats[0]:.4f}/{label_stats[1]:.4f}")

    if getattr(cfg.data, "dci_k", None) is not None:
        print(f"Using DCI coupling to build directed edges (dci_k={cfg.data.dci_k})")
    if getattr(cfg.data, "dci_threshold", None) is not None:
        print(f"Using DCI coupling to build directed edges (dci_threshold={cfg.data.dci_threshold})")

    dataset_train = Tem1BetaLactamaseDataset(dms_data=train_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours, radius=getattr(cfg.data, "radius", None), dci_k=getattr(cfg.data, "dci_k", None), dci_threshold=getattr(cfg.data, "dci_threshold", None), dci_add_spatial=getattr(cfg.data, "dci_add_spatial", False),
                                              aa_features=getattr(cfg.data, "aa_features", "aaindex8"))
    train_loader = DataLoader(dataset_train, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle,
                              num_workers=cfg.data.num_workers, persistent_workers=cfg.data.num_workers > 0)

    dataset_val= Tem1BetaLactamaseDataset(dms_data=val_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours, radius=getattr(cfg.data, "radius", None), dci_k=getattr(cfg.data, "dci_k", None), dci_threshold=getattr(cfg.data, "dci_threshold", None), dci_add_spatial=getattr(cfg.data, "dci_add_spatial", False),
                                            aa_features=getattr(cfg.data, "aa_features", "aaindex8"))
    val_loader = DataLoader(dataset_val, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, persistent_workers=cfg.data.num_workers > 0)

    # take the input dims from the data, not the config if mismatched.
    sample = dataset_train[0]
    node_in, edge_in = int(sample.node_features.shape[1]), int(sample.distance_features.shape[1])
    if node_in != cfg.model.node_in_channels or edge_in != cfg.model.edge_features_dim:
        print(f"note: config says node_in={cfg.model.node_in_channels} "
              f"edge_in={cfg.model.edge_features_dim}, data says node_in={node_in} "
              f"edge_in={edge_in}; using the data")

    model_gnn = Tem1BetaGNN(
        node_in_channels=node_in,
        edge_features_dim=edge_in,
        hidden_channels=cfg.model.hidden_channels,
        reg_hidden_channels=cfg.model.reg_hidden_channels,
        mp_layers=cfg.model.mp_layers,
        head_layers=cfg.model.head_layers,
        dropout=cfg.model.dropout,
        # mp_rounds is the number of propagate calls (receptive field); 
        # mp_layers above is the depth of the MLP inside one of them. Default 1 reproduces the original model.
        encoder_rounds=getattr(cfg.model, "encoder_rounds", 1),
        decoder_rounds=getattr(cfg.model, "decoder_rounds", 1),
    )
    print(f"model params: {sum(p.numel() for p in model_gnn.parameters())/1000:.0f}k "
          f"(encoder_rounds={model_gnn.encoder_rounds}, decoder_rounds={model_gnn.decoder_rounds})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(cfg.logging.output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model_gnn.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    scheduler = build_scheduler(cfg, optimizer, cfg.train.epochs)
    print(f"scheduler: {getattr(cfg.train, 'scheduler', 'plateau')}")

    train(model=model_gnn,
          num_epochs=cfg.train.epochs,
          train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=torch.nn.MSELoss(reduction='mean'),
            device=device,
            output_dir=cfg.logging.output_dir,
            scheduler=scheduler,
            grad_clip=getattr(cfg.train, "grad_clip", 1.0),
            label_stats=label_stats,
            # only meaningful when both groups are present; a singles-only run leaves it off
            group_weights=(group_loss_weights(train_df, device)
                           if getattr(cfg.train, "group_balance", False) and train_df['Single'].nunique() > 1
                           else None),
            early_stopping=None if args.no_early_stopping else EarlyStopping(patience=getattr(cfg.train, "patience", 20), delta=0.0001))
