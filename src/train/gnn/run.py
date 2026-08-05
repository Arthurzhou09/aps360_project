import argparse
import json
import math
import pandas as pd
from torch_geometric.loader import DataLoader
import torch
import os
import numpy as np
import sys

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.tem_beta import Tem1BetaLactamaseDataset
from data.data_utils import load_cif_structure, parse_structure, load_dms
from data.split import split_by_structural_position
from model.gnn import Tem1BetaGNN
from train.train_utils import (EarlyStopping, standardize, load_config, set_seed,
                               safe_pearson, safe_spearman)


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
    Run the model over a loader and return MSE plus rank/linear correlation, both
    overall and separately for single and double mutants.

    The per-group split matters whenever singles and doubles are trained together.
    Double mutants are worse and more. Mean is what the training loop selects on.

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
    Per-group loss weights that give single and double mutants equal total influence.

    The combined set is 69% doubles / 31% singles, so an unweighted mean loss lets doubles
    contribute ~2.2x the gradient. 

    returns:
        weights: tensor indexed by the is_single flag (index 0 = double, 1 = single)
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

    Model selection is on validation Spearman rather than validation MSE: MSE on this
    target is dominated by the thin high-fitness tail, so the lowest-MSE epoch is not
    generally the best-ranking one, and ranking is what gets compared to the baseline.

    When singles and doubles are trained together, selection uses the group-BALANCED
    Spearman (mean of the within-single and within-double values) instead of the combined
    one, so an epoch is not rewarded for separating the two populations.
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

    # input channels is the number of edge attribute features (rbf number(8) * 16 (4*4 atomic positions) = 128)
    # node in channels: mutation index mask (1) + mutant identity (AAindex(8) + one-hot(20)) +
    # AAindex property delta (mutant - WT) at the mutated site(s), zero elsewhere (8) +
    # WT identity one-hot at the mutated site(s), zero elsewhere (20) = 1+28+8+20 = 57
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the dms directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration JSON file.")

    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--directed", action="store_true", help="Whether to create directed edges in the graph. False if not specified.")
    parser.add_argument("--seed", type=int, default=1012, help="Random seed for reproducibility.")

    parser.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping (on by default).")
    args = parser.parse_args()
    cfg= load_config(args.config)

    set_seed(args.seed)

    ### Data Loading ###
    # split on structural residue index, not the raw 'Ambler Index': the single and pair
    # experiments use different Experiment Sequences, so the same Ambler Index names a
    # different residue in each and the old split put 59% of "held-out" val residues in train.
    # processed_dms is <processed>/single or <processed>/both; the cif sits in <processed>
    pdb_dir = os.path.dirname(args.processed_dms.rstrip("/\\"))
    wt_sequence, _ = parse_structure(
        load_cif_structure(os.path.join(pdb_dir, f"{args.pdb_id}.cif"), args.pdb_id))

    # load_dms drops WT-to-itself rows,
    dms = load_dms(args.processed_dms, wt_sequence)

    train_df, val_df, _ = split_by_structural_position(
        dms, wt_sequence, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size, seed=args.seed,
        n_blocks=getattr(cfg.data, "n_blocks", None))
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

    model_gnn = Tem1BetaGNN(
        node_in_channels=cfg.model.node_in_channels,
        edge_features_dim=cfg.model.edge_features_dim,
        hidden_channels=cfg.model.hidden_channels,
        reg_hidden_channels=cfg.model.reg_hidden_channels,
        mp_layers=cfg.model.mp_layers,
        head_layers=cfg.model.head_layers,
        dropout=cfg.model.dropout,
        # mp_rounds is the number of propagate calls (receptive field); mp_layers above is
        # the depth of the MLP inside one of them. Default 1 reproduces the original model.
        encoder_rounds=getattr(cfg.model, "encoder_rounds", 1),
        decoder_rounds=getattr(cfg.model, "decoder_rounds", 1),
    )
    print(f"model params: {sum(p.numel() for p in model_gnn.parameters())/1000:.0f}k "
          f"(encoder_rounds={model_gnn.encoder_rounds}, decoder_rounds={model_gnn.decoder_rounds})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(cfg.logging.output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model_gnn.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # linear warmup then cosine decay: the previous constant lr let the model keep
    # descending into the training set long after val had bottomed out at epoch ~15
    warmup_epochs = getattr(cfg.train, "warmup_epochs", 3)
    min_lr_factor = getattr(cfg.train, "min_lr_factor", 0.05)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, cfg.train.epochs - warmup_epochs)
        return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,)
    #torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
