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
from data.data_utils import load_cif_structure, parse_structure
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
        },
        # carried so inference can invert the label scaling without re-deriving it from
        # the split (which silently breaks if the split rule ever changes again)
        'label_stats': label_stats,
    }


def evaluate(model, loader, criterion, device):
    """
    Run the model over a loader and return MSE plus rank/linear correlation.

    Spearman is tracked because that - not MSE - is the metric the evolutionary PSSM
    baseline is scored on, so it is the number that decides whether this encoder is
    worth the cost of the self-supervised run.
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    predictions, targets = [], []
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
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    return (total_loss / total_samples,
            safe_spearman(predictions, targets),
            safe_pearson(predictions, targets))


def train(model, num_epochs, train_loader, val_loader, optimizer, criterion, device, output_dir,
          early_stopping=None, scheduler=None, grad_clip=None, label_stats=None):
    """
    trainig loop. use "mean" on mse

    Model selection is on validation Spearman rather than validation MSE: MSE on this
    target is dominated by the thin high-fitness tail, so the lowest-MSE epoch is not
    generally the best-ranking one, and ranking is what gets compared to the baseline.
    """
    model.to(device)
    train_m = []
    val_m = []
    val_spearman_m = []
    best_spearman = None
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
            loss = criterion(output.squeeze(-1), batch.fitness)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
        mean_loss = total_loss / total_samples
        train_m.append(mean_loss)

        # validation
        val_loss, val_spearman, val_pearson = evaluate(model, val_loader, criterion, device)
        val_m.append(val_loss)
        val_spearman_m.append(val_spearman)

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs}  train {mean_loss:.4f}  val {val_loss:.4f}  "
              f"val_spearman {val_spearman:.4f}  val_pearson {val_pearson:.4f}  lr {lr_now:.2e}")

        # logging: keep the best-ranking epoch, and hold its weights so training can be
        # rolled back to it at the end instead of finishing on an overfit final epoch
        if best_spearman is None or val_spearman > best_spearman:
            best_spearman = val_spearman
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(_checkpoint(model, label_stats), os.path.join(output_dir, "best_model.pt"))
            print(f"new model saved (val_spearman={val_spearman:.6f})")

        if early_stopping is not None:
            early_stopping(-val_spearman) # EarlyStopping minimizes, so feed it negated Spearman
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

    torch.save(_checkpoint(model, label_stats), os.path.join(output_dir, "final_model.pt"))
    np.save(os.path.join(output_dir, "train_loss.npy"), train_m)
    np.save(os.path.join(output_dir, "val_loss.npy"), val_m)
    np.save(os.path.join(output_dir, "val_spearman.npy"), val_spearman_m)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights from epoch {best_epoch} (val_spearman={best_spearman:.4f})")

    return best_spearman, best_epoch


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
    dms = pd.read_csv(args.processed_dms + "/dms_processed.csv")

    # split on structural residue index, not the raw 'Ambler Index': the single and pair
    # experiments use different Experiment Sequences, so the same Ambler Index names a
    # different residue in each and the old split put 59% of "held-out" val residues in train.
    # processed_dms is <processed>/single or <processed>/both; the cif sits in <processed>
    pdb_dir = os.path.dirname(args.processed_dms.rstrip("/\\"))
    wt_sequence, _ = parse_structure(
        load_cif_structure(os.path.join(pdb_dir, f"{args.pdb_id}.cif"), args.pdb_id))

    train_df, val_df, _ = split_by_structural_position(
        dms, wt_sequence, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size, seed=args.seed,
        n_blocks=getattr(cfg.data, "n_blocks", None))
    train_df, val_df, _, label_stats = standardize(train_df, val_df, None)
    print(f"train {len(train_df)} rows, val {len(val_df)} rows; "
          f"label mean/std {label_stats[0]:.4f}/{label_stats[1]:.4f}")

    dataset_train = Tem1BetaLactamaseDataset(dms_data=train_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours)
    train_loader = DataLoader(dataset_train, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle,
                              num_workers=cfg.data.num_workers, persistent_workers=cfg.data.num_workers > 0)

    dataset_val= Tem1BetaLactamaseDataset(dms_data=val_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours)
    val_loader = DataLoader(dataset_val, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, persistent_workers=cfg.data.num_workers > 0)

    model_gnn = Tem1BetaGNN(
        node_in_channels=cfg.model.node_in_channels,
        edge_features_dim=cfg.model.edge_features_dim,
        hidden_channels=cfg.model.hidden_channels,
        reg_hidden_channels=cfg.model.reg_hidden_channels,
        mp_layers=cfg.model.mp_layers,
        head_layers=cfg.model.head_layers,
        dropout=cfg.model.dropout
    )

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

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
            early_stopping=None if args.no_early_stopping else EarlyStopping(patience=getattr(cfg.train, "patience", 20), delta=0.0))
