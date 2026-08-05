
import argparse
import copy
import json
import math
import pandas as pd
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
import sys

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.tem_beta import MLPDataset
from data.data_utils import load_cif_structure, parse_structure, load_dms
from data.split import split_by_structural_position
from model.mlp import MLP
from train.train_utils import (EarlyStopping, standardize, set_seed,
                               safe_pearson, safe_spearman)


DATA_DEFAULTS = {"train_size": 0.8, "val_size": 0.1, "batch_size": 32, "num_workers": 0,
                 "aa_features": "pca19", "n_blocks": None, "seed": 1012}

def resolve_data_settings(args):
    """
    Merge DATA_DEFAULTS <- --config <- explicit CLI flags into the data/split settings.

    The overridable CLI flags default to None precisely so "not passed" stays
    distinguishable from "passed the same value as the default"; that is what lets a config
    file win without silently overriding a deliberate flag.

    args:
        args: parsed argparse namespace
    returns:
        settings: dict with train_size, val_size, batch_size, num_workers, aa_features,
            n_blocks and seed
    """
    settings = dict(DATA_DEFAULTS)

    if args.config:
        with open(args.config) as f:
            gnn_cfg = json.load(f)
        for key in ("train_size", "val_size", "batch_size", "num_workers", "aa_features", "n_blocks"):
            if key in gnn_cfg.get("data", {}):
                settings[key] = gnn_cfg["data"][key]
        # the split is seeded from the config's top-level seed, so it must come from there too
        if "seed" in gnn_cfg:
            settings["seed"] = gnn_cfg["seed"]

    for key in ("train_size", "val_size", "batch_size", "num_workers", "aa_features", "n_blocks", "seed"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    return settings


def _checkpoint(model, label_stats):
    return {
        'model_state': model.state_dict(),
        'config': {'input_dim': model.input_dim, 'bottleneck_hidden_dim': model.bottleneck_hidden_dim},
        'label_stats': label_stats,
    }


def group_loss_weights(train_df, device):
    """
    Per-group loss weights giving singles and doubles equal influence. Same rule as
    src/train/gnn/run.py's group_loss_weights - the baseline has to be trained under the
    same objective as the GNN for the comparison to mean anything.
    """
    counts = {1: int((train_df['Single'] == 1).sum()), 0: int((train_df['Single'] == 0).sum())}
    total = sum(counts.values())
    weights = torch.ones(2, dtype=torch.float, device=device)
    for flag, count in counts.items():
        weights[flag] = (total / (2.0 * count)) if count else 0.0
    print(f"group loss weights: single {weights[1]:.3f} (n={counts[1]}), double {weights[0]:.3f} (n={counts[0]})")
    return weights


def evaluate(model, loader, criterion, device):
    """
    MSE plus rank/linear correlation, overall and per group, matching
    src/train/gnn/run.py's evaluate so the baseline and the GNN are read off the same
    numbers. See that function for why the combined Spearman alone is not enough: a
    predictor that only counts mutations already scores ~0.32 on the combined test split,
    and the MLP is handed that count directly as its has_site_2 input feature.
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    predictions, targets, groups = [], [], []
    with torch.no_grad():
        for x, y, g in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            groups.append(g.numpy().reshape(-1))
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
        metrics[f"spearman_{name}"] = safe_spearman(predictions[sub], targets[sub]) if sub.sum() > 1 else float("nan")
        metrics[f"n_{name}"] = int(sub.sum())
    present = [metrics[f"spearman_{n}"] for n in ("single", "double") if metrics[f"n_{n}"] > 1]
    metrics["balanced"] = float(np.mean(present)) if present else float("nan")
    return metrics


def train(model, num_epochs, train_loader, val_loader, optimizer, criterion, device, output_dir,
          early_stopping=None, scheduler=None, grad_clip=None, label_stats=None, group_weights=None):
    """
    trainig loop. use "mean" on mse

    Mirrors src/train/gnn/run.py: same per-epoch metrics, same selection on validation
    Spearman rather than MSE, same saved curves. The baseline is only interpretable next
    to the GNN if both are logged and selected the same way - picking the lowest-MSE epoch
    here and the best-ranking epoch there would compare two different things.
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
            x, y, g = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            if group_weights is None:
                loss = criterion(output.squeeze(-1), y) #check: ok to have mean or sum before back prop?
            else:
                # per-sample loss, then a weighted mean so singles and doubles count equally
                per_sample = torch.nn.functional.mse_loss(output.squeeze(-1), y, reduction='none')
                w = group_weights[g.to(device).view(-1)]
                loss = (per_sample * w).sum() / w.sum()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
        mean_loss = total_loss / total_samples
        train_m.append(mean_loss)

        # validation
        metrics = evaluate(model, val_loader, criterion, device)
        val_m.append(metrics["loss"])
        val_spearman_m.append(metrics["spearman"])
        val_balanced_m.append(metrics["balanced"])

        if scheduler is not None:
            scheduler.step()

        selection_score = metrics["balanced"] if not np.isnan(metrics["balanced"]) else metrics["spearman"]

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs}  train {mean_loss:.4f}  val {metrics['loss']:.4f}  "
              f"val_spearman {metrics['spearman']:.4f} (single {metrics['spearman_single']:.4f} "
              f"n={metrics['n_single']} | double {metrics['spearman_double']:.4f} n={metrics['n_double']} "
              f"| balanced {metrics['balanced']:.4f})  lr {lr_now:.2e}")

        # logging: keep the best-ranking epoch and hold its weights so training can be
        # rolled back to it instead of finishing on an overfit final epoch
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the dms directory.")
    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--config", type=str, default=None,
                        help="A GNN config (e.g. configs/gnn_base_pca19.json). The data and split settings "
                             "are read from it so the two runs cannot end up on different held-out residues "
                             "or different amino-acid features. CLI flags win over it.")

    #logging
    parser.add_argument("--output_dir", type=str, default="./src/train/base/output", help="Directory to save the trained model and logs.")

    parser.add_argument("--aa_features", type=str, default="pca19", choices=["aaindex8", "pca19"], help="Per-residue property set. Must match the GNN it is compared against.")
    parser.add_argument("--train_size", type=float, default=None, help="Fraction of data to use for training.")
    parser.add_argument("--val_size", type=float, default=None, help="Fraction of data to use for validation.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training and validation.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers for data loading.")
    parser.add_argument("--n_blocks", type=int, default=None, help="Contiguous residue blocks for the split. Must match the GNN run.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Taken from the config's top-level seed when unset.")

    # training: the baseline keeps its own optimisation settings
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")

    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience, in epochs without a val Spearman improvement.")
    parser.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping (on by default).")
    parser.add_argument("--no_group_balance", dest="group_balance", action="store_false", help="Disable equal single/double loss weighting (on by default, matching the GNN config).")

    #model
    parser.add_argument("--input_dim", type=int, default=None,
                        help="Number of input features. Derived from the dataset by default; pass only to assert a value.")
    parser.add_argument("--bottleneck_hidden_dim", type=int, default=64, help="number of final hidden channels")

    args = parser.parse_args()
    data_cfg = resolve_data_settings(args)
    args.seed = data_cfg["seed"]
    print(f"data/split from: {args.config or '(defaults)'}  seed {args.seed}  "
          f"aa_features {data_cfg['aa_features']}  train/val {data_cfg['train_size']}/{data_cfg['val_size']}  "
          f"n_blocks {data_cfg['n_blocks']}")

    ### Data Loading ###
    set_seed(args.seed)
    pdb_dir = os.path.dirname(args.processed_dms.rstrip("/\\"))
    wt_sequence, _ = parse_structure(load_cif_structure(os.path.join(pdb_dir, f"{args.pdb_id}.cif"), args.pdb_id))
    dms = load_dms(args.processed_dms, wt_sequence)
    train_df, val_df, _ = split_by_structural_position(
        dms, wt_sequence, train_frac=data_cfg["train_size"], val_frac=data_cfg["val_size"],
        seed=args.seed, n_blocks=data_cfg["n_blocks"])
    train_df, val_df, _, label_stats = standardize(train_df, val_df, None)
    print(f"train {len(train_df)} rows, val {len(val_df)} rows; "
          f"label mean/std {label_stats[0]:.4f}/{label_stats[1]:.4f}")

    dataset_train = MLPDataset(dms_data=train_df, pdb_id=args.pdb_id, aa_features=data_cfg["aa_features"])
    train_loader = DataLoader(dataset_train, batch_size=data_cfg["batch_size"], shuffle=True, num_workers=data_cfg["num_workers"])

    dataset_val= MLPDataset(dms_data=val_df, pdb_id=args.pdb_id, aa_features=data_cfg["aa_features"])
    val_loader = DataLoader(dataset_val, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"])

    # take the width from the data: 137 for aaindex8 but 203 for pca19's 19 components, and
    # the old hardcoded 137 default silently belonged to only one of them
    input_dim = dataset_train.feature_dim
    if args.input_dim is not None and args.input_dim != input_dim:
        raise ValueError(f"--input_dim {args.input_dim} does not match the {input_dim} features "
                         f"MLPDataset produces for aa_features={data_cfg['aa_features']!r}")

    model_mlp = MLP(
        input_dim = input_dim,
        bottleneck_hidden_dim = args.bottleneck_hidden_dim,
    )
    print(f"model params: {sum(p.numel() for p in model_mlp.parameters())/1000:.0f}k (input_dim={input_dim})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    train(model=model_mlp,
          num_epochs=args.epochs,
          train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=torch.nn.MSELoss(reduction='mean'),
            device=device,
            output_dir=args.output_dir,
            scheduler=None,
            grad_clip=args.grad_clip if args.grad_clip > 0 else None,
            label_stats=label_stats,
            # matches the GNN's train.group_balance; skipped when only one group is present
            group_weights=(group_loss_weights(train_df, device)
                           if args.group_balance and train_df['Single'].nunique() > 1 else None),
            early_stopping=None if args.no_early_stopping else EarlyStopping(patience=args.patience, delta=0.0)
    )