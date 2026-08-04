"""
for the homolog model
"""

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
from data.homolog import HomologMaskedDataset
from data.split import split_by_cluster, split_by_homolog
from model.gnn import Tem1MaskedResidueGNN
from train.train_utils import EarlyStopping, load_config, set_seed


def _checkpoint(model):
    return {
        'model_state': model.state_dict(),
        'config': {
            'node_in_channels': model.node_in_channels,
            'edge_features_dim': model.edge_features_dim,
            'hidden_channels': model.hidden_channels,
            'head_hidden_channels': model.head_hidden_channels,
            'mp_layers': model.mp_layers,
            'head_layers': model.head_layers,
            'dropout': model.dropout,
            'encoder_rounds': model.encoder_rounds,
            'decoder_rounds': model.decoder_rounds,
        },
    }


def evaluate(model, loader, criterion, device):
    """
    Mean per-masked-position cross-entropy and top-1 recovery accuracy.

    Averaging is weighted by the number of masked positions in each batch, not by the
    number of graphs: with mask_ratio masking, graphs no longer carry exactly one label
    each, so a per-graph weighting would silently reweight long homologs.
    """
    model.eval()
    total_loss, total_masked, total_correct = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.node_features, batch.edge_index, batch.distance_features)
            mask = batch.labels != -100
            n_masked = int(mask.sum())
            loss = criterion(logits, batch.labels)
            total_loss += loss.item() * n_masked
            total_masked += n_masked
            total_correct += int((logits[mask].argmax(dim=-1) == batch.labels[mask]).sum())
    mean_nll = total_loss / total_masked
    return mean_nll, total_correct / total_masked


def train(model, num_epochs, train_loader, val_loader, optimizer, criterion, device, output_dir,
          early_stopping=None, scheduler=None, grad_clip=None):
    """
    training loop. cross-entropy over per-node AA logits (ignore_index=-100 skips every
    non-masked node, so this only ever scores the masked residues in each graph).
    """
    model.to(device)
    train_m = []
    val_m = []
    val_acc_m = []
    best_val_loss = None
    best_state = None
    best_epoch = None

    for epoch in range(num_epochs):
        # reseed the masking stream: without this every epoch masks identical positions
        # once num_workers > 0 (workers are respawned from a pickled dataset each epoch)
        train_loader.dataset.set_epoch(epoch)

        total_loss = 0.0
        total_masked = 0
        model.train()
        for b_i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch.node_features, batch.edge_index, batch.distance_features)
            loss = criterion(output, batch.labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            n_masked = int((batch.labels != -100).sum())
            total_loss += loss.item() * n_masked
            total_masked += n_masked
        mean_loss = total_loss / total_masked
        train_m.append(mean_loss)

        # validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_m.append(val_loss)
        val_acc_m.append(val_acc)

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs}  train {mean_loss:.4f}  val {val_loss:.4f}  "
              f"val_acc {val_acc:.4f}  val_ppl {math.exp(val_loss):.3f}  lr {lr_now:.2e}")

        # logging
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(_checkpoint(model), os.path.join(output_dir, "best_model.pt"))
            print(f"new model saved (val_loss={val_loss:.6f})")

        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

    torch.save(_checkpoint(model), os.path.join(output_dir, "final_model.pt"))
    np.save(os.path.join(output_dir, "train_loss.npy"), train_m)
    np.save(os.path.join(output_dir, "val_loss.npy"), val_m)
    np.save(os.path.join(output_dir, "val_accuracy.npy"), val_acc_m)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    return best_val_loss, best_epoch


if __name__ == "__main__":

    # node in channels: mask index (1) + AA one-hot (20) + AAindex props (8) +
    # AAindex property delta vs the structure's own residue (8) = 37. Same layout and
    # ordering as the first 37 channels Tem1BetaGNN sees.
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, required=True, help="Path to the directory containing homolog_processed.csv.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration JSON file.")

    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--directed", action="store_true", help="Whether to create directed edges in the graph. False if not specified.")
    parser.add_argument("--seed", type=int, default=1012, help="Random seed for reproducibility.")
    parser.add_argument("--split", type=str, default="homolog", choices=["cluster", "homolog"], help="Homolog split rule. Must match infer_pretrain.py.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override logging.output_dir from the config. Needed when running several splits off one config so they do not overwrite each other's checkpoints.")
    parser.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping (on by default).")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg.logging.output_dir = args.output_dir

    set_seed(args.seed)

    ### Data Loading ###
    homolog_df = pd.read_csv(args.processed + "/homolog_processed.csv")

    # cluster-aware split: keeps near-duplicate homologs (e.g. TEM-2, TEM-3, ...) together
    # in one split so val/test aren't contaminated by near-copies seen in train. Clustering
    # the full family takes a few minutes, so the assignment is cached and shared with
    # infer_pretrain.py rather than recomputed on every run.
    #
    # This defaults to "cluster" because infer_pretrain.py splits with split_by_cluster
    # unconditionally: pretraining on the split_by_homolog partition instead put 80% of the
    # homologs that inference calls "test" into the pretraining training set, so the
    # reported test accuracy was measured largely on trained-on sequences.
    cluster_cache_path = os.path.join(args.processed, "homolog_clusters_cache.csv")
    if args.split == "cluster":
        train_df, val_df, _ = split_by_cluster(homolog_df, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size, seed=args.seed,
                                               k=cfg.data.cluster_kmer_k, threshold=cfg.data.cluster_similarity_threshold,
                                               cache_path=cluster_cache_path)
    else:
        train_df, val_df, _ = split_by_homolog(homolog_df, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size, seed=args.seed)
    print(f"split={args.split}: train {len(train_df)} homologs, val {len(val_df)} homologs")

    mask_ratio = getattr(cfg.data, "mask_ratio", 0.15)
    dataset_train = HomologMaskedDataset(train_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours,
                                         seed=args.seed, mask_ratio=mask_ratio)
    train_loader = DataLoader(dataset_train, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle, num_workers=cfg.data.num_workers)

    dataset_val = HomologMaskedDataset(val_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours,
                                       seed=args.seed, mask_ratio=mask_ratio)
    val_loader = DataLoader(dataset_val, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model_gnn = Tem1MaskedResidueGNN(
        node_in_channels=cfg.model.node_in_channels,
        edge_features_dim=cfg.model.edge_features_dim,
        hidden_channels=cfg.model.hidden_channels,
        head_hidden_channels=cfg.model.head_hidden_channels,
        mp_layers=cfg.model.mp_layers,
        head_layers=cfg.model.head_layers,
        dropout=cfg.model.dropout,
        encoder_rounds=cfg.model.encoder_rounds,
        decoder_rounds=cfg.model.decoder_rounds,
    )
    print(f"model params: {sum(p.numel() for p in model_gnn.parameters())/1000:.0f}k "
          f"(encoder_rounds={model_gnn.encoder_rounds}, decoder_rounds={model_gnn.decoder_rounds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(cfg.logging.output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model_gnn.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    warmup_epochs = getattr(cfg.train, "warmup_epochs", 2)
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
          criterion=torch.nn.CrossEntropyLoss(ignore_index=-100),
          device=device,
          output_dir=cfg.logging.output_dir,
          scheduler=scheduler,
          grad_clip=getattr(cfg.train, "grad_clip", 1.0),
          early_stopping=None if args.no_early_stopping else EarlyStopping(patience=getattr(cfg.train, "patience", 10), delta=0.001))
