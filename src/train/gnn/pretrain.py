"""
for the homolog model
"""

import argparse
import json
import pandas as pd
from torch_geometric.loader import DataLoader
import torch
import os
import numpy as np
import sys

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.homolog import HomologMaskedDataset
from data.split import split_by_cluster
from model.gnn import Tem1MaskedResidueGNN
from train.train_utils import EarlyStopping, load_config


def train(model, num_epochs, train_loader, val_loader, optimizer, criterion, device, output_dir, early_stopping=None):
    """
    training loop. cross-entropy over per-node AA logits (ignore_index=-100 skips every
    non-masked node, so this only ever scores the one masked residue per graph).
    """
    model.to(device)
    train_m = []
    val_m = []
    best_val_loss = None

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for b_i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch.node_features, batch.edge_index, batch.distance_features)
            loss = criterion(output, batch.labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
        mean_loss = total_loss / total_samples
        train_m.append(mean_loss)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {mean_loss:.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.node_features, batch.edge_index, batch.distance_features)
                loss = criterion(pred, batch.labels)
                val_loss += loss.item() * batch.num_graphs
                val_samples += batch.num_graphs
        val_loss /= val_samples
        val_m.append(val_loss)
        print("Val loss:", val_loss)

        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

        # logging
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state': model.state_dict(),
                        "config": {"node_in_channels": model.node_in_channels,
                                   "edge_features_dim": model.edge_features_dim,
                                   "hidden_channels": model.hidden_channels,
                                   "head_hidden_channels": model.head_hidden_channels,
                                   "mp_layers": model.mp_layers,
                                   "head_layers": model.head_layers,
                                   "dropout": model.dropout
                                   }}
                       , os.path.join(output_dir, "best_model.pt"))
            print(f"new model saved (val_loss={val_loss:.6f})")

    torch.save({'model_state': model.state_dict(), 'config': {'node_in_channels': model.node_in_channels,
                                                               'edge_features_dim': model.edge_features_dim,
                                                               'hidden_channels': model.hidden_channels,
                                                               'head_hidden_channels': model.head_hidden_channels,
                                                                'mp_layers': model.mp_layers,
                                                                'head_layers': model.head_layers,
                                                                'dropout': model.dropout}},
                                                                os.path.join(output_dir, "final_model.pt"))
    np.save(os.path.join(output_dir, "train_loss.npy"), train_m)
    np.save(os.path.join(output_dir, "val_loss.npy"), val_m)

    return


if __name__ == "__main__":

    # node in channels: mask index (1) + AA one-hot (20) + AAindex props (8) = 29, same as Tem1BetaGNN
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, required=True, help="Path to the directory containing homolog_processed.csv.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration JSON file.")

    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--directed", action="store_true", help="Whether to create directed edges in the graph. False if not specified.")
    parser.add_argument("--seed", type=int, default=1012, help="Random seed for reproducibility.")

    parser.add_argument("--use_early_stopping", action="store_true", help="Whether to use early stopping during training. False if not specified.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    ### Data Loading ###
    homolog_df = pd.read_csv(args.processed + "/homolog_processed.csv")

    # cluster-aware split: keeps near-duplicate homologs (e.g. TEM-2, TEM-3, ...) together
    # in one split so val/test aren't contaminated by near-copies seen in train. Clustering
    # the full family takes a few minutes, so the assignment is cached and shared with
    # infer_pretrain.py rather than recomputed on every run.
    cluster_cache_path = os.path.join(args.processed, "homolog_clusters_cache.csv")
    train_df, val_df, _ = split_by_cluster(homolog_df, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size, seed=args.seed,
                                            k=cfg.data.cluster_kmer_k, threshold=cfg.data.cluster_similarity_threshold,
                                            cache_path=cluster_cache_path)

    dataset_train = HomologMaskedDataset(train_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours, seed=args.seed)
    train_loader = DataLoader(dataset_train, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle, num_workers=cfg.data.num_workers)

    dataset_val = HomologMaskedDataset(val_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours, seed=args.seed)
    val_loader = DataLoader(dataset_val, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model_gnn = Tem1MaskedResidueGNN(
        node_in_channels=cfg.model.node_in_channels,
        edge_features_dim=cfg.model.edge_features_dim,
        hidden_channels=cfg.model.hidden_channels,
        head_hidden_channels=cfg.model.head_hidden_channels,
        mp_layers=cfg.model.mp_layers,
        head_layers=cfg.model.head_layers,
        dropout=cfg.model.dropout
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(cfg.logging.output_dir, exist_ok=True)

    train(model=model_gnn,
          num_epochs=cfg.train.epochs,
          train_loader=train_loader,
          val_loader=val_loader,
          optimizer=torch.optim.AdamW(model_gnn.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay),
          criterion=torch.nn.CrossEntropyLoss(ignore_index=-100),
          device=device,
          output_dir=cfg.logging.output_dir,
          early_stopping=EarlyStopping(patience=20, delta=0.00001) if args.use_early_stopping else None)
