import argparse
import json
import pandas as pd
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
import sys

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.tem_beta import MLPDataset
from data.data_utils import load_cif_structure, parse_structure
from data.split import split_by_structural_position
from model.mlp import MLP
from train.train_utils import EarlyStopping, standardize, set_seed


def train(model, num_epochs, train_loader, val_loader, optimizer, criterion, device, output_dir, early_stopping=None):
    """
    trainig loop. use "mean" on mse
    """
    model.to(device)
    train_m = []
    val_m = []
    best_val_loss =None

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for b_i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output.squeeze(-1), y) #check: ok to have mean or sum before back prop?
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
        mean_loss = total_loss / total_samples
        train_m.append(mean_loss)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {mean_loss:.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                x,y = batch
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred.squeeze(-1), y)
                val_loss += loss.item() * y.size(0)
                val_samples += y.size(0)
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
            torch.save({'model_state': model.state_dict(), 'config': {'input_dim': model.input_dim, 'bottleneck_hidden_dim': model.bottleneck_hidden_dim}}, os.path.join(output_dir, "best_model.pt"))
            print(f"new model saved (val_loss={val_loss:.6f})")
    
    torch.save({'model_state': model.state_dict(), 'config': {'input_dim': model.input_dim, 'bottleneck_hidden_dim': model.bottleneck_hidden_dim}}, os.path.join(output_dir, "final_model.pt"))
    np.save(os.path.join(output_dir, "train_loss.npy"), train_m)
    np.save(os.path.join(output_dir, "val_loss.npy"), val_m)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the dms directory.")
    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")

    #logging
    parser.add_argument("--output_dir", type=str, default="./src/train/base/output", help="Directory to save the trained model and logs.")

    # data
    parser.add_argument("--train_size", type=float, default=0.8, help="Fraction of data to use for training.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of data to use for validation.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")

    # training
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer.")
    parser.add_argument("--seed", type=int, default=1012, help="Random seed for reproducibility.")
    parser.add_argument("--use_early_stopping", action="store_true", help="Whether to use early stopping during training.")

    #model
    parser.add_argument("--input_dim", type=int, default=129, help="Number of input channels for node features.") # 2 mut sites * (2*20 one-hot wt/mut identity + 3*8 AAindex wt/mut/delta props) + has_site_2 flag
    parser.add_argument("--bottleneck_hidden_dim", type=int, default=64, help="number of final hidden channels")
    
    args = parser.parse_args()
    
    ### Data Loading ###
    dms = pd.read_csv(args.processed_dms + "/dms_processed.csv")

    # same split and label scaling as the GNN (src/train/gnn/run.py) - the MLP is only a
    # meaningful baseline if it is held out on the same residues
    set_seed(args.seed)
    pdb_dir = os.path.dirname(args.processed_dms.rstrip("/\\"))
    wt_sequence, _ = parse_structure(load_cif_structure(os.path.join(pdb_dir, f"{args.pdb_id}.cif"), args.pdb_id))
    train_df, val_df, _ = split_by_structural_position(dms, wt_sequence, train_frac=args.train_size, val_frac=args.val_size, seed=args.seed)
    train_df, val_df, _, _ = standardize(train_df, val_df, None)

    dataset_train = MLPDataset(dms_data=train_df, pdb_id=args.pdb_id,)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    dataset_val= MLPDataset(dms_data=val_df, pdb_id=args.pdb_id,)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)  

    model_mlp = MLP(
        input_dim = args.input_dim,
        bottleneck_hidden_dim = args.bottleneck_hidden_dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(model=model_mlp, 
          num_epochs=args.epochs, 
          train_loader=train_loader,
            val_loader=val_loader, 
            optimizer=torch.optim.AdamW(model_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay),
            criterion=torch.nn.MSELoss(reduction='mean'),
            device=device,
            output_dir=args.output_dir,
            early_stopping=EarlyStopping(patience=10, delta=0.000001) if args.use_early_stopping else None
    )