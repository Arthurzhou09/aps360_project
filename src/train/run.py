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
from data.split import split_by_position
from model.gnn import Tem1BetaGNN


class Config:
    """
    Configuration class with parameter as attributes.
    """
    def __init__(self, d):
        for k, v in d.items():
            # recursive convert until root of dicionary.
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)

def load_config(path: str) -> Config:
    """
    Load a JSON training configuration.
    args:
        path: Path to the JSON configuration file.
    returns:
        config: Config object with parameters as attributes.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return Config(data)



def train(model, num_epochs, train_loader, val_loader, optimizer, criterion, device, output_dir, log_every=None):
    """ 
    trainig loop. use "sum' on mse
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
            output = model(batch)
            loss = criterion(output, batch.fitness)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
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
                pred = model(batch)
                loss = criterion(pred, batch.fitness)
                val_loss += loss.item()
                val_samples += batch.num_graphs
        val_loss /= val_samples
        val_m.append(val_loss)
        print("Val loss:", val_loss)

        # logging
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"new model saved (val_loss={val_loss:.6f})")
    
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    np.save(os.path.join(output_dir, "train_loss.npy"), train_m)
    np.save(os.path.join(output_dir, "val_loss.npy"), val_m)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the dms directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration JSON file.")

    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--directed", action="store_true", help="Whether to create directed edges in the graph. False if not specified.")

    args = parser.parse_args()
    cfg= load_config(args.config)

    ### Data Loading ###
    dms = pd.read_csv(args.processed_dms + "/dms_processed.csv")
    ex_seq =dms['Experiment Sequence'].iloc[0]

    train_df, val_df, test_df = split_by_position(dms, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size)

    dataset_train = Tem1BetaLactamaseDataset(dms_data=train_df, pdb_id=args.pdb_id, experiment_wt_sequence =ex_seq, directed=args.directed, max_neighbours=cfg.data.neighbours)
    train_loader = DataLoader(dataset_train, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle, num_workers=cfg.data.num_workers)
    
    dataset_val= Tem1BetaLactamaseDataset(dms_data=val_df, pdb_id=args.pdb_id, experiment_wt_sequence =ex_seq, directed=args.directed, max_neighbours=cfg.data.neighbours)
    val_loader = DataLoader(dataset_val, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)  

    model_gnn = Tem1BetaGNN(
        in_channels=cfg.model.in_channels,
        hidden_channels=cfg.model.hidden_channels,
        reg_hidden_channels=cfg.model.reg_hidden_channels,
        mp_layers=cfg.model.mp_layers,
        head_layers=cfg.model.head_layers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model=model_gnn, 
          num_epochs=cfg.train.epochs, 
          train_loader=train_loader,
            val_loader=val_loader, 
            optimizer=torch.optim.AdamW(model_gnn.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay),
            criterion=torch.nn.MSELoss(reduction='sum'), 
            device=device,
            output_dir=cfg.logging.output_dir, 
            log_every=cfg.logging.log_every)
