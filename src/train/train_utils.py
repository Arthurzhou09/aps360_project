import json
import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed: int):
    """
    Seed every RNG that affects a run: python, numpy and torch (CPU + CUDA).

    run.py previously only used its seed for the data split, so model init, dropout
    masks and DataLoader shuffling were unseeded - two runs of the same config could
    differ by more than the effect being measured.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config:
    """
    Configuration class with parameter as attributes. Instaniate with load_config
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



# taken from:" https://github.com/Bjarten/early-stopping-pytorch/blob/main/early_stopping_pytorch/early_stopping.py"
class EarlyStopping:
    """
    early stopping.

    returns:
        True or False to stop.
    """
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):

        if np.isnan(val_loss):
            print("Validation loss is NaN. ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


def standardize(train_set: pd.DataFrame, val_set: pd.DataFrame|None, test_set: pd.DataFrame|None,
                ) -> tuple[pd.DataFrame, pd.DataFrame|None, pd.DataFrame|None, list[float]]:
    """
    Z-score the fitness label using train-set statistics.

    Preferred over min_max_normalize for this data. TEM-1 fitness is bimodal (dead vs
    WT-like) with a thin tail out to 2.90 while the 90th percentile is ~1.05, so
    dividing by (max - min) squashes 90% of the labels into [0, 0.36]. That leaves the
    target variance at ~0.022, which (a) makes MSE numbers unreadable - a constant
    predictor already scores 0.020, so "loss 0.010" is really R^2 ~= 0.5 - and (b) shrinks
    the gradient scale, so lr and weight_decay end up tuned against an arbitrary
    constant. After z-scoring, a constant predictor scores 1.0 and MSE reads directly
    as 1 - R^2.

    args:
        train_set/val_set/test_set: DataFrames with a 'Fitness' column
    returns:
        standardized train, val, test and [train_mean, train_std] for inverting
    """
    x_train = train_set.copy()
    x_val = val_set.copy() if val_set is not None else None
    x_test = test_set.copy() if test_set is not None else None

    train_mean = float(train_set['Fitness'].mean())
    train_std = float(train_set['Fitness'].std())

    x_train['Fitness'] = (train_set['Fitness'] - train_mean) / train_std
    if val_set is not None:
        x_val['Fitness'] = (val_set['Fitness'] - train_mean) / train_std
    if test_set is not None:
        x_test['Fitness'] = (test_set['Fitness'] - train_mean) / train_std

    return x_train, x_val, x_test, [train_mean, train_std]




def min_max_normalize(train_set: pd.DataFrame, val_set: pd.DataFrame|None, test_set: pd.DataFrame|None,)-> tuple[pd.DataFrame, pd.DataFrame|None, pd.DataFrame|None, list[float]]:
    """
    Min-max normalize the input array fitness to the range [0, 1].
    args:
        arr: input array
    returns:
        normalized_arr train, val, test: min-max normalized array
        [train_min, train_max]: min and max values of the training set
    """
    x_train = train_set.copy()
    x_val = val_set.copy() if val_set is not None else None
    x_test = test_set.copy() if test_set is not None else None

    train_min = train_set['Fitness'].min()
    train_max = train_set['Fitness'].max()

    x_train['Fitness'] = (train_set['Fitness'] - train_min) / (train_max - train_min)
    if val_set is not None:
        x_val['Fitness'] = (val_set['Fitness'] - train_min) / (train_max - train_min)
    if test_set is not None:
        x_test['Fitness'] = (test_set['Fitness'] - train_min) / (train_max - train_min)

    return x_train, x_val, x_test, [train_min, train_max]


def safe_pearson(x, y) -> float:
    """
    Pearson correlation computed via elementwise numpy ops only (mean, sum, sqrt).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    return float((x_centered * y_centered).sum() / denom) if denom > 0 else float('nan')


def safe_spearman(x, y) -> float:
    """
    Spearman correlation: Pearson correlation of the rank-transformed values.
    """
    x_rank = pd.Series(x).rank().to_numpy()
    y_rank = pd.Series(y).rank().to_numpy()
    return safe_pearson(x_rank, y_rank)
