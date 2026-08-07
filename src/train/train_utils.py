import math
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

    Preferred over min_max_normalize for this data. TEM-1 fitness is bimodal (good vs bad).
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

# Claude did below
class IgnoreMetricScheduler:
    """
    Adapter so an epoch-based scheduler survives a training loop that calls
    `scheduler.step(val_loss)` - a ReduceLROnPlateau signature. Both run.py's and
    baseline.py's loops do this.

    Handing a LambdaLR to that call does not raise. Torch reads the positional argument as
    the epoch index, so the schedule ends up indexed by the validation loss VALUE. Measured
    against a real curve, a warmup+cosine schedule that should step

        3.3e-4 -> 6.7e-4 -> 1.0e-3 -> 1.0e-3 -> 9.98e-4

    instead produces

        3.3e-4 -> 5.1e-4 -> 5.1e-4 -> 5.4e-4 -> 5.2e-4

    - no warmup, no decay, silently, forever. This wrapper drops the metric and steps by
    epoch, which is what the cosine lr_lambda was written to do.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step(self, metric=None):
        self.scheduler.step()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)


def build_scheduler(cfg, optimizer, epochs): # this is claude
    """
    Build the LR schedule named by cfg.train.scheduler, so the GNN and the MLP baseline can
    be driven by the same field of the same config file. A comparison between two models
    trained under different LR schedules measures the schedules as much as the models.

    args:
        cfg: Config with a .train section (scheduler, warmup_epochs, min_lr_factor)
        optimizer: optimizer to schedule
        epochs: total planned epochs, needed to size the cosine decay
    returns:
        scheduler with a .step(metric) signature, or None
    """
    kind = getattr(cfg.train, "scheduler", "plateau")

    if kind == "none":
        return None

    if kind == "plateau":
        # note this reduces on val MSE while epochs are SELECTED on val Spearman; the two
        # disagree on this target, so which one to use is a real choice, not a formality
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    if kind == "cosine":
        warmup = getattr(cfg.train, "warmup_epochs", 3)
        min_factor = getattr(cfg.train, "min_lr_factor", 0.05)

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

        return IgnoreMetricScheduler(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda))

    raise ValueError(f"unknown train.scheduler {kind!r}; expected 'cosine', 'plateau' or 'none'")
