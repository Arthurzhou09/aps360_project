import json
import numpy as np
import torch
import pandas as pd


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
