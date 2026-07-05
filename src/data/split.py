


import numpy as np
import pandas as pd
import re

def get_positions(code):
    """
    Extract mutatation residue positions from mutation code
    returns:
        list of residue positions (Ambler)
    """
    return [int(x) for x in re.findall(r'_([0-9]+)_', code)]

def split_by_position(
    df,
    train_frac=0.8,
    val_frac=0.1,
    seed=1012,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dms data by mutation position. All mutations in the same residue position will be
    in the same split. TODO: Adress double mutants robustly.

    returns:
        train_df, val_df, test_df
    """
    df = df.copy()

    df["positions"] = df["Ambler Index"]

    # unique residue positions as a set
    all_positions = sorted(
        {p for lst in df.positions for p in lst}
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(all_positions)
    n = len(all_positions)

    train_pos = set(all_positions[:int(train_frac*n)])
    val_pos = set(all_positions[int(train_frac*n):
                                int((train_frac+val_frac)*n)])
    test_pos = set(all_positions[int((train_frac+val_frac)*n):])    

    # check if mutated positions are contained within their repspective sets
    train_mask = df.positions.apply(lambda x: set(x) <= train_pos)
    val_mask   = df.positions.apply(lambda x: set(x) <= val_pos)
    test_mask  = df.positions.apply(lambda x: set(x) <= test_pos)

    train = df[train_mask].reset_index(drop=True)
    val = df[val_mask].reset_index(drop=True)
    test = df[test_mask].reset_index(drop=True)

    return train, val, test

