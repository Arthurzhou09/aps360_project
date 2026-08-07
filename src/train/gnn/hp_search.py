"""
Small random hyperparameter search for the supervised GNN.

    python src/train/gnn/hp_search.py --trials 12
    python src/train/gnn/hp_search.py --trials 20 --config configs/gnn_base_enc.json
    python src/train/gnn/hp_search.py --trials 12 --dry_run     # print the sampled trials only

A config (default configs/gnn_base_pca19.json) is the TEMPLATE: every field is inherited,
and only the keys in SEARCH_SPACE are resampled per trial. Trial 0 is always the template
untouched, so every result has a baseline to be read against.

Two stages, because val is only 27 held-out residues (487 rows) and a single-seed score on
that is noisy enough to rank a mediocre config first:

    1. search  - `trials` random configs, one seed each. Cheap and wide.
    2. confirm - the top `confirm_top` configs re-run on `confirm_seeds` seeds total,
                 ranked on the MEAN. A config that only won its first seed drops out here.

A seed reseeds the SPLIT as well as model init, so each seed is a different held-out set of
residues. That is deliberate - it scores a config on generalising across splits rather than
on fitting one lucky val set - but it does mean per-seed scores are not directly comparable
and the spread is wide. Observed on this data: one config scored 0.730 and 0.562 on two
seeds. Treat any single-seed number, including a leader, as provisional.

Selection is on val balanced Spearman (train.train's return value), the same quantity the
training loop selects epochs on - not val MSE, which is dominated by the high-fitness tail.

Everything runs in-process and reuses train.gnn.run.train, so this stays honest to what
run.py actually does. Splits and datasets are cached across trials by the parameters that
affect them.

Outputs, under --output_dir:
    results.csv        one row per (trial, seed), appended as it goes
    summary.csv        per-trial aggregate, ranked
    best_config.json   the winning config, runnable with run.py --config
    trial_XX_seedYYYY/ per-run checkpoints, curves and train.log
"""

import argparse
import contextlib
import copy
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(REPO, "src"))

from data.data_utils import load_cif_structure, parse_structure, load_dms
from data.split import split_by_structural_position
from data.tem_beta import Tem1BetaLactamaseDataset
from model.gnn import Tem1BetaGNN
from train.train_utils import Config, EarlyStopping, build_scheduler, set_seed, standardize
from train.gnn.run import group_loss_weights, train


SEARCH_SPACE = {
    "train.lr":              ("loguniform", 5e-5, 1e-3),
    "train.weight_decay":    ("loguniform", 1e-5, 1e-1),
    "train.scheduler":       ("choice", ["cosine", "plateau"]),
    "model.dropout":         ("choice", [0.1, 0.2, 0.3, 0.4, 0.5]),
    "model.hidden_channels": ("choice", [32, 64, 96]),
    "model.mp_layers":       ("choice", [2, 3]),
    "model.encoder_rounds":  ("choice", [1, 2, 3]),
    "model.decoder_rounds":  ("choice", [1, 2]),
}

_SPLIT_CACHE = {}
_DATASET_CACHE = {}


def _describe(overrides):
    """Compact one-line rendering of a trial's overrides, for the console tables."""
    if not overrides:
        return "template"
    return "  ".join(f"{k.split('.')[1]}={v:.2e}" if isinstance(v, float) else f"{k.split('.')[1]}={v}"
                     for k, v in overrides.items())


def sample_overrides(rng, space):
    """
    Draw one config's worth of overrides from the search space.

    args:
        rng: numpy Generator
        space: {"section.key": ("choice", [...]) | ("loguniform", lo, hi)}
    returns:
        overrides: {"section.key": value}, JSON-safe scalars
    """
    overrides = {}
    for key, spec in space.items():
        if spec[0] == "choice":
            choice = spec[1][int(rng.integers(len(spec[1])))]
            overrides[key] = choice
        elif spec[0] == "loguniform":
            lo, hi = spec[1], spec[2]
            overrides[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            raise ValueError(f"unknown spec {spec[0]!r} for {key}")
    return overrides


def apply_overrides(template, overrides):
    """
    Copy the template config dict and set the dotted keys in it.

    args:
        template: config as a plain dict
        overrides: {"section.key": value}
    returns:
        config dict
    """
    out = copy.deepcopy(template)
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        out.setdefault(section, {})[key] = value
    return out


def get_datasets(cfg, seed, args, wt_sequence, dms):
    """
    Split, standardize and build the train/val datasets, memoized on everything that
    actually changes them. 

    returns:
        dataset_train, dataset_val, train_df, label_stats
    """
    split_key = (seed, cfg.data.train_size, cfg.data.val_size, getattr(cfg.data, "n_blocks", None))
    if split_key not in _SPLIT_CACHE:
        train_df, val_df, _ = split_by_structural_position(
            dms, wt_sequence, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size,
            seed=seed, n_blocks=getattr(cfg.data, "n_blocks", None))
        train_df, val_df, _, label_stats = standardize(train_df, val_df, None)
        _SPLIT_CACHE[split_key] = (train_df, val_df, label_stats)
    train_df, val_df, label_stats = _SPLIT_CACHE[split_key]

    data_key = split_key + (
        getattr(cfg.data, "aa_features", "aaindex8"), getattr(cfg.data, "radius", None),
        getattr(cfg.data, "dci_k", None), getattr(cfg.data, "dci_threshold", None),
        getattr(cfg.data, "dci_add_spatial", False), cfg.data.neighbours, args.directed)
    if data_key not in _DATASET_CACHE:
        common = dict(pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours,
                      radius=getattr(cfg.data, "radius", None), dci_k=getattr(cfg.data, "dci_k", None),
                      dci_threshold=getattr(cfg.data, "dci_threshold", None),
                      dci_add_spatial=getattr(cfg.data, "dci_add_spatial", False),
                      aa_features=getattr(cfg.data, "aa_features", "aaindex8"))
        _DATASET_CACHE[data_key] = (Tem1BetaLactamaseDataset(dms_data=train_df, **common),
                                    Tem1BetaLactamaseDataset(dms_data=val_df, **common))
    dataset_train, dataset_val = _DATASET_CACHE[data_key]

    return dataset_train, dataset_val, train_df, label_stats


def run_trial(cfg, seed, args, wt_sequence, dms, trial_dir, device):
    """
    Train one config at one seed and return its scalar results.

    returns:
        record: dict of metrics for this run
    """
    os.makedirs(trial_dir, exist_ok=True)
    set_seed(seed)

    dataset_train, dataset_val, train_df, label_stats = get_datasets(cfg, seed, args, wt_sequence, dms)

    train_loader = DataLoader(dataset_train, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle,
                              num_workers=cfg.data.num_workers, persistent_workers=cfg.data.num_workers > 0)
    val_loader = DataLoader(dataset_val, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, persistent_workers=cfg.data.num_workers > 0)

    # make sure input dims are never wrong
    sample = dataset_train[0]
    node_in = int(sample.node_features.shape[1])
    edge_in = int(sample.distance_features.shape[1])
    if node_in != cfg.model.node_in_channels or edge_in != cfg.model.edge_features_dim:
        print(f"  note: config says node_in={cfg.model.node_in_channels} edge_in={cfg.model.edge_features_dim}, "
              f"data says node_in={node_in} edge_in={edge_in}; using the data")

    model = Tem1BetaGNN(
        node_in_channels=node_in,
        edge_features_dim=edge_in,
        hidden_channels=cfg.model.hidden_channels,
        reg_hidden_channels=cfg.model.reg_hidden_channels,
        mp_layers=cfg.model.mp_layers,
        head_layers=cfg.model.head_layers,
        dropout=cfg.model.dropout,
        encoder_rounds=getattr(cfg.model, "encoder_rounds", 1),
        decoder_rounds=getattr(cfg.model, "decoder_rounds", 1),
    )
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(cfg, optimizer, cfg.train.epochs)

    group_weights = (group_loss_weights(train_df, device)
                     if getattr(cfg.train, "group_balance", False) and train_df['Single'].nunique() > 1
                     else None)

    started = time.time()
    log_path = os.path.join(trial_dir, "train.log")
    with open(log_path, "w") as log:
        with contextlib.redirect_stdout(log):
            best_score, best_epoch = train(
                model=model,
                num_epochs=cfg.train.epochs,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=torch.nn.MSELoss(reduction='mean'),
                device=device,
                output_dir=trial_dir,
                scheduler=scheduler,
                grad_clip=getattr(cfg.train, "grad_clip", 1.0),
                label_stats=label_stats,
                group_weights=group_weights,
                early_stopping=None if args.no_early_stopping else EarlyStopping(
                    patience=getattr(cfg.train, "patience", 20), delta=0.0001),
            )
    elapsed = time.time() - started

    # train() writes these
    train_curve = np.load(os.path.join(trial_dir, "train_loss.npy"))
    val_curve = np.load(os.path.join(trial_dir, "val_loss.npy"))

    return {
        "seed": seed,
        "val_balanced_spearman": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(train_curve)),
        "train_loss_at_best": float(train_curve[best_epoch]),
        "val_loss_at_best": float(val_curve[best_epoch]),
        "overfit_gap": float(val_curve[best_epoch] - train_curve[best_epoch]),
        "final_gap": float(val_curve[-1] - train_curve[-1]),
        "n_params": n_params,
        "seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=os.path.join(REPO, "configs", "gnn_base_pca19.json"),
                        help="Template config. Every field is inherited; only SEARCH_SPACE keys are resampled.")
    parser.add_argument("--processed_dms", type=str, default=os.path.join(REPO, "src", "data", "processed", "single"),
                        help="Path to the dms directory (the cif is read from its parent).")
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO, "src", "train", "gnn", "hp_search"))
    parser.add_argument("--trials", type=int, default=12, help="Random configs to try, including the template at trial 0.")
    parser.add_argument("--seed", type=int, default=1012, help="Seed for the data split and for trial 0.")
    parser.add_argument("--search_seed", type=int, default=0, help="Seed for sampling the search space itself.")
    parser.add_argument("--confirm_top", type=int, default=3, help="Re-run this many leading configs on more seeds. 0 disables.")
    parser.add_argument("--confirm_seeds", type=int, default=3, help="Total seeds each confirmed config is scored on.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs for every trial.")
    parser.add_argument("--pdb_id", type=str, default="1BTL")
    parser.add_argument("--directed", action="store_true", help="Directed spatial edges. Off by default, matching run.py.")
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print the sampled trials and exit without training.")
    args = parser.parse_args()

    with open(args.config) as f:
        template = json.load(f)
    if args.epochs is not None:
        template.setdefault("train", {})["epochs"] = args.epochs
    template.setdefault("train", {}).setdefault("scheduler", "plateau")  # what run.py currently does

    # trial 0 is the template verbatim, so the search always has an anchor to be read against
    rng = np.random.default_rng(args.search_seed)
    trials = [{}] + [sample_overrides(rng, SEARCH_SPACE) for _ in range(max(0, args.trials - 1))]

    print(f"template : {args.config}")
    print(f"trials   : {len(trials)} (trial 0 = template unchanged)")
    print(f"searching: {', '.join(SEARCH_SPACE)}")
    for i, ov in enumerate(trials):
        print(f"  [{i:2d}] {_describe(ov)}")
    if args.dry_run:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device   : {device}\noutput   : {args.output_dir}\n")

    pdb_dir = os.path.dirname(args.processed_dms.rstrip("/\\"))
    wt_sequence, _ = parse_structure(
        load_cif_structure(os.path.join(pdb_dir, f"{args.pdb_id}.cif"), args.pdb_id))
    dms = load_dms(args.processed_dms, wt_sequence)

    results_path = os.path.join(args.output_dir, "results.csv")
    records = []

    def score(trial_id, overrides, seed, stage):
        cfg_dict = apply_overrides(template, overrides)
        trial_dir = os.path.join(args.output_dir, f"trial_{trial_id:02d}_seed{seed}")
        started = time.time()
        record = run_trial(Config(cfg_dict), seed, args, wt_sequence, dms, trial_dir, device)
        record.update({"trial": trial_id, "stage": stage, "config": json.dumps(overrides, sort_keys=True)})
        records.append(record)
        pd.DataFrame(records).to_csv(results_path, index=False)
        print(f"  [{trial_id:2d}] seed {seed}  spearman {record['val_balanced_spearman']:.4f}  "
              f"epoch {record['best_epoch']:>2d}/{record['epochs_run']:<2d}  "
              f"gap {record['overfit_gap']:+.3f}  {record['n_params']/1000:.0f}k  "
              f"{time.time() - started:.0f}s", flush=True)
        return record

    print("=" * 96)
    print("STAGE 1  search")
    print("=" * 96)
    for trial_id, overrides in enumerate(trials):
        score(trial_id, overrides, args.seed, "search")

    frame = pd.DataFrame(records)
    ranking = (frame.groupby("trial")["val_balanced_spearman"].mean()
               .sort_values(ascending=False).index.tolist())

    # run on three different seeds to get more realiable score for top three initial models
    if args.confirm_top > 0 and args.confirm_seeds > 1:
        extra_seeds = [args.seed + 1000 * i for i in range(1, args.confirm_seeds)]
        print("\n" + "=" * 96)
        print(f"STAGE 2  confirm top {min(args.confirm_top, len(ranking))} on seeds {extra_seeds}")
        print("=" * 96)
        for trial_id in ranking[:args.confirm_top]:
            for seed in extra_seeds:
                score(trial_id, trials[trial_id], seed, "confirm")

    frame = pd.DataFrame(records)
    summary = (frame.groupby("trial")
               .agg(n_seeds=("seed", "count"),
                    spearman_mean=("val_balanced_spearman", "mean"),
                    spearman_std=("val_balanced_spearman", "std"),
                    spearman_min=("val_balanced_spearman", "min"),
                    best_epoch=("best_epoch", "mean"),
                    overfit_gap=("overfit_gap", "mean"),
                    n_params=("n_params", "first"),
                    config=("config", "first"))
               .reset_index())
    # the score on the seed EVERY trial ran
    base = frame[frame.seed == args.seed].set_index("trial")["val_balanced_spearman"]
    summary["spearman_base_seed"] = summary["trial"].map(base)

    # confirmed trials rank first, then by mean.
    summary = summary.sort_values(["n_seeds", "spearman_mean"], ascending=[False, False]).reset_index(drop=True)
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)

    per_seed = frame[frame.trial.isin(frame[frame.stage == "confirm"].trial.unique())]
    print("\n" + "=" * 96)
    print("RESULTS  (confirmed trials first, then by mean; base-seed column is comparable across all rows)")
    print("=" * 96)
    if len(per_seed):
        means = per_seed.groupby("seed")["val_balanced_spearman"].mean()
        print("split difficulty - mean over confirmed trials, per seed: "
              + "  ".join(f"{s}:{v:.4f}" for s, v in means.items()))
    print(f"{'trial':>5} {'seeds':>5} {'spearman':>17} {'worst':>7} {'seed'+str(args.seed):>9} "
          f"{'epoch':>6} {'gap':>7} {'params':>8}  config")
    for _, r in summary.iterrows():
        std = "" if pd.isna(r.spearman_std) else f" +/-{r.spearman_std:.3f}"
        print(f"{int(r.trial):>5} {int(r.n_seeds):>5} {r.spearman_mean:>10.4f}{std:>7} {r.spearman_min:>7.4f} "
              f"{r.spearman_base_seed:>9.4f} {r.best_epoch:>6.1f} {r.overfit_gap:>+7.3f} "
              f"{r.n_params/1000:>7.0f}k  {_describe(json.loads(r.config))}")

    best = summary.iloc[0]
    best_config = apply_overrides(template, json.loads(best.config))
    best_config.setdefault("logging", {})["output_dir"] = "./src/train/gnn/output_pca19_hp"
    best_path = os.path.join(args.output_dir, "best_config.json")
    with open(best_path, "w") as f:
        json.dump(best_config, f, indent=2)

    baseline = summary.loc[summary.trial == 0, "spearman_mean"]
    print(f"\nbest: trial {int(best.trial)}  val balanced spearman {best.spearman_mean:.4f}"
          + (f"  (template trial 0: {baseline.iloc[0]:.4f})" if len(baseline) else ""))
    print(f"wrote {best_path}")
    print(f"      {os.path.join(args.output_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
