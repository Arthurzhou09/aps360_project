"""
Run the two homolog-split pretraining variants back to back.

    python src/train/gnn/run_pretrain_splits.py

Runs pretrain.py twice, sequentially (the second starts only after the first exits), so
the two never contend for the GPU:

    1. --split homolog   (random homolog_ID assignment)
    2. --split cluster   (k-mer cluster assignment, keeps near-duplicates together)

Each run gets its own output_dir - output_pretrain_homolog / output_pretrain_cluster -
because both would otherwise write best_model.pt and val_loss.npy to the same
logging.output_dir and the second would silently overwrite the first.

Output is streamed live and also written to <output_dir>/train.log.
"""
import argparse
import os
import subprocess
import sys
import time

REPO = r'C:\Users\Arthur Zhou\GitHub\aps360_project'
PRETRAIN = os.path.join(REPO, "src", "train", "gnn", "pretrain.py")


def run_one(split, processed, config, output_dir, extra_args):
    """Run pretrain.py for one split, streaming stdout to the console and a log file."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    cmd = [sys.executable, "-u", PRETRAIN,
           "--processed", processed,
           "--config", config,
           "--split", split,
           "--output_dir", output_dir] + extra_args

    print("=" * 78, flush=True)
    print(f"RUN  split={split}  ->  {output_dir}", flush=True)
    print("  " + " ".join(cmd), flush=True)
    print("=" * 78, flush=True)

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        # line-buffered pipe + explicit flush so a long run is followed live rather than
        # appearing all at once when the process exits
        process = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        process.wait()

    elapsed = time.time() - start
    print(f"\n--- split={split} finished: exit={process.returncode}  {elapsed/60:.1f} min  log={log_path}\n", flush=True)
    return process.returncode, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, default=os.path.join(REPO, "src", "data", "processed", "homologs"))
    parser.add_argument("--config", type=str, default=os.path.join(REPO, "configs", "gnn_pretrain.json"))
    parser.add_argument("--output_root", type=str, default=os.path.join(REPO, "src", "train", "gnn"))
    parser.add_argument("--splits", nargs="+", default=["homolog", "cluster"], choices=["homolog", "cluster"])
    # anything after -- is forwarded to pretrain.py (e.g. --seed 7, --no_early_stopping)
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to pretrain.py, after a bare --")
    args = parser.parse_args()

    extra = [a for a in args.rest if a != "--"]

    results = []
    for split in args.splits:
        output_dir = os.path.join(args.output_root, f"output_pretrain_{split}")
        code, elapsed = run_one(split, args.processed, args.config, output_dir, extra)
        results.append((split, code, elapsed, output_dir))
        # keep going even if one split fails, so a crash in the first does not cost the second

    print("=" * 78, flush=True)
    print("SUMMARY", flush=True)
    for split, code, elapsed, output_dir in results:
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"  {split:8s} {status:20s} {elapsed/60:5.1f} min  {output_dir}", flush=True)
    print("=" * 78, flush=True)

    sys.exit(0 if all(code == 0 for _, code, _, _ in results) else 1)
