#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np
import torch

# If your function lives alongside this file, make sure PYTHONPATH includes it.
# e.g., export PYTHONPATH=/path/to/your/code:$PYTHONPATH
try:
    from prepare_observational_data import prepare_dataset
except Exception as e:
    raise ImportError(
        "Could not import prepare_observational_data. "
        "Set PYTHONPATH correctly or place this script next to that module."
    ) from e

def build_edge_index(W):
    """Turn dense square matrix W (N x N) into edge_index [2,E] and edge_weight [E].
       Drops self-loops; keeps all edges > 0."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got {W.shape}")
    mask = W > 0
    np.fill_diagonal(mask, False)
    src, dst = np.where(mask)
    # edge_index = torch.tensor([src, dst], dtype=torch.long)
    # edge_weight = torch.tensor(W[src, dst], dtype=torch.float32)
    indices_np = np.stack([src, dst], axis=0).astype(np.int64) # (2, E)
    edge_index = torch.from_numpy(indices_np)                       
    weights_np = W[src, dst].astype(np.float32)                 # (E,)
    edge_weight = torch.from_numpy(weights_np)
    return edge_index, edge_weight

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, help="Folder with simulated exports")
    ap.add_argument("--W-path", required=True, help="Path to W_avg.csv (dense matrix)")
    ap.add_argument("--save-dir", required=True, help="Output base directory")
    ap.add_argument("--r", type=float, required=True, help="r value used to name output subfolder")
    ap.add_argument("--g-idx", type=int, required=True, help="Index of the simulation realisation (e.g., 0-299)")
    ap.add_argument("--sep", type=int, required=True, help="Separator (e.g., 49, 42, ...)")
    args = ap.parse_args()

    export_dir = Path(args.export_dir).expanduser().resolve()
    W_path = Path(args.W_path).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve() / f"r-{args.r}_{args.g_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # load W and build graph
    W = np.loadtxt(W_path, delimiter=None)
    edge_index, edge_weight = build_edge_index(W)

    # default r-grid (use your original if you want)
    rs = np.array([10.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.025, 5.,  2.5  , 3.333,  1.333,  0.667,  0.286,  0.133,
        0.067,  0.033, 0.37,7.4,20,13.333])

    # make dataset and save
    dataset = prepare_dataset(str(export_dir), rs, args.sep, edge_index, edge_weight)
    out_path = save_dir / f"dataset_{args.sep}.pt"
    torch.save(dataset, out_path)
    print(f"[OK] saved {out_path}")

if __name__ == "__main__":
    main()
