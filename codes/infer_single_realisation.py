#!/usr/bin/env python3
import argparse, os, gzip
from pathlib import Path
import numpy as np
import random
import torch
from torch_geometric.data import Data
# If baysian_gcn.py is not in the same dir, add its folder to PYTHONPATH or adjust import.
from baysian_gcn import GCNBayesian  # uses your existing model class

def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple MPS if you ever hop onto a Mac:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seeds(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # keep these non-strict to avoid CuBLAS determinism errors unless you export the env var in sbatch
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def load_gzipped_numpy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with gzip.open(path, "rb") as f:
        arr = np.load(f, allow_pickle=True)
    return arr

def build_edge_index_from_W(W: np.ndarray):
    """Directed edges for all W[i,j] > 0 (no self-loops)."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got {W.shape}")
    mask = W > 0
    np.fill_diagonal(mask, False)
    src, dst = np.where(mask)
    edge_index = torch.as_tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.as_tensor(W[src, dst].astype(np.float32), dtype=torch.float32)
    return edge_index, edge_weight

def main():
    ap = argparse.ArgumentParser(description="Infer r from a single simulated realization with a trained Bayesian GCN.")
    ap.add_argument("--r", type=float, required=True, help="r used for the simulation subfolder name (e.g., 2.0).")
    ap.add_argument("--es-idx", type=int, default=0, help="Ensemble idx used in the filename (default: 0).")
    ap.add_argument("--data-root", type=str, required=True,
                    help="Root dir containing /NewInf_r-{r}_{es}.npy.gz") ### or rr !!!
    ap.add_argument("--W-path", type=str, required=True, help="Path to W_avg.csv")
    ap.add_argument("--model-path", type=str, required=True,
                    help="Path to trained checkpoint .pth from infer_gpu.py (the *best_model_gpu_r-*.pth).")
    ap.add_argument("--t-start", type=int, default=10, help="Start time index (inclusive) for feature window (default 10).")
    ap.add_argument("--t-len",   type=int, default=49, help="Number of timesteps for node features (default 49).")
    ap.add_argument("--samples", type=int, default=1000, help="Number of stochastic forward passes for uncertainty.")
    ap.add_argument("--sigma-floor", type=float, default=0.01,  # << add σ_floor (log-space)
                    help="Fixed log-space std to add to MC variance")
    ap.add_argument("--seed",    type=int, default=123, help="RNG seed.")
    ap.add_argument("--tau", type=float, default=1.0,
                help="Uncertainty temperature: scales epistemic variance")
    args = ap.parse_args()

    set_seeds(args.seed)
    device = set_device()

    # ---- Files
    data_root = Path(args.data_root).expanduser().resolve()
    # sim_dir   = data_root / f"branching_r-{np.round(args.r, 3)}"
    # sim_file  = data_root / f"NewInf_r-{np.round(args.r, 3)}_{args.es_idx}.npy.gz"
    sim_file  = data_root / f"NewInf_r-{np.round(args.r, 3)}_{args.es_idx}_rr.npy.gz"

    W_path    = Path(args.W_path).expanduser().resolve()
    ckpt_path = Path(args.model_path).expanduser().resolve()

    # ---- Load data
    NewInf = load_gzipped_numpy(sim_file)      # shape (N, T_total)
    W      = np.loadtxt(W_path)                # shape (N, N)
    N, T_total = NewInf.shape
    # ----- linearly varying reporting rate adjustment ----
    # one_location = np.linspace(0.1, 0.2, T_total)
    # m_alpha = np.tile(one_location, (N, 1))
    # NewInf_true = NewInf / m_alpha  # adjust for reporting rate
    # ---- Time window (must match training)
    t0 = args.t_start
    t1 = t0 + args.t_len                       # exclusive
    if t1 > T_total:
        raise ValueError(f"Requested window [{t0}:{t1}) exceeds data length T={T_total}")
    
    x_np = NewInf[:, t0:t1].astype(np.float32) # (N, t_len)
    ############----------check here to see if you need reporting rate adjustment----
    # reporting_rate = np.loadtxt('reporting_rates_test.csv')  # shape (N,)
    x_np = x_np/0.175  # adjust for reporting rate
    # print('new x shape:',x_np.shape)

    # 1D array: 60 values linearly spaced from 0.1 to 0.2


    # ---- Graph
    edge_index, edge_weight = build_edge_index_from_W(W)
    # batch = torch.zeros(x_np.shape[0], dtype=torch.long)
    data = Data(x=torch.from_numpy(x_np),
                edge_index=edge_index,
                edge_attr=edge_weight)

    # ---- Model
    num_node_features = args.t_len
    hidden_channels   = 128
    model = GCNBayesian(num_node_features=num_node_features, hidden_channels=hidden_channels).to(device)
    # for torch.load of PyG Data objects in older saves:
    torch.serialization.add_safe_globals([Data])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()  # enable variational/dropout stochasticity

    data = data.to(device)

    # ---- Stochastic forward passes (predicting log(r))
    with torch.no_grad():
        samples = []
        for _ in range(args.samples):
            mu = model(data)            # shape (1, 1) since global pooling
            samples.append(mu.squeeze().item())
    samples = np.array(samples, dtype=np.float64)

    # ---- Convert log(r) → r using lognormal transform

    mu_log = samples.mean()
    var_ep = samples.var(ddof=0)

    var_tot = (args.tau ** 2) * var_ep + args.sigma_floor ** 2
    sigma_tot = np.sqrt(var_tot)
    
    r_mean = np.exp(mu_log + 0.5*var_tot)
    r_median = np.exp(mu_log) 
    r_sd   = np.sqrt((np.exp(var_tot) - 1.0) * np.exp(2*mu_log + var_tot))
    
    # Parametric 95% CI from log-normal
    z = 1.959963984540054  # 95% quantile
    ci_low = np.exp(mu_log - z * sigma_tot)
    ci_high = np.exp(mu_log + z * sigma_tot)


    #--------------------------------------------------------
    print(f"Device: {device}")
    print(f"Loaded: {sim_file.name} | W: {W_path.name} | model: {ckpt_path.name}")
    print(f"Window: t=[{t0}:{t1}) len={args.t_len} | MC={args.samples} | σ_floor={args.sigma_floor}")
    print(f"Inferred r (median): {r_median:.4f}")
    print(f"Inferred r (mean):   {r_mean:.4f}")
    print(f"95% CI for r:        [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Inferred r: {r_mean:.4f} ± {r_sd:.4f}")
    print(f"σ_floor={args.sigma_floor}, τ={args.tau}")

if __name__ == "__main__":
    main()