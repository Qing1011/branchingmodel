#!/usr/bin/env python3
# infer_real.py
import argparse, gzip, json, math
import numpy as np
import torch, h5py
from torch_geometric.data import Data
from baysian_gcn_TV import DenseTemporalBayesGCN3_Min

def load_x(path: str) -> np.ndarray:
    if path.endswith(".npy.gz"):
        with gzip.open(path, "rb") as f:
            return np.load(f)
    elif path.endswith(".npy"):
        return np.load(path)
    raise ValueError("x must be .npy or .npy.gz (shape N×T_total)")

def load_A_window(A_pt: str, W_path: str, t0: int, T: int) -> np.ndarray:
    if bool(A_pt) == bool(W_path):
        raise ValueError("Provide exactly one of --A-pt or --W-path")
    if A_pt:
        A_full = torch.load(A_pt, map_location="cpu")  # (T_total,N,N)
        if not isinstance(A_full, torch.Tensor) or A_full.dim() != 3:
            raise ValueError(f"{A_pt} must be a 3D tensor (T_total,N,N)")
        A_np = A_full[t0:t0+T].cpu().numpy()
    else:
        with h5py.File(W_path, "r") as f:
            if "WM" not in f:
                raise KeyError(f"{W_path} must contain dataset 'WM'")
            A_np = f["WM"][t0:t0+T]  # (T,N,N)
    return np.asarray(A_np, dtype=np.float32)


# -------------------- Inference utils --------------------
@torch.no_grad()
def mc_predict_logr(model, data, mc: int, use_amp: bool = True) -> np.ndarray:
    preds = []
    if use_amp and torch.cuda.is_available():
        dtype_ac = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast("cuda", dtype=dtype_ac)
    else:
        from contextlib import nullcontext as ctx
        ctx = ctx()
    model.eval()
    with ctx:
        for _ in range(mc):
            preds.append(model(data).item())
    return np.asarray(preds, dtype=np.float64)

def summarize_variants(preds_logr: np.ndarray, sigma_floor: float | None, mc_seed: int = 0):
    """
    Return a dict with ALL variants:
      variance_mode in {"epistemic","total"}  x  sampler in {"analytic","mc"}
      and point estimates in r: mean, median, mode for each variance_mode.
    """
    rng = np.random.default_rng(mc_seed)
    mu_logr_mc = float(preds_logr.mean())
    epi_var = float(preds_logr.var(ddof=1)) if preds_logr.size > 1 else 0.0
    alea_var = 0.0 if (sigma_floor is None) else float(sigma_floor**2)

    results = {}
    for variance_mode in ("epistemic", "total"):
        var_logr = epi_var + (alea_var if variance_mode == "total" else 0.0)
        sd_logr = float(np.sqrt(max(var_logr, 0.0)))

        # Point estimates in r for this variance choice
        r_mean   = float(np.exp(mu_logr_mc + 0.5 * var_logr))
        r_median = float(np.exp(mu_logr_mc))                 # independent of variance
        r_mode   = float(np.exp(mu_logr_mc - var_logr))

        # Analytic (log-normal) CI
        lo_logr = mu_logr_mc - 1.96 * sd_logr
        hi_logr = mu_logr_mc + 1.96 * sd_logr
        ana = {
            "mu_logr": mu_logr_mc,
            "sd_logr": sd_logr,
            "ci95_logr": [float(lo_logr), float(hi_logr)],
            "mu_r": float(np.exp(mu_logr_mc + 0.5 * var_logr)),
            "sd_r": float(np.sqrt((np.exp(var_logr) - 1.0) * np.exp(2*mu_logr_mc + var_logr))),
            "ci95_r": [float(np.exp(lo_logr)), float(np.exp(hi_logr))],
        }

        # MC sampler (adds aleatoric only if "total")
        if variance_mode == "total" and alea_var > 0:
            z = preds_logr + rng.normal(0.0, np.sqrt(alea_var), size=preds_logr.shape[0])
        else:
            z = preds_logr
        r_samp = np.exp(z)
        mc = {
            "mu_logr": mu_logr_mc,
            "sd_logr": float(np.sqrt(var_logr)),
            "ci95_logr": [float(mu_logr_mc - 1.96*np.sqrt(var_logr)),
                          float(mu_logr_mc + 1.96*np.sqrt(var_logr))],
            "mu_r": float(r_samp.mean()),
            "sd_r": float(r_samp.std(ddof=1)) if r_samp.size > 1 else 0.0,
            "ci95_r": [float(np.percentile(r_samp, 2.5)), float(np.percentile(r_samp, 97.5))],
        }

        results[variance_mode] = {
            "variance_components": {"epi_var": epi_var, "alea_var": alea_var if variance_mode=="total" else 0.0},
            "point_r": {"mean": r_mean, "median": r_median, "mode": r_mode},
            "analytic": ana,
            "mc": mc,
        }
    return results


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser("Infer r (print ALL variants) with a time-varying graph")
    ap.add_argument("--ckpt", required=True, help="Path to best.pt saved by train_timevarying.py")
    ap.add_argument("--x-path", required=True, help="N×T_total infections (.npy or .npy.gz)")
    ap.add_argument("--A-pt", default="", help="torch .pt with A_t: (T_total,N,N)")
    ap.add_argument("--W-path", default="", help="HDF5 with dataset 'WM': (T_total,N,N)")
    ap.add_argument("--t0", type=int, required=True, help="start index (inclusive)")
    ap.add_argument("--T",  type=int, required=True, help="window length")
    ap.add_argument("--mc", type=int, default=400, help="MC samples for epistemic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sigma-floor", type=float, default=None,
                    help="Sigma floor in log-r; defaults to ckpt args if present")
    ap.add_argument("--mc-seed", type=int, default=0, help="Seed used for MC sampler in r-space")
    ap.add_argument("--json-out", default="", help="Optional: path to save all results in JSON")
    args = ap.parse_args()

    # Performance knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # --- Load ckpt & model ---
    ckpt = torch.load(args.ckpt, map_location=args.device)
    margs = ckpt.get("args", {}) or {}
    model = DenseTemporalBayesGCN3_Min(
        h1=margs.get("h1", 64),
        h2=margs.get("h2", 64),
        h3=margs.get("h3", 32),
        dropout=margs.get("dropout", 0.5),
        agg=margs.get("agg", "mean"),
        norm=margs.get("norm", "none"),
        add_self_loops=margs.get("add_self_loops", False),
    ).to(args.device)

    state = ckpt["model_state"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    state.pop("A_buf", None)
    model.load_state_dict(state, strict=True)

    # sigma floor default from ckpt if not provided
    sigma_floor = args.sigma_floor
    if sigma_floor is None:
        sigma_floor = float(margs.get("sigma_floor", 0.0))
    print(f"[CKPT] epoch={int(ckpt.get('epoch', -1))} | "
          f"best Val NLL={float(ckpt.get('best_val_nll', float('nan'))):.6f} | "
          f"best Val MSE={float(ckpt.get('best_val_mse', float('nan'))):.6f} | "
          f"best Val MAE={float(ckpt.get('best_val_mae', float('nan'))):.6f}")
    print(f"[Config] sigma_floor(log-r)={sigma_floor} | MC={args.mc} | mc_seed={args.mc_seed}")

    # --- Slice window & set A_t ---
    X_full = load_x(args.x_path)                     # (N, T_total)
    t0, T = int(args.t0), int(args.T)
    x_np = X_full[:, t0:t0+T]                        # (N, T)
    A_np = load_A_window(args.A_pt, args.W_path, t0, T)  # (T, N, N)
    N = x_np.shape[0]
    if A_np.shape != (T, N, N):
        raise ValueError(f"A window mismatch: got {A_np.shape}, expected ({T},{N},{N})")

    x = torch.from_numpy(x_np.astype("float32", copy=False)).contiguous().to(args.device)
    A = torch.from_numpy(A_np.astype("float32", copy=False)).contiguous().to(args.device)
    model.set_A(A)
    data = Data(x=x)

    # --- MC posterior on log r ---
    preds_logr = mc_predict_logr(model, data, mc=args.mc, use_amp=True)

    # --- Summarize ALL variants ---
    all_stats = summarize_variants(preds_logr, sigma_floor=sigma_floor, mc_seed=args.mc_seed)

    # --- Pretty print ---
    for var_mode in ("epistemic", "total"):
        block = all_stats[var_mode]
        print("\n==============================")
        print(f"Variance mode: {var_mode.upper()}  "
              f"(epi_var={block['variance_components']['epi_var']:.6f}, "
              f"alea_var={block['variance_components']['alea_var']:.6f})")
        print(f"Point r: mean={block['point_r']['mean']:.6f}  "
              f"median={block['point_r']['median']:.6f}  "
              f"mode={block['point_r']['mode']:.6f}")

        ana = block["analytic"]
        print("[Analytic]  "
              f"logr: mu={ana['mu_logr']:.6f}, sd={ana['sd_logr']:.6f}, "
              f"95%CI=[{ana['ci95_logr'][0]:.6f}, {ana['ci95_logr'][1]:.6f}]  |  "
              f"r: mean={ana['mu_r']:.6f}, sd={ana['sd_r']:.6f}, "
              f"95%CI=[{ana['ci95_r'][0]:.6f}, {ana['ci95_r'][1]:.6f}]")

        mc = block["mc"]
        print("[MC]        "
              f"logr: mu={mc['mu_logr']:.6f}, sd={mc['sd_logr']:.6f}, "
              f"95%CI=[{mc['ci95_logr'][0]:.6f}, {mc['ci95_logr'][1]:.6f}]  |  "
              f"r: mean={mc['mu_r']:.6f}, sd={mc['sd_r']:.6f}, "
              f"95%CI=[{mc['ci95_r'][0]:.6f}, {mc['ci95_r'][1]:.6f}]")

    # --- JSON dump (optional) ---
    if args.json_out:
        out = {
            "meta": {
                "ckpt_epoch": int(ckpt.get("epoch", -1)),
                "best_val_nll": float(ckpt.get("best_val_nll", float("nan"))),
                "best_val_mse": float(ckpt.get("best_val_mse", float("nan"))),
                "best_val_mae": float(ckpt.get("best_val_mae", float("nan"))),
            },
            "config": {"mc": args.mc, "t0": t0, "T": T, "sigma_floor": sigma_floor, "mc_seed": args.mc_seed},
            "all_stats": all_stats,
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print("\nSaved JSON to", args.json_out)


if __name__ == "__main__":
    main()
