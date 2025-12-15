#!/usr/bin/env python3
# train_timevarying.py

from pathlib import Path
import argparse, json, math, random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split


# time-varying Bayesian GCN (3-layer, one LN & one Dropout)
from baysian_gcn_TV import DenseTemporalBayesGCN3_Min  

# --------------------- utils ---------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False # when True, deterministic but slower
    torch.backends.cudnn.benchmark = True # when False, deterministic but slower


def split_dataset(dataset, seed=0, val_frac=0.2, test_frac=0.2):
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    val_size  = max(1, int(n * val_frac))
    test_size = max(1, int(n * test_frac))
    if val_size + test_size >= n:
        val_size  = max(1, min(val_size,  n - 2))
        test_size = max(1, min(test_size, n - 1 - val_size))
    train_size = n - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size], generator=g)


def gaussian_nll_fixed(pred: torch.Tensor, target: torch.Tensor, sigma_floor: float) -> torch.Tensor:
    """
    Fixed-sigma Gaussian negative log-likelihood (per-sample mean).
    pred, target are scalars (log r). sigma_floor is in log-r space.
    """
    # Use an additive floor (keeps gradients healthy vs hard clamp)
    s = pred.new_tensor(float(sigma_floor))
    var = s * s
    return 0.5 * (((target - pred) ** 2) / var + torch.log(var))

def kl_weight_linear(epoch: int, total_epochs: int, max_w: float = 1e-3, warmup_frac: float = 0.8) -> float:
    """
    Linear KL warmup up to warmup_frac * total_epochs, then hold.
    """
    warm = max(1, int(total_epochs * warmup_frac))
    return float(max_w) * min(1.0, epoch / warm)


@torch.no_grad()
def evaluate(model,
             loader,
             device: str,
             mc: int = 75,
             sigma_floor: float = 0.5,
             rng_seed: int = 12345):
    """
    Returns (mse_logr, mae_logr, nll_logr) averaged over dataset.
    Uses fixed RNG so metrics are comparable across epochs.
    """
    model.eval()
    mse_logr = 0.0
    mae_logr = 0.0
    nll_logr = 0.0
    n = 0

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    with torch.random.fork_rng(devices=[device] if device.startswith("cuda") else []):
        torch.manual_seed(rng_seed)
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            for data in loader:
                data = data.to(device, non_blocking=True)
                # MC over stochastic forward passes (epistemic)
                preds = []
                for _ in range(mc):
                    preds.append(model(data))  # scalar log r
                pred_logr = torch.stack(preds).mean()  # posterior mean of log r
                true_logr = data.y.view(-1)[0]

                diff = (pred_logr - true_logr).item()
                mse_logr += diff * diff
                mae_logr += abs(diff)

                # NLL with fixed sigma floor (aleatoric in log-r space)
                nll_logr += gaussian_nll_fixed(pred_logr, true_logr, sigma_floor).item()
                n += 1

    n = max(1, n)
    return mse_logr / n, mae_logr / n, nll_logr / n


@torch.no_grad()
def summarize_predictions(model,
                          loader,
                          device: str,
                          mc: int = 300,
                          sigma_floor: float = 0.5):
    """
    Optional helper: produce posterior mean of log r, epistemic var (log r),
    total var = epi + sigma_floor^2, and r-space mean using log-normal moment.
    """
    model.eval()
    all_mu = []

    for _ in range(mc):
        mus_b = []
        for data in loader:
            data = data.to(device, non_blocking=True)
            mus_b.append(model(data).detach().cpu())  # [1,1] or scalar
        all_mu.append(torch.cat(mus_b, dim=0).unsqueeze(0))  # [1, N, 1]

    mus = torch.cat(all_mu, dim=0)          # [M, N, 1]
    mu_post = mus.mean(dim=0).squeeze(1).numpy()     # [N]
    epi_var = mus.var(dim=0, unbiased=False).squeeze(1).numpy()  # [N]
    tot_var = epi_var + (sigma_floor ** 2)
    r_mean = np.exp(mu_post + 0.5 * tot_var)         # E[r] for log-normal
    return mu_post, epi_var, tot_var, r_mean


    
# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser("Train Dense Temporal Bayesian GCN (time-varying, variance floor)")
    # Data
    ap.add_argument("--data-pt", required=True, help="Dataset .pt (list[torch_geometric.data.Data])")
    ap.add_argument("--A-pt", required=True, help="Time-varying adjacency tensor .pt with shape (T, N, N)")
    # I/O
    ap.add_argument("--save-dir", default="runs/tv_bayes3", help="Directory to save best.pt and history.json")
    ap.add_argument("--predict-csv", default="", help="Optional path to dump test predictions summary")
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=1, help="Keep 1 unless your model supports batching per-graph A_t")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Bayesian / ELBO
    ap.add_argument("--elbo-samples", type=int, default=4, help="MC samples inside ELBO per batch")
    ap.add_argument("--sigma-floor", type=float, default=0.5, help="Fixed sigma in log-r space (variance floor)")
    ap.add_argument("--kl-max", type=float, default=1e-3, help="Max KL weight after warmup")
    ap.add_argument("--kl-warmup-frac", type=float, default=0.8, help="Fraction of epochs to warm up KL")
    # Model shape (match-ish baysian_gcn defaults)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--h3", type=int, default=256)
    ap.add_argument("--agg", choices=["mean", "attn"], default="mean")
    ap.add_argument("--norm", choices=["none", "gcn"], default="none")
    ap.add_argument("--add-self-loops", action="store_true")
    # Splits
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    # Eval / Early stopping
    ap.add_argument("--val-mc", type=int, default=75)
    ap.add_argument("--test-mc", type=int, default=300)
    ap.add_argument("--val-seed", type=int, default=12345, help="Fixed RNG seed for validation MC sampling")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--resume", type=str, default="",
                help="Path to checkpoint (best.pt) to resume from")
    ap.add_argument("--resume-reset-lr", action="store_true",
                help="If set, do not load optimizer state (keeps current LR)")

    args = ap.parse_args()

    if args.batch_size != 1:
        print("[WARN] For safety, forcing batch_size=1 because A_t is per-graph/time-varying.")
        args.batch_size = 1

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data_pt}")
    dataset = torch.load(args.data_pt, weights_only=False, map_location="cpu")
    print(f"Loading A_t: {args.A_pt}")
    A_cpu = torch.load(args.A_pt, map_location="cpu")  # expected (T, N, N)

    # Ensure per-sample Data objects don't carry stale A; we will set shared A_t on the model
    for d in dataset:
        if hasattr(d, "A"):
            d.A = None

    print(f"Total samples: {len(dataset)}")
    train_ds, val_ds, test_ds = split_dataset(dataset, seed=args.seed,
                                              val_frac=args.val_frac, test_frac=args.test_frac)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, pin_memory=True)


    def fixed_gauss_criterion(pred, target):
        return gaussian_nll_fixed(pred, target, args.sigma_floor)

    # ----- Model -----
    model = DenseTemporalBayesGCN3_Min(
        h1=args.h1, h2=args.h2, h3=args.h3,
        dropout=args.dropout,
        agg=args.agg,
        norm=args.norm,
        add_self_loops=args.add_self_loops
    ).to(args.device)
    model.set_A(A_cpu.to(args.device, non_blocking=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    #     # scheduler tracks val NLL to align with training objective
    # )

    # Complexity weight base (scaled by KL warmup)
    base_complexity = 1.0 / max(1, len(train_loader))

    # Mixed precision settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    # ----- Resume if requested -----
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model_state"])
        if not args.resume_reset_lr and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "best_val_mse_logr" in ckpt:
            best_val = ckpt["best_val_mse_logr"]
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"[Resume] Loaded {args.resume} (epoch {ckpt.get('epoch','?')}, best Val MSE(log r)={best_val:.4f})")

    # ---------- Training loop ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_elbo = 0.0

        # KL warmup weight
        beta = kl_weight_linear(epoch, args.epochs, max_w=args.kl_max, warmup_frac=args.kl_warmup_frac)
        # complexity_cost_weight = beta * base_complexity

        for data in train_loader:
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Forward under AMP
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                # Define criterion in-closure so it captures current args.sigma_floor

                loss = model.sample_elbo(
                    inputs=data,
                    labels=data.y.float().view(1, 1),
                    criterion=fixed_gauss_criterion,
                    sample_nbr=args.elbo_samples,
                    complexity_cost_weight=beta
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_elbo += float(loss.item())

        train_elbo = total_elbo / max(1, len(train_loader))

        # ----- Validation -----
        val_mse, val_mae, val_nll = evaluate(
            model, val_loader, args.device,
            mc=args.val_mc, sigma_floor=args.sigma_floor, rng_seed=args.val_seed
        )

        history.append({
            "epoch": epoch,
            "train_elbo": train_elbo,
            "val_mse_logr": val_mse,
            "val_mae_logr": val_mae,
            "val_nll_logr": val_nll,
            "kl_weight": beta
        })

        print(f"Epoch {epoch:03d} | ELBO {train_elbo:.4f} | "
              f"Val MSE(log r) {val_mse:.4f} | Val MAE(log r) {val_mae:.4f} | Val NLL {val_nll:.4f} | KLw {beta:.2e}")

        # Step LR scheduler on the objective we care about
        # scheduler.step(val_nll)

        # ----- Checkpointing (ONLY best) by Val NLL -----
        es_warmup = 10
        improved = (best_val - val_nll) > args.min_delta
        if improved:
            best_val = val_nll
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt = {
                "epoch": epoch,
                "args": vars(args),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_nll": best_val,
                "best_val_mse": val_mse,
                "best_val_mae": val_mae,
            }
            torch.save(ckpt, Path(args.save_dir) / "best.pt")
        else:
            if epoch >= es_warmup:
                epochs_no_improve += 1

        # ----- Early stopping -----
        if epoch >= es_warmup and epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (best at {best_epoch}, Val NLL={best_val:.6f}).")
            break

    # Save history
    with open(Path(args.save_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---------- Test using best checkpoint ----------
    ckpt = torch.load(Path(args.save_dir) / "best.pt", map_location=args.device)
    model.load_state_dict(ckpt["model_state"])

    test_mse, test_mae, test_nll = evaluate(
        model, test_loader, args.device,
        mc=args.test_mc, sigma_floor=args.sigma_floor, rng_seed=args.val_seed  # reuse seed or choose another
    )
    print(f"TEST  MSE(log r): {test_mse:.6f} | TEST MAE(log r): {test_mae:.6f} | TEST NLL: {test_nll:.6f}")
    print(f"Done. Best Val NLL(log r): {best_val:.6f} (epoch {best_epoch}).")

    # Optional: dump per-sample predictive summary (posterior mean/vars, r-space mean)
    if args.predict_csv:
        mu_post, epi_var, tot_var, r_mean = summarize_predictions(
            model, test_loader, args.device, mc=args.test_mc, sigma_floor=args.sigma_floor
        )
        import csv
        with open(args.predict_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "mu_logr", "epi_var_logr", "tot_var_logr", "r_mean"])
            for i in range(len(mu_post)):
                w.writerow([i, float(mu_post[i]), float(epi_var[i]), float(tot_var[i]), float(r_mean[i])])
        print(f"Saved test prediction summary to {args.predict_csv}")


if __name__ == "__main__":
    main()
