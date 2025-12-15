#!/usr/bin/env python3
import os, sys, time, math, random, pickle, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.data import Data, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.stats import norm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../codes")))

from baysian_gcn import *


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def set_seeds(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prefer speed over strict determinism:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def kl_weight(epoch, total_epochs, max_w=3e-3, warmup_frac=0.4):
    warm = int(total_epochs * warmup_frac)
    return max_w * min(1.0, epoch / max(1, warm))

### ELBO uses a complexity_cost_weight (the KL multiplier). If that weight is too small, the prior won’t help; too big and you’ll over-shrink.

def gaussian_nll_fixed(pred, target, sigma_floor: float):
    """
    NLL for y ~ N(pred, sigma^2), with sigma fixed (log-space std).
    Returns mean over batch.
    """
    # keep on correct device/dtype
    s = pred.new_tensor(sigma_floor)
    var = s * s
    # 0.5 * ((y - μ)^2 / σ^2 + log(σ^2)) ; add small eps to the log if you want
    return 0.5 * (((target - pred) ** 2) / var + torch.log(var)).mean()

class Tee:
    def __init__(self, path):
        self.file = open(path, 'w')
    def write(self, msg):
        sys.__stdout__.write(msg)
        self.file.write(msg)
        self.file.flush()
    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()
    
# -------------------- args --------------------
# my_r   = float(sys.argv[1]) if len(sys.argv) >= 2 else 0.1
my_r   = float(sys.argv[1]) if len(sys.argv) >= 2 else 1.5
seed   = int(sys.argv[2])   if len(sys.argv) >= 3 else 123
sigma_floor = 0.5  # <-- fixed σ in log-space

# optional 2nd arg: seed
seed = int(sys.argv[2]) if len(sys.argv) >= 3 else 123
# set_seeds(seed)
device = get_device()

# -------------------- data --------------------
torch.serialization.add_safe_globals([Data])
# data_path = './gnn_inference_data/r-{}/'.format(my_r)
data_path = './R-{}/'.format(my_r)
# dataset = torch.load(data_path+'dataset_49.pt')
# dataset = torch.load(data_path+'dataset_49.pt', weights_only=False)
# load dataset
dataset = torch.load(data_path + 'dataset_50.pt', weights_only=False)

# slice features to first 49
for data in dataset:
    data.x = data.x[:, :49]   # keep only first 49 features

os.makedirs('infer_results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

timestamp = time.strftime('%Y%m%d-%H%M%S')
save_stub = f'infer_results/R{my_r}_seed{seed}_{timestamp}'
model_out = f'{save_stub}_best_model.pth'
pickle_out = f'{save_stub}_results.pkl'
# npz_out    = f'{save_stub}_results.npz'
train_log  = f'logs/train_R{my_r}_seed{seed}_{timestamp}.log'
sys.stdout = Tee(train_log)

print(f"Device: {device}")
print(f"R target directory: R-{my_r}")
print(f"Seed: {seed}")
print(f"Fixed σ_floor (log-space): {sigma_floor}")

# -------------------- loaders --------------------

num_node_features = 49
hidden_channels = 128
batch_size = 16
learning_rate = 1e-3
num_epochs = 200
patience = 30

train_size = int(len(dataset) * 0.6)
val_size = int(len(dataset) * 0.2)
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# -------------------- model --------------------
model = GCNBayesian(num_node_features=num_node_features, hidden_channels=hidden_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
counter = 0
loss_history = []
criterion = torch.nn.MSELoss()

# criterion to pass into Blitz's sample_elbo (adds KL automatically)
def fixed_gauss_criterion(pred, target):
    return gaussian_nll_fixed(pred, target, sigma_floor)



# -------------------- train / val --------------------

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    train_losses = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        # --------------old----------------
        # loss = model.sample_elbo_loss(batch, criterion=criterion, sample_nbr=5, complexity_cost_weight=1e-2)
        # loss = model.sample_elbo_loss(..., sample_nbr=5, complexity_cost_weight=1e-2)
        # loss.backward()
        # --------------new----------------
        elbo = model.sample_elbo_loss(
            data=batch,
            criterion=fixed_gauss_criterion,
            sample_nbr=30,
            complexity_cost_weight=kl_weight(epoch, num_epochs, max_w=1e-3, warmup_frac=0.8)
        )
        elbo.backward()
        # --------------end of changes----------------
        optimizer.step()
        # train_losses.append(loss.item())
        train_losses.append(float(elbo.item()))
    avg_train_loss = float(np.mean(train_losses))
    # print(f"  Train Loss: {avg_train_loss:.4f}")

    model.eval()

    val_loss_total, val_count = 0.0, 0
    muvar_over_s2_accum = 0.0  # optional diagnostic
    mc_val = 30                # same order as training S
    with torch.no_grad():
        s2 = sigma_floor * sigma_floor
        for batch in val_loader:
            batch = batch.to(device)
            # pred = model(batch)
            mus = torch.stack([model(batch) for _ in range(mc_val)], dim=0)  # [S,B,...]
            mu_mean = mus.mean(dim=0)
            mu_var  = mus.var(dim=0, unbiased=False) 
            nll = 0.5 * ( ((batch.y - mu_mean)**2 + mu_var) / s2 + math.log(s2) )
            batch_mean = nll.mean().item()
            val_loss_total += batch_mean * batch.y.shape[0]
            val_count      += batch.y.shape[0]
            muvar_over_s2_accum += (mu_var / s2).mean().item()
            # val_losses.append(val_loss.item())
        val_nll_exp = val_loss_total / val_count
        mean_muvar_over_s2 = muvar_over_s2_accum / len(val_loader)
    # print(f"  Val Loss: {val_loss_avg:.4f}")
    val_loss_avg = val_nll_exp
    print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {val_nll_exp:.4f}| mean Var(mu_s)/sigma^2: {mean_muvar_over_s2:.4f}")
    loss_history.append(val_loss_avg)

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        counter = 0
        torch.save(model.state_dict(), model_out)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break


# Testing with uncertainty
model.eval()
mc_test = 100  # enough for stable intervals
# test_preds = []

with torch.no_grad():
    mu_list = []
    for _ in range(mc_test):
        # preds_batch = []
        mu_b =  []
        for batch in test_loader:
            batch = batch.to(device)
            
            mu = model(batch)
            mu_b.append(mu.detach().cpu())
            
        mu_list.append(torch.cat(mu_b, dim=0).unsqueeze(0))  # [1,N,1]
    mus = torch.cat(mu_list, dim=0)    

mu_post = mus.mean(dim=0).squeeze(1).numpy()                     # posterior mean of log r
epi_var = mus.var(dim=0, unbiased=False).squeeze(1).numpy()      # epistemic var (log r)
tot_var = epi_var + (sigma_floor ** 2)                           # add fixed aleatoric
tot_sigma = np.sqrt(tot_var)                    # [N]
#--------------------------------------------------------
z025, z975 = norm.ppf(0.025), norm.ppf(0.975)   # ≈ (-1.96, +1.96)
r_ci_lo = np.exp(mu_post + z025 * tot_sigma)    # [N]
r_ci_hi = np.exp(mu_post + z975 * tot_sigma)    # [N]

# Convert to r-space (log-normal moments)
r_mean = np.exp(mu_post + 0.5 * tot_var)
r_median = np.exp(mu_post) 
r_sd   = np.sqrt((np.exp(tot_var) - 1.0) * np.exp(2 * mu_post + tot_var))

# Ground truth
y_true_log = torch.cat([b.y for b in test_loader], dim=0).numpy().flatten()
r_true = np.exp(y_true_log)

print(f"Inferred log r (mean over samples): {r_mean.mean().item():.3g}  "
      f"[{r_mean.min().item():.3g}, {r_mean.max().item():.3g}]  "
      f"avg sd: {r_sd.mean().item():.3g}")

results = {
    "r_true": r_true,
    "r_pred": r_mean,
    "r_pred_median": r_median,   # log-normal median
    "r_std":  r_sd,
    "r_ci_lo": r_ci_lo,          # 2.5th percentile
    "r_ci_hi": r_ci_hi,          # 97.5th percentile
    "loss_history": loss_history,
    "epi_var_logr": epi_var,
    "sigma_floor": sigma_floor
}
with open(pickle_out, "wb") as f:
    pickle.dump(results, f)

print(f"Saved model to: {model_out}")
print(f"Saved results to: {pickle_out}")

