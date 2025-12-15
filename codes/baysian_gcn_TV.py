# dense_temporal_bayes_gcn_3layer_min.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

def gcn_norm(A: torch.Tensor, add_self_loops: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """
    Optional Kipf-GCN normalization per time step:
      Â_t = D^{-1/2} (A_t + I) D^{-1/2}
    A: (T, N, N)
    """
    T, N, _ = A.shape
    if add_self_loops:
        A = A + torch.eye(N, dtype=A.dtype, device=A.device).unsqueeze(0)
    deg = A.sum(dim=2).clamp_min(eps)           # (T, N)
    d_is = deg.pow(-0.5).unsqueeze(2)           # (T, N, 1)
    return d_is * A * d_is.transpose(1, 2)      # (T, N, N)



@variational_estimator
class DenseTemporalBayesGCN3_Min(nn.Module):
    """
    3-layer dense time-varying GCN with a single LayerNorm + single Dropout.
    Expects each Data sample to have:
      data.x : (N, T)
      data.A : (T, N, N)
      data.y : (1,) or (1,1)
    """
    def __init__(self,
                 h1: int = 128, h2: int = 64, h3: int = 32,
                 dropout: float = 0.5,
                 agg: str = "mean",         # "mean" or "attn"
                 norm: str = "none",        # "none" or "gcn"
                 add_self_loops: bool = True):
        super().__init__()
        # three dense "GCN" layers (pure matmul style)
        self.lin1 = nn.Linear(1,  h1, bias=False)
        self.lin2 = nn.Linear(h1, h2, bias=False)
        self.lin3 = nn.Linear(h2, h3, bias=False)

        # exactly one normalization and one dropout (applied after layer 3)
        self.ln = nn.LayerNorm(h3)
        self.dropout = nn.Dropout(dropout)

        # aggregator over time
        assert agg in {"mean", "attn"}
        self.agg = agg
        if agg == "attn":
            self.time_att = nn.Sequential(
                nn.Linear(h3, max(2, h3 // 2)), nn.Tanh(),
                nn.Linear(max(2, h3 // 2), 1)
            )

        # optional GCN normalization of A
        assert norm in {"none", "gcn"}
        self.norm = norm
        self.add_self_loops = add_self_loops

        # Bayesian head (32 -> 8 -> BLR -> 1)
        self.fc    = nn.Linear(h3, 8)
        self.bayes = BayesianLinear(8, 1)

        # init
        for m in (self.lin1, self.lin2, self.lin3):
            nn.init.xavier_uniform_(m.weight)
        self.act = nn.ELU()
    
    def set_A(self, A: torch.Tensor):
    # A: (T,N,N) on device
        self.register_buffer("A_buf", A)

    def forward(self, data):
        x: torch.Tensor = data.x        # (N, T)
        # A: torch.Tensor = data.A        # (T, N, N)
        A = getattr(data, "A", None)
        if A is None:
            A = self.A_buf                     # use preloaded buffer
        assert x.dim()==2 and A.dim()==3, f"x(N,T), A(T,N,N) required, got {x.shape}, {A.shape}"
        N, T = x.shape
        assert A.shape == (T, N, N), f"A must be (T,N,N); got {A.shape}"

        # choose adjacency operator
        A_use = gcn_norm(A, self.add_self_loops) if self.norm == "gcn" else A

        # per-time node features: (T, N, 1)
        X = x.T.unsqueeze(-1)

        # Layer 1: H1_t = A_t @ (X_t W1)
        Z1 = F.linear(X, self.lin1.weight, bias=None)  # (T, N, h1)
        H1 = torch.bmm(A_use, Z1)                      # (T, N, h1)
        H1 = self.act(H1)

        # Layer 2: H2_t = A_t @ (H1_t W2)
        Z2 = F.linear(H1, self.lin2.weight, bias=None) # (T, N, h2)
        H2 = torch.bmm(A_use, Z2)                      # (T, N, h2)
        H2 = self.act(H2)

        # Layer 3: H3_t = A_t @ (H2_t W3)
        Z3 = F.linear(H2, self.lin3.weight, bias=None) # (T, N, h3)
        H3 = torch.bmm(A_use, Z3)                      # (T, N, h3)
        H3 = self.act(H3)

        # ---- single normalization + single dropout (once) ----
        H3 = self.ln(H3)                  # (T, N, h3)
        H3 = self.dropout(H3)

        # aggregate over time -> (N, h3)
        if self.agg == "mean":
            H = H3.mean(dim=0)
        else:
            m_t = H3.mean(dim=1)                             # (T, h3)
            alpha = torch.softmax(self.time_att(m_t).squeeze(-1), dim=0)  # (T,)
            H = (H3 * alpha.view(T, 1, 1)).sum(dim=0)        # (N, h3)

        # pool nodes -> (h3,)
        g = H.mean(dim=0)

        # Bayesian head -> (1,1)
        z = self.act(self.fc(g))
        out = self.bayes(z).unsqueeze(0)
        return out
