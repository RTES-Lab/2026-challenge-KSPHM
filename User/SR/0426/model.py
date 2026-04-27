"""
Adaptive MAGNN-TCN for bearing RUL prediction.

Reference: Y. Ye, J. Wang, J. Yang, D. Yao, T. Zhou,
"Adaptive MAGNN-TCN: An Innovative Approach for Bearings Remaining Useful Life
Prediction," IEEE Sensors J. 25(4), 2025.

Components implemented:
  • Multiscale pyramid feature extractor (1D, K=3 layers, stride-2 downsamples)
  • AGL (Adaptive Graph Learning) module — per-scale adjacency from learnable
    node + scale embeddings + topK sparsification
  • GCN per scale on the (variables × time) feature map
  • Gated multiscale fusion + dilated TCN
  • Regression head -> normalized RUL

Input shape: (B, N_vars, T)   -- N_vars = number of features per file
                                  T      = window length (number of files)
Output:      (B, 1)            -- RUL in [0,1] (normalized lifetime)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Adaptive Graph Learning — per-scale dynamic adjacency
# ------------------------------------------------------------------
class AGL(nn.Module):
    """
    Per-scale adjacency over N variables.

      E_spec_k = E_nodes ⊙ E_scale_k          (N, d_e)
      M1_k     = tanh(E_spec_k · θ_k)         (N,)  -- broadcasted
      M2_k     = tanh(E_spec_k · φ_k)         (N,)
      A_full   = ReLU(M1 · M2^T - M2 · M1^T)
      A_k      = topK(softmax(A_full), k=top_k)
    """

    def __init__(self, n_nodes: int, n_scales: int, d_emb: int = 16, top_k: int = 8):
        super().__init__()
        self.n_nodes, self.n_scales, self.top_k = n_nodes, n_scales, top_k
        self.E_nodes = nn.Parameter(torch.randn(n_nodes, d_emb) * 0.1)
        self.E_scale = nn.Parameter(torch.randn(n_scales, d_emb) * 0.1)
        self.theta = nn.Parameter(torch.randn(n_scales, d_emb) * 0.1)
        self.phi   = nn.Parameter(torch.randn(n_scales, d_emb) * 0.1)

    def forward(self) -> list[torch.Tensor]:
        """Returns list of K sparse adjacency matrices, each (N, N)."""
        out = []
        for k in range(self.n_scales):
            E_spec = self.E_nodes * self.E_scale[k].unsqueeze(0)        # (N, d)
            M1 = torch.tanh(E_spec @ self.theta[k])                      # (N,)
            M2 = torch.tanh(E_spec @ self.phi[k])                        # (N,)
            # rank-1 difference => skew-symmetric => oriented adjacency
            A = F.relu(torch.outer(M1, M2) - torch.outer(M2, M1))        # (N, N)
            A = F.softmax(A, dim=-1)
            # top-K sparsification per row
            top_k = min(self.top_k, self.n_nodes)
            vals, idx = torch.topk(A, k=top_k, dim=-1)
            A_sparse = torch.zeros_like(A)
            A_sparse.scatter_(1, idx, vals)
            # add self-loop and symmetric-normalize
            A_sparse = A_sparse + torch.eye(self.n_nodes, device=A.device)
            d = A_sparse.sum(dim=-1).clamp_min(1e-6)
            d_inv_sqrt = d.pow(-0.5)
            A_norm = A_sparse * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)
            out.append(A_norm)
        return out


# ------------------------------------------------------------------
# Multiscale pyramid 1D feature extractor
# ------------------------------------------------------------------
class PyramidLayer(nn.Module):
    """1×1 conv (channel reduction) + stride-2 downsample (1D conv)."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1)
        self.down = nn.Conv1d(c_out, c_out, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.proj(x))
        x = self.act(self.down(x))
        return x


class MultiscalePyramid(nn.Module):
    """Returns list of K feature maps with progressively halved time length."""
    def __init__(self, c_in: int, c_hidden: int, n_scales: int):
        super().__init__()
        self.layers = nn.ModuleList()
        c_prev = c_in
        for k in range(n_scales):
            self.layers.append(PyramidLayer(c_prev, c_hidden))
            c_prev = c_hidden
        self.n_scales = n_scales

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return outs   # list of (B, C, T_k)


# ------------------------------------------------------------------
# Graph convolution per scale: aggregate over the *variables* axis
# ------------------------------------------------------------------
class GCNScale(nn.Module):
    """
    Treat each scale's feature map (B, N, T_k) as N nodes of a graph,
    where each node carries a T_k-dim signal. Apply A_k on the N-axis.
    """
    def __init__(self, t_dim: int):
        super().__init__()
        self.W = nn.Linear(t_dim, t_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # h: (B, N, T_k); A: (N, N)
        h_lin = self.W(h)
        h_out = torch.einsum("ij,bjt->bit", A, h_lin)
        return self.act(h_out)


# ------------------------------------------------------------------
# Dilated causal TCN
# ------------------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, p_drop=0.1):
        super().__init__()
        pad = (k - 1) * d
        self.pad = pad
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, dilation=d)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, padding=pad, dilation=d)
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def _causal(self, x, conv):
        y = conv(x)
        if self.pad:
            y = y[..., :-self.pad]
        return y

    def forward(self, x):
        y = F.relu(self._causal(x, self.conv1))
        y = self.drop(y)
        y = F.relu(self._causal(y, self.conv2))
        y = self.drop(y)
        return F.relu(y + self.proj(x))


class TCN(nn.Module):
    def __init__(self, c_in, c_hidden, n_blocks=4, k=3, p_drop=0.1):
        super().__init__()
        layers = []
        for i in range(n_blocks):
            layers.append(TemporalBlock(c_in if i == 0 else c_hidden,
                                        c_hidden, k, d=2 ** i, p_drop=p_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------
class AdaptiveMAGNN_TCN(nn.Module):
    """
    Full Adaptive MAGNN-TCN.

    Forward:
      x : (B, N_vars, T)
      |- multiscale pyramid: list of (B, c_h, T_k)  k=1..K
      |- per-scale GCN over the *variables* axis using A_k from AGL
      |  (we transpose so the variables axis is the graph axis)
      |- gated fusion across scales => single (B, c_f, T_out)
      |- TCN over the temporal axis
      |- global pool + FC -> normalized RUL
    """

    def __init__(self,
                 n_vars: int,
                 t_in: int = 16,
                 c_hidden: int = 32,
                 n_scales: int = 3,
                 d_emb: int = 16,
                 top_k: int = 8,
                 tcn_hidden: int = 64,
                 tcn_blocks: int = 4,
                 p_drop: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.n_scales = n_scales

        # multi-scale temporal pyramid (channels = variables; time shrinks ×2 per layer)
        self.pyramid = MultiscalePyramid(c_in=n_vars, c_hidden=n_vars, n_scales=n_scales)

        # adaptive graph learning per scale
        self.agl = AGL(n_nodes=n_vars, n_scales=n_scales, d_emb=d_emb, top_k=top_k)

        # GCN per scale (note: each scale has different time length T_k)
        self.gcns = nn.ModuleList()
        t_k = t_in
        self.scale_t = []
        for k in range(n_scales):
            t_k = max(t_k // 2, 1)
            self.scale_t.append(t_k)
            self.gcns.append(GCNScale(t_dim=t_k))

        # interpolate all scales to the smallest T_k for fusion
        self.t_fuse = self.scale_t[-1]

        # gated fusion across scales: produce per-scale weights then sum
        self.gate = nn.Linear(n_vars * n_scales, n_scales)

        # TCN on the fused (B, n_vars, t_fuse)
        self.tcn = TCN(c_in=n_vars, c_hidden=tcn_hidden,
                       n_blocks=tcn_blocks, k=3, p_drop=p_drop)

        # head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(tcn_hidden, tcn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(tcn_hidden, 1),
            nn.Sigmoid(),                # RUL in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T)
        B = x.size(0)
        scales = self.pyramid(x)         # list of (B, N, T_k)
        As = self.agl()                  # list of (N, N)

        gcn_outs = []
        for k, (h, A, gcn) in enumerate(zip(scales, As, self.gcns)):
            # ensure h has expected T_k (pad/clip to the configured size)
            if h.size(-1) != self.scale_t[k]:
                h = F.adaptive_avg_pool1d(h, self.scale_t[k])
            h = gcn(h, A)                # (B, N, T_k)
            gcn_outs.append(h)

        # interpolate every scale to t_fuse
        aligned = [F.adaptive_avg_pool1d(h, self.t_fuse) for h in gcn_outs]

        # gated fusion: weights per scale, derived from each scale's mean
        scale_means = torch.stack([h.mean(-1) for h in aligned], dim=-1)  # (B, N, K)
        gate_in = scale_means.flatten(1)                                  # (B, N*K)
        w = F.softmax(self.gate(gate_in), dim=-1)                         # (B, K)
        # weighted sum across scales
        stacked = torch.stack(aligned, dim=-1)                            # (B,N,Tf,K)
        fused = (stacked * w.view(B, 1, 1, -1)).sum(-1)                   # (B,N,Tf)

        # temporal model
        z = self.tcn(fused)              # (B, tcn_hidden, Tf)
        return self.head(z).squeeze(-1)  # (B,)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, T = 8, 64, 16
    model = AdaptiveMAGNN_TCN(n_vars=N, t_in=T, c_hidden=32, n_scales=3)
    x = torch.randn(B, N, T)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"input  : {tuple(x.shape)}")
    print(f"output : {tuple(y.shape)}  range=[{y.min():.3f},{y.max():.3f}]")
    print(f"params : {n_params:,}")
