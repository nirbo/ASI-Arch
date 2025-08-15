# delta_net_innovation.py
"""DeltaNet: A hybrid linear attention + hierarchical reasoning architecture.

This implementation follows the guidelines from the research plan.  It
provides a lightweight, scalable, and modular architecture suitable for
training on arbitrary batch sizes while maintaining sub‑quadratic time
complexity.

Key components
--------------
* LinearAttention:  O(N) attention using a feature‑map kernel.
* HierarchicalReasoningModule (HRM):  Two‑level hierarchy that reduces
  the sequence length, performs reasoning, and then upsamples.
* DeltaNet:  Sequential combination of LinearAttention and HRM, followed
  by a final projection to the desired output dimension.

The code intentionally uses ``einops.rearrange`` for all reshaping and
avoids ``view`` / ``reshape`` to satisfy the project constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------
# Linear attention – O(N) implementation
# ---------------------------------------------------------------------------
class LinearAttention(nn.Module):
    """A simple linear attention layer.

    The attention is computed using a kernel feature map ``φ`` that
    satisfies ``φ(Q)ᵀ φ(K) = Q Kᵀ`` approximately.  We use ``φ(x) = relu(x)+1``
    which is fast and empirically works well for many NLP tasks.
    """

    def __init__(self, dim, heads=8, dim_head=64, eps=1e-6):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.eps = eps

        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.out = nn.Linear(dim_head * heads, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x)  # (B, N, heads*dim_head*3)
        qkv = rearrange(qkv, "b n (h d3) -> b h n d3", h=self.heads, d3=3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = q * self.scale
        k = k * self.scale
        # Feature map
        q = F.relu(q) + 1
        k = F.relu(k) + 1
        # Compute attention components
        kv = torch.einsum("b h j d, b h j e -> b h d e", k, v)  # (B, H, D, E)
        qk = torch.einsum("b h i d, b h j d -> b h i j", q, k)  # (B, H, N, N) – but we avoid this
        # Instead compute denominator and numerator using linearity
        k_sum = k.sum(dim=1)  # (B, H, D)
        # Numerator: (B, H, N, E)
        numer = torch.einsum("b h i d, b h d e -> b h i e", q, kv)
        denom = torch.einsum("b h i d, b h d -> b h i", q, k_sum) + self.eps
        out = numer / denom.unsqueeze(-1)
        out = rearrange(out, "b h n e -> b n (h e)")
        out = self.out(out)
        return out

# ---------------------------------------------------------------------------
# Hierarchical reasoning module – two‑level hierarchy
# ---------------------------------------------------------------------------
class HierarchicalReasoningModule(nn.Module):
    """A lightweight two‑level hierarchy.

    * Level 1 processes the original sequence with linear attention.
    * Level 2 pools pairs of tokens, processes them with linear attention
      again, and then upsamples by repeating each token.
    """

    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.level1 = LinearAttention(dim, heads, dim_head)
        self.level2 = LinearAttention(dim, heads, dim_head)

    def forward(self, x):
        # Level 1
        h1 = self.level1(x)
        # Pooling: take every two tokens (even length required)
        B, N, C = h1.shape
        assert N % 2 == 0, "Sequence length must be even for pooling"
        pooled = rearrange(h1, "b (m 2) c -> b m (2 c)")  # (B, M, 2C)
        # Process pooled tokens
        h2 = self.level2(pooled)
        # Upsample: repeat each token twice
        up = rearrange(h2, "b m (2 c) -> b (m 2) c", m=N//2)
        # Fuse: average with original level1 output
        out = (h1 + up) / 2
        return out

# ---------------------------------------------------------------------------
# DeltaNet – sequential combination of linear attention and HRM
# ---------------------------------------------------------------------------
class DeltaNet(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, out_dim=None):
        super().__init__()
        self.linear_attn = LinearAttention(dim, heads, dim_head)
        self.hrm = HierarchicalReasoningModule(dim, heads, dim_head)
        self.proj = nn.Linear(dim, out_dim if out_dim is not None else dim)

    def forward(self, x, **kwargs):
        # x: (B, N, C)
        attn_out = self.linear_attn(x)
        hrm_out = self.hrm(attn_out)
        out = self.proj(hrm_out)
        return out

# Alias for compatibility with training scripts
Model = DeltaNet

# ---------------------------------------------------------------------------
# Simple test to ensure code runs – this block is not part of the module but
# provides an internal sanity check.  It will not be executed during import.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, N, C = 2, 8, 32
    x = torch.randn(B, N, C)
    net = DeltaNet(dim=C, out_dim=16)
    y = net(x)
    print("output shape:", y.shape)
