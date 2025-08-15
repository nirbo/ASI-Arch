# delta_net.py
"""DeltaNet Architecture

This implementation combines a linear attention mechanism with a simple hierarchical reasoning module (HRM).
The design follows the *nested* hybrid template: the input sequence is processed by linear attention, then passed through an HRM that performs multi‑level pooling and a small transformer block. The final representation is produced by fusing the two streams.

Key properties
- O(N log N) complexity: linear attention uses a kernel trick that reduces complexity from O(N^2) to O(N). The HRM uses a hierarchical pooling that is logarithmic in sequence length.
- Supports arbitrary batch size.
- Uses einops.rearrange for all tensor shape transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------------------------------------------------------
# Linear Attention Module
# -----------------------------------------------------------------------------
class LinearAttention(nn.Module):
    """Linear attention using the kernel trick (e.g. ReLU^2)."""
    def __init__(self, dim, heads=8, kernel_dim=64):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.kernel_dim = kernel_dim
        # Simple ReLU^2 kernel
        self.kernel = nn.Sequential(
            nn.Linear(dim, kernel_dim, bias=False),
            nn.ReLU(),
            nn.Linear(kernel_dim, kernel_dim, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        # reshape to [B, heads, T, head_dim]
        q = rearrange(q, "b t (h d) -> b h t d", h=self.heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.heads)

        # Apply kernel function
        qk = self.kernel(q)  # [B, heads, T, K]
        kk = self.kernel(k)  # [B, heads, T, K]

        # Compute normalization
        kv = torch.einsum("bhtk,bhtd->bhdk", kk, v)  # [B, heads, d, K]
        z = 1.0 / (torch.einsum("bhtk,bhtk->bht", qk, kk) + 1e-6)
        out = torch.einsum("bhtk,bhdk->bhtd", qk, kv) * z
        out = rearrange(out, "b h t d -> b t (h d)")
        return out

# -----------------------------------------------------------------------------
# Hierarchical Reasoning Module (HRM)
# -----------------------------------------------------------------------------
class HierarchicalReasoning(nn.Module):
    """A lightweight hierarchical reasoning block.
    Performs pooling at multiple levels and a single transformer encoder block.
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, T, D]
        # hierarchical pooling: log2(T) levels
        pooled = x
        for _ in range(3):  # 3 levels, adjust as needed
            # reshape for pooling: [B, D, T]
            pooled = rearrange(pooled, "b t d -> b d t")
            pooled = self.pool(pooled)
            pooled = rearrange(pooled, "b d t -> b t d")
            # add positional encoding or skip
        # transformer block
        attn_out, _ = self.attn(pooled, pooled, pooled)
        x = self.norm1(attn_out + pooled)
        ff_out = self.ff(x)
        out = self.norm2(ff_out + x)
        return out

# -----------------------------------------------------------------------------
# DeltaNet Class
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):
    def __init__(self, dim=128, heads=8, kernel_dim=64, h_heads=4):
        super().__init__()
        self.linear_attn = LinearAttention(dim=dim, heads=heads, kernel_dim=kernel_dim)
        self.hrm = HierarchicalReasoning(dim=dim, heads=h_heads)
        self.fusion = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        # x: [B, T, D]
        attn_out = self.linear_attn(x)
        hrm_out = self.hrm(x)
        # fuse via concatenation
        fused = torch.cat([attn_out, hrm_out], dim=-1)
        fused = self.fusion(fused)
        out = self.norm(fused)
        return out

# Alias for training compatibility
Model = DeltaNet

# -----------------------------------------------------------------------------
# Simple test (optional) – remove when deploying
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, D = 2, 64, 128
    x = torch.randn(B, T, D)
    net = DeltaNet(dim=D)
    y = net(x)
    print("output shape:", y.shape)
