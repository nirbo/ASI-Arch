# delta_net.py
"""DeltaNet Architecture

A lightweight, hybrid linear attention + hierarchical reasoning model.
Designed to achieve O(N log N) complexity while supporting arbitrary batch size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----- Linear Attention Module -----
class LinearAttention(nn.Module):
    """Linear attention using kernel trick for O(Nd) complexity.
    Implements the formulation from Reformer and Linear Transformers.
    """
    def __init__(self, dim, heads=4, dim_head=32, kernel_fn=F.elu):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.kernel_fn = kernel_fn
        self.out_proj = nn.Linear(inner_dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.to_qkv(x)  # (B, L, 3*heads*dim_head)
        qkv = rearrange(qkv, "b l (q k v) h d -> (q k v) b h l d", q=3, k=1, v=1, h=self.heads, d=self.dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, L, dim_head)

        # Apply kernel function
        q = self.kernel_fn(q + 1e-6)  # avoid negative values
        k = self.kernel_fn(k + 1e-6)

        # Compute KV^T and QK^T * V efficiently
        kv = torch.einsum("b h l d, b h l e -> b h d e", k, v)  # (B, heads, dim_head, dim_head)
        z = torch.inverse(torch.einsum("b h l d, b h l e -> b h d e", k, torch.ones_like(v)))  # normalization
        out = torch.einsum("b h l d, b h d e -> b h l e", q, kv) * z
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out_proj(out)

# ----- Hierarchical Reasoning Module -----
class HierarchicalReasoning(nn.Module):
    """Simple two-level hierarchical reasoning.
    First level: local linear attention over token groups.
    Second level: linear attention over group representations.
    """
    def __init__(self, dim, group_size=16, heads=4, dim_head=32):
        super().__init__()
        self.group_size = group_size
        self.local_attn = LinearAttention(dim, heads=heads, dim_head=dim_head)
        self.global_attn = LinearAttention(dim, heads=heads, dim_head=dim_head)

    def forward(self, x):
        B, L, C = x.shape
        # Pad to multiple of group_size
        pad_len = (self.group_size - L % self.group_size) % self.group_size
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, C, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x
        Lp = x_padded.shape[1]
        G = Lp // self.group_size
        # Reshape into groups
        groups = rearrange(x_padded, "b (g s) c -> b g s c", g=G, s=self.group_size)
        # Local attention per group
        local_out = self.local_attn(groups)  # (B, G, group_size, C)
        # Aggregate group representations (mean over tokens)
        group_repr = local_out.mean(dim=2)  # (B, G, C)
        # Global attention over groups
        global_out = self.global_attn(group_repr)  # (B, G, C)
        # Expand back to tokens
        expanded = rearrange(global_out, "b g c -> b g 1 c")
        expanded = expanded.repeat_interleave(self.group_size, dim=2)
        expanded = expanded[:, :, :L, :]
        # Residual connection with original
        return x + expanded

# ----- DeltaNet -----
class DeltaNet(nn.Module):
    """DeltaNet combines linear attention and hierarchical reasoning.
    Supports arbitrary batch size and achieves O(N log N) complexity.
    """
    def __init__(self, dim, heads=4, dim_head=32, group_size=16, **kwargs):
        super().__init__()
        self.linear_attn = LinearAttention(dim, heads=heads, dim_head=dim_head)
        self.hrm = HierarchicalReasoning(dim, group_size=group_size, heads=heads, dim_head=dim_head)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, **kwargs):
        # x: (B, L, C)
        attn_out = self.linear_attn(x)
        hrm_out = self.hrm(x)
        fused = attn_out + hrm_out
        fused = self.norm(fused)
        ff_out = self.ff(fused)
        return ff_out + fused

# Alias for training scripts
Model = DeltaNet
