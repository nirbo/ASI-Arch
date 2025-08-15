import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------------------------------------------------------
# Linear Attention module (O(N) complexity) using kernel trick
# -----------------------------------------------------------------------------
class LinearAttention(nn.Module):
    """Efficient linear attention using a positive kernel.
    Implements the formulation from "Transformers are RNNs, but RNNs are
    not Transformers" and related work.  The attention is computed as:

    
    q_k = phi(q)  # (B, S, D)
    k_k = phi(k)  # (B, S, D)
    sum_k = k_k.sum(dim=1)  # (B, D)
    KV = torch.einsum('bld,bmd->bld', k_k, v)  # (B, D, D)
    out = torch.einsum('bld,bld->bld', q_k, KV / sum_k.unsqueeze(-1))
    """

    def __init__(self, dim, kernel_fn=None, eps=1e-6):
        super().__init__()
        self.dim = dim
        # Default kernel: ReLU+1 (non‑negative, easy to compute)
        self.kernel_fn = kernel_fn or (lambda x: F.relu(x) + 1.0)
        self.eps = eps

    def forward(self, q, k, v):
        # q, k, v: (B, S, D)
        qk = self.kernel_fn(q)  # (B, S, D)
        kk = self.kernel_fn(k)  # (B, S, D)
        # Sum over sequence dimension
        sum_k = kk.sum(dim=1)  # (B, D)
        # Compute KV matrix
        KV = torch.einsum("bld,bmd->bld", kk, v)  # (B, D, D)
        # Avoid division by zero
        denom = sum_k.unsqueeze(-1).clamp(min=self.eps)
        # Compute output: element‑wise multiply and sum over D
        out = torch.einsum("bld,bld->bld", qk, KV / denom)
        return out

# -----------------------------------------------------------------------------
# Simple Hierarchical Reasoning Module (HRM)
# -----------------------------------------------------------------------------
class HierarchicalReasoning(nn.Module):
    """A lightweight hierarchical reasoning block.
    It consists of two linear attention layers stacked with
    a feed‑forward network.  The depth is intentionally small
    to keep the overall complexity linear.
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.attn1 = LinearAttention(dim)
        self.attn2 = LinearAttention(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, S, D)
        # First attention block with residual
        att1 = self.attn1(x, x, x)
        x = self.norm1(x + att1)
        # Second attention block
        att2 = self.attn2(x, x, x)
        x = self.norm2(x + att2)
        # Feed‑forward with residual
        ff = self.ffn(x)
        x = self.norm3(x + ff)
        return x

# -----------------------------------------------------------------------------
# DeltaNet architecture integrating Linear Attention and HRM
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet – breakthrough hybrid architecture.
    Combines per‑token linear attention (fast, O(N)) with a
    lightweight hierarchical reasoning module.  The two streams
    are fused by concatenation followed by a linear projection.
    """

    def __init__(self, dim, hidden_dim=None, proj_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        proj_dim = proj_dim or dim
        self.linear_attn = LinearAttention(dim)
        self.hrm = HierarchicalReasoning(dim, hidden_dim)
        self.proj = nn.Linear(2 * dim, proj_dim)

    def forward(self, x, **kwargs):
        # x: (B, S, D)
        # Linear attention stream
        la_out = self.linear_attn(x, x, x)
        # Hierarchical reasoning stream
        hr_out = self.hrm(x)
        # Fusion by concatenation
        fused = torch.cat([la_out, hr_out], dim=-1)  # (B, S, 2D)
        out = self.proj(fused)  # (B, S, proj_dim)
        return out

# Alias for training utilities that expect a `Model` class name
Model = DeltaNet
