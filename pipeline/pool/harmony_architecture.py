import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LinearAttention(nn.Module):
    """Linear attention mechanism based on kernel trick.
    Computes Q @ (Kᵀ V) which is linear in the sequence length.
    """
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.proj = nn.Linear(dim, heads * dim_head * 3, bias=True)
        self.out_proj = nn.Linear(heads * dim_head, dim, bias=True)

    def forward(self, x):
        # x: (batch, seq, dim)
        batch, seq, _ = x.size()
        # Linear projections
        qkv = self.proj(x)  # (batch, seq, heads*dim_head*3)
        # Rearrange to separate heads and Q/K/V
        qkv = rearrange(qkv, 'b s (h d3) -> b h s d3', h=self.heads, d3=3 * self.dim_head)
        q, k, v = qkv[:, :, :, :self.dim_head], qkv[:, :, :, self.dim_head:2 * self.dim_head], qkv[:, :, :, 2 * self.dim_head:]
        # Optional: apply activation to ensure positivity if using kernel
        # Here we keep raw linear projections for simplicity
        # Compute KV matrix per head
        kv = torch.matmul(k.transpose(-2, -1), v)  # (batch, heads, dim_head, dim_head)
        # Compute attention output
        out = torch.matmul(q, kv)  # (batch, heads, seq, dim_head)
        out = rearrange(out, 'b h s d -> b s (h d)')  # (batch, seq, heads*dim_head)
        out = self.out_proj(out)  # (batch, seq, dim)
        return out


class HRM(nn.Module):
    """Hierarchical Reasoning Module.
    Performs global pooling, MLP reasoning, and broadcasts back to token level.
    """
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # mean pooling over sequence
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        # x: (batch, seq, dim)
        # Global context
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (batch, dim)
        # Reasoning MLP
        reasoning = self.mlp(pooled)  # (batch, dim)
        # Broadcast to sequence length
        reasoning = reasoning.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq, dim)
        return reasoning


class DeltaNet(nn.Module):
    """DeltaNet combines linear attention and hierarchical reasoning.
    Architecture: Linear Attention → HRM reasoning → fused output.
    """
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.linear_attn = LinearAttention(dim, heads, dim_head)
        self.hrm = HRM(dim)
        # Optionally, a final projection can be added
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, **kwargs):
        # x: (batch, seq, dim)
        attn_out = self.linear_attn(x)
        hrm_out = self.hrm(x)
        fused = attn_out + hrm_out
        out = self.out_proj(fused)
        return out

# Alias for training utilities
Model = DeltaNet
