import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------
# Linear Attention Layer
# ----------------------------
class LinearAttention(nn.Module):
    """Linear Self‑Attention via kernel trick.
    Complexity: O(N · d) where N is sequence length and d is hidden dim.
    """
    def __init__(self, dim: int, kernel_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj   = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.kernel_dim = kernel_dim
        # Optional linear projection for kernel
        self.kernel_proj = nn.Linear(dim, kernel_dim, bias=False)

    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Non‑negative kernel (ReLU + eps)."""
        return F.relu(x) + self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, dim]
        q = self.query_proj(x)  # [b, n, d]
        k = self.key_proj(x)    # [b, n, d]
        v = self.value_proj(x)  # [b, n, d]

        # Apply kernel to queries and keys
        qk = self.kernel(q)   # [b, n, d]
        kk = self.kernel(k)   # [b, n, d]

        # Compute K^T V per batch
        kv = torch.einsum('bnd,bmd->bnd', kk, v)  # [b, d, d]
        # Compute attention scores and apply to kv
        attn = torch.einsum('bnd,bnd->bd', qk, kv)  # [b, d]
        attn = attn.unsqueeze(1)  # [b, 1, d]
        out = attn  # linear attention output
        return out

# ----------------------------
# Hierarchical Reasoning Module
# ----------------------------
class HierarchicalReasoning(nn.Module):
    """A shallow Transformer stack to model reasoning steps.
    Depth is a small constant; each step operates on the same sequence.
    """
    def __init__(self, dim: int, depth: int = 2, num_heads: int = 4, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, dim]
        # Transformer expects shape [seq, batch, dim]
        x_t = rearrange(x, 'b n d -> n b d')
        out = self.encoder(x_t)
        out = rearrange(out, 'n b d -> b n d')
        return out

# ----------------------------
# Gated Fusion Layer
# ----------------------------
class GatedFusion(nn.Module):
    """Learned gating between linear attention and reasoning outputs."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, la: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # la, hr: [batch, seq, dim]
        combined = torch.cat([la, hr], dim=-1)  # [b, n, 2d]
        gate = self.gate_mlp(combined)  # [b, n, 1]
        out = gate * la + (1 - gate) * hr
        return out

# ----------------------------
# DeltaNet Architecture
# ----------------------------
class DeltaNet(nn.Module):
    """Hybrid Linear Attention + Hierarchical Reasoning network."""
    def __init__(self, dim: int = 128, **kwargs):
        super().__init__()
        self.dim = dim
        self.linear_attention = LinearAttention(dim=dim)
        self.hierarchical_reasoning = HierarchicalReasoning(dim=dim)
        self.fusion = GatedFusion(dim=dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [batch, seq, dim]
        la = self.linear_attention(x)  # [b, d]
        # Expand to match seq dimension
        la = rearrange(la, 'b d -> b 1 d')
        hr = self.hierarchical_reasoning(x)  # [b, seq, dim]
        # Broadcast la to seq
        la = la.repeat(1, hr.size(1), 1)
        out = self.fusion(la, hr)
        return out

# Alias for training utilities
Model = DeltaNet
