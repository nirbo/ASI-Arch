import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class LinearAttention(nn.Module):
    """Linear attention using a simple kernel approximation.
    Complexity: O(N * D) per batch.
    """
    def __init__(self, d_model: int, phi=nn.ReLU()):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.phi = phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        phi_q = self.phi(q)
        phi_k = self.phi(k)
        # Compute KV = phi(K)^T @ V
        # phi_k: [B, L, D]
        # v: [B, L, D]
        kv = torch.bmm(phi_k.transpose(1, 2), v)  # [B, D, D]
        out = torch.bmm(phi_q, kv)  # [B, L, D]
        return out


class HierarchicalReasoning(nn.Module):
    """A simple hierarchical reasoning module.
    Each level applies average pooling to reduce sequence length by ~2x,
    processes with a small MLP, then upsamples by repeating tokens.
    Complexity: O(N log N).
    """
    def __init__(self, d_model: int, n_levels: int, mlp_dim: int):
        super().__init__()
        self.n_levels = n_levels
        self.blocks = nn.ModuleList()
        for _ in range(n_levels):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(d_model, mlp_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_dim, d_model),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        current = x
        # Down-sampling path
        for block in self.blocks:
            # Average pooling to halve length
            # Transpose for conv1d: [B, D, L]
            current_t = current.transpose(1, 2)
            pooled = F.avg_pool1d(current_t, kernel_size=2, stride=2, ceil_mode=True)
            current = pooled.transpose(1, 2)  # back to [B, L', D]
            current = block(current)
        # Up-sampling path
        for block in reversed(self.blocks):
            # Repeat each token twice
            current = repeat(current, "b l d -> b (l r) d", r=2)
            current = block(current)
        return current


class DeltaNet(nn.Module):
    """DeltaNet: hybrid linear attention + hierarchical reasoning.
    Supports arbitrary batch sizes and maintains O(N log N) complexity.
    """
    def __init__(self, d_model: int = 512, n_levels: int = 3, mlp_dim: int = 1024, phi=nn.ReLU()):
        super().__init__()
        self.linear_attn = LinearAttention(d_model, phi=phi)
        self.hr_module = HierarchicalReasoning(d_model, n_levels, mlp_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, L, D]
        attn_out = self.linear_attn(x)
        hr_out = self.hr_module(x)
        fused = torch.cat([attn_out, hr_out], dim=-1)  # [B, L, 2D]
        out = self.fusion(fused)  # [B, L, D]
        return out

# Alias for training compatibility
Model = DeltaNet
