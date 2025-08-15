import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------
# DeltaNet – Hierarchical Adaptive Attention with Dynamic Fusion Gate
# ---------------------------------------------------------------------------
# This architecture combines a linear‑attention backbone (O(N log N)) with a
# two‑stage attention hierarchy. The first stage selects key tokens via a
# lightweight linear attention; the second stage applies a full multi‑head
# attention only on the selected tokens. A learned fusion gate merges the fast
# linear‑attention stream with the richer hierarchical reasoning stream.
# ---------------------------------------------------------------------------

class DeltaNet(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 4,
                 top_k: int = 32,
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 **kwargs):
        """DeltaNet constructor.

        Parameters
        ----------
        dim : int
            Embedding dimension of the input tokens.
        num_heads : int, optional
            Number of heads for the hierarchical multi‑head attention.
        top_k : int, optional
            Number of tokens to select for the second‑stage attention.
        hidden_dim : int, optional
            Hidden dimension used in the fusion gate MLP.
        dropout : float, optional
            Dropout probability.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        # Linear‑attention projection layers
        self.query_lin = nn.Linear(dim, dim)
        self.key_lin = nn.Linear(dim, dim)
        self.value_lin = nn.Linear(dim, dim)

        # Feature map for linear attention (Performer style)
        self.feature_map = lambda x: F.relu(x) + 1.0

        # Second‑stage multi‑head attention (full quadratic on small K)
        self.h_attn = nn.MultiheadAttention(embed_dim=dim,
                                             num_heads=num_heads,
                                             dropout=dropout,
                                             batch_first=True)

        # Hierarchical reasoning transformer encoder layer
        self.hier_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                     nhead=num_heads,
                                                     dim_feedforward=hidden_dim,
                                                     dropout=dropout,
                                                     batch_first=True)

        # Fusion gate MLP (takes concatenated L & H representations)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Output projection (for classification/regression tasks)
        self.out_proj = nn.Linear(dim, dim)

    def linear_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Fast linear attention.

        Implements a kernel‑based linear attention as in Performer.
        Complexity: O(N log N) due to the feature map.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, dim).

        Returns
        -------
        torch.Tensor
            Output of the linear attention, same shape as input.
        """
        # Project to query/key/value
        q = self.query_lin(x)  # (B, N, dim)
        k = self.key_lin(x)
        v = self.value_lin(x)

        # Apply feature map
        phi_q = self.feature_map(q)  # (B, N, dim)
        phi_k = self.feature_map(k)

        # Compute context: (phi_k^T @ v) for each batch
        # Use einsum for efficient batched multiplication
        context = torch.einsum('bnd,bnd->bd', phi_k, v)  # (B, dim)

        # Output: (phi_q @ context) normalized by sum of phi_q
        out = torch.einsum('bnd,bd->bnd', phi_q, context)  # (B, N, dim)

        return out

    def select_topk(self, scores: torch.Tensor, top_k: int) -> torch.Tensor:
        """Select top‑k token indices based on scores.

        Parameters
        ----------
        scores : torch.Tensor
            Tensor of shape (B, N) containing importance scores.
        top_k : int
            Number of tokens to keep.

        Returns
        -------
        torch.Tensor
            Indices of shape (B, top_k).
        """
        _, idx = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=False)
        return idx

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of DeltaNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, dim).
        """
        B, N, _ = x.shape
        device = x.device

        # 1. Fast linear‑attention stream
        l_out = self.linear_attention(x)  # (B, N, dim)

        # 2. Compute token importance scores (sum over feature dim)
        scores = l_out.sum(dim=-1)  # (B, N)
        topk_idx = self.select_topk(scores, self.top_k)  # (B, top_k)

        # 3. Gather selected tokens for second‑stage attention
        # Expand indices for gathering feature dim
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, self.dim)
        selected = torch.gather(l_out, dim=1, index=idx_expanded)  # (B, top_k, dim)

        # 4. Hierarchical (full) attention on selected tokens
        # MultiheadAttention expects (B, L, E) when batch_first=True
        h_out, _ = self.h_attn(selected, selected, selected)  # (B, top_k, dim)

        # 5. Global context from H‑module (mean over selected tokens)
        h_global = h_out.mean(dim=1, keepdim=True)  # (B, 1, dim)
        h_global_expanded = h_global.expand(-1, N, -1)  # (B, N, dim)

        # 6. Dynamic fusion gate
        fusion_input = torch.cat([l_out, h_global_expanded], dim=-1)  # (B, N, 2*dim)
        gate = self.fusion_mlp(fusion_input).expand(-1, -1, self.dim)  # (B, N, 1) -> (B, N, dim)

        fused = gate * l_out + (1 - gate) * h_global_expanded  # (B, N, dim)

        # 7. Hierarchical reasoning transformer layer
        hier = self.hier_layer(fused)  # (B, N, dim)

        # 8. Final projection (optional downstream head)
        out = self.out_proj(hier)  # (B, N, dim)

        return out

# Alias for training utilities
Model = DeltaNet
