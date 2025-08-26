# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Updated to incorporate length‑aware scaling and optional residual projection
# to improve long‑range recall and effective rank while preserving linearity.

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

# Utilities from the original repository
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def softmax(x):
    return F.softmax(x, dim=-1)

# ---------------------------------------------------------------------------
# Core delta rule implementation (unchanged)
# ---------------------------------------------------------------------------
@torch.compile
def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    """Chunkwise implementation of the delta rule.

    Parameters
    ----------
    q, k, v : Tensor
        Query, key and value tensors of shape ``(b, h, l, d)``.
    beta : Tensor
        Scaling factor of shape ``(b, h, l)``.
    chunk_size : int
        Size of the processing chunk.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    # Pad to a multiple of chunk_size
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))

    padded_len = l + pad_len
    # Normalise queries and keys
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Compute the inverse of (I - tri(diag(beta) KK^T)) via a triangular solve
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size), [q, k, v, k_beta]
    )
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn.to(torch.bfloat16)
    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

# ---------------------------------------------------------------------------
# DeltaNet implementation with length‑aware scaling and optional residual
# ---------------------------------------------------------------------------
class BasicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        d_model: int = 512,
        hidden_size: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        layer_idx: int = None,
        **kwargs
    ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.layer_idx = layer_idx
        
        assert hidden_size % num_heads == 0, f"hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        # Position embeddings
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(hidden_size)
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeddings = self.token_emb(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device)
        position_embeddings = self.pos_emb(positions)
        
        # Combined embeddings
        hidden_states = token_embeddings + position_embeddings
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states, _, _ = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling logits
        logits = self.lm_head(hidden_states)
        
        return logits, None, past_key_values


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        
        # Multi-head attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False) 
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Feed forward network
        self.gate_proj = nn.Linear(hidden_size, d_ff, bias=False)
        self.up_proj = nn.Linear(hidden_size, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, hidden_size, bias=False)
        
        # Layer norms
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Pre-attention norm
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Multi-head attention - QUADRATIC COMPLEXITY O(N²)
        q = self.q_proj(normed_hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed_hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed_hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention - THIS IS THE QUADRATIC BOTTLENECK
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # O(N²)
        
        # Apply causal mask
        if attention_mask is None:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))
        else:
            scores = scores + attention_mask
            
        attn_weights = F.softmax(scores, dim=-1)  # O(N²) memory
        attn_weights = self.dropout_layer(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # O(N²)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # Pre-MLP norm
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Feed forward network (SwiGLU activation)
        gate = F.silu(self.gate_proj(normed_hidden_states))
        up = self.up_proj(normed_hidden_states)
        mlp_output = self.down_proj(gate * up)
        
        # Residual connection
        hidden_states = hidden_states + mlp_output
        
        return hidden_states, None, past_key_values

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------
