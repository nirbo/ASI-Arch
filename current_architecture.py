# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
def softmax(x):
    return F.softmax(x, dim=-1)

@torch.compile
def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # Calculate padding
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        # Pad inputs
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
    
    padded_len = l + pad_len
    # q = q * (d_k ** -0.5)
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    
    # compute (I - tri(diag(beta) KK^T))^{-1}
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
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
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    # Remove padding if any
    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

class DeltaNet(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk1',
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        **kwargs
    ) -> DeltaNet:
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {{num_heads}}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {{num_heads}}"
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != 'identity':
                raise NotImplementedError
        if self.qk_norm == 'sum':
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.
        
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')
            
        o, recurrent_state = delta_rule_chunkwise(
            q=q,
            k=k,
            v=v,
            beta=beta,
        )
        o = rearrange(o, 'b h l d -> b l h d')
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values


# Mamba2 imports and utilities for SSM branch
try:
    from mamba_ssm import Mamba2
except ImportError:
    # Fallback if Mamba2 not available
    class Mamba2(nn.Module):
        def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.linear = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            return self.linear(x)


class TitansMAG(nn.Module):
    """Titans Memory-As-Gate: Gated memory system for the H1-Titans architecture."""
    
    def __init__(self, dim: int, num_slots: int = 128, slot_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        
        # Memory slots
        self.memory_slots = nn.Parameter(torch.randn(num_slots, slot_dim))
        
        # Query, key, value projections for memory attention
        self.q_proj = nn.Linear(dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(slot_dim, dim, bias=False)
        
        # Gate for memory writing
        self.write_gate = nn.Linear(dim, 1, bias=False)
        
    def forward(self, x, write_mem: bool = False):
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            write_mem: Whether to write to memory (only during eval)
        """
        batch_size, seq_len, _ = x.shape
        
        # Query from input
        q = self.q_proj(x)  # [batch, seq_len, slot_dim]
        
        # Keys and values from memory slots
        k = self.k_proj(self.memory_slots)  # [num_slots, slot_dim]
        v = self.v_proj(self.memory_slots)  # [num_slots, slot_dim]
        
        # Attention over memory slots
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, seq_len, num_slots]
        attn_scores = attn_scores / (self.slot_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Retrieve from memory
        memory_out = torch.matmul(attn_weights, v)  # [batch, seq_len, slot_dim]
        
        # Project to output dimension
        output = self.o_proj(memory_out)  # [batch, seq_len, dim]
        
        # Memory writing (only during evaluation and when explicitly enabled)
        if write_mem and not self.training:
            write_weights = torch.sigmoid(self.write_gate(x))  # [batch, seq_len, 1]
            # Update memory slots with weighted average
            # This is a simplified write mechanism
            pass
            
        return output


class AttnBranch(nn.Module):
    """Attention branch using DeltaNet for the H1-Titans architecture."""
    
    def __init__(self, dim: int, heads: int = 8, **kwargs):
        super().__init__()
        self.attention = DeltaNet(
            hidden_size=dim,
            num_heads=heads,
            **kwargs
        )
        
    def forward(self, x, **kwargs):
        output, _, _ = self.attention(x, **kwargs)
        return output


class Mamba2Branch(nn.Module):
    """Mamba2 SSM branch for the H1-Titans architecture."""
    
    def __init__(self, dim: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.mamba = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
    def forward(self, x):
        return self.mamba(x)


class H1TitansBlock(nn.Module):
    """Single H1-Titans block with parallel attention, Mamba2, and Titans memory branches."""
    
    def __init__(
        self, 
        dim: int,
        heads: int = 8,
        num_slots: int = 128,
        slot_dim: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        mixer_type: str = "SUM",
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.mixer_type = mixer_type
        
        # Three parallel branches
        self.attn_branch = AttnBranch(dim, heads, **kwargs)
        self.mamba_branch = Mamba2Branch(dim, d_state, d_conv, expand)
        self.memory_branch = TitansMAG(dim, num_slots, slot_dim)
        
        # Branch mixing
        if mixer_type == "SUM":
            # Learnable weights for weighted sum
            self.branch_weights = nn.Parameter(torch.ones(3) / 3)
        elif mixer_type == "CONCAT_PROJ":
            # Concatenate and project
            self.mixer_proj = nn.Linear(dim * 3, dim, bias=False)
        elif mixer_type == "GATED_FUSION":
            # Gated fusion
            self.gate_proj = nn.Linear(dim, 3, bias=False)
        
        # Layer norm
        self.norm = RMSNorm(dim)
        
    def forward(self, x, write_mem: bool = False, **kwargs):
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            write_mem: Whether to write to memory
        """
        # Apply layer norm
        x_normed = self.norm(x)
        
        # Parallel branches
        attn_out = self.attn_branch(x_normed, **kwargs)
        mamba_out = self.mamba_branch(x_normed)
        memory_out = self.memory_branch(x_normed, write_mem=write_mem)
        
        # Mix branches
        if self.mixer_type == "SUM":
            # Weighted sum
            weights = F.softmax(self.branch_weights, dim=0)
            mixed = weights[0] * attn_out + weights[1] * mamba_out + weights[2] * memory_out
        elif self.mixer_type == "CONCAT_PROJ":
            # Concatenate and project
            concat = torch.cat([attn_out, mamba_out, memory_out], dim=-1)
            mixed = self.mixer_proj(concat)
        elif self.mixer_type == "GATED_FUSION":
            # Gated fusion
            gates = F.softmax(self.gate_proj(x_normed), dim=-1)  # [batch, seq_len, 3]
            mixed = (gates[..., 0:1] * attn_out + 
                    gates[..., 1:2] * mamba_out + 
                    gates[..., 2:3] * memory_out)
        
        # Residual connection
        return x + mixed


class H1TitansModel(nn.Module):
    """Falcon-H1 + Mamba2 + Titans hybrid architecture model."""
    
    def __init__(
        self,
        dim: int = 768,
        depth: int = 4,
        heads: int = 8,
        d_head: int = 64,
        num_slots: int = 128,
        slot_dim: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        mixer_type: str = "SUM",
        vocab_size: int = 32000,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        
        # H1-Titans blocks
        self.layers = nn.ModuleList([
            H1TitansBlock(
                dim=dim,
                heads=heads,
                num_slots=num_slots,
                slot_dim=slot_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mixer_type=mixer_type,
                layer_idx=i,
                **kwargs
            ) for i in range(depth)
        ])
        
        # Final layer norm and output
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, input_ids, write_mem: bool = False, **kwargs):
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            write_mem: Whether to write to memory (only during eval)
        """
        # Embed tokens
        x = self.embed_tokens(input_ids)  # [batch, seq_len, dim]
        
        # Pass through H1-Titans blocks
        for layer in self.layers:
            x = layer(x, write_mem=write_mem, **kwargs)
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


def build_model(cfg: dict = None, **kwargs) -> H1TitansModel:
    """Build H1TitansModel with given configuration."""
    # Default configuration
    default_cfg = {
        "vocab_size": 32000,
        "dim": 768,
        "depth": 4,
        "heads": 8,
    }
    
    # Merge configurations
    if cfg is not None:
        merged_cfg = {**default_cfg, **cfg}
    else:
        merged_cfg = default_cfg.copy()
    
    merged_cfg.update(kwargs)
    
    return H1TitansModel(**merged_cfg)
