from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm
from fla.ops import delta_rule_chunkwise
from fla.modules.l2norm import l2norm
from fla.models.utils import register_model

class Config:
    def __init__(self, d_model=768, num_heads=12, expand_k=2.0, expand_v=2.0, use_beta=True, use_gate=True, mem_slots=64, mem_dim=64):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_k = (d_model * expand_k) // num_heads
        self.head_v = (d_model * expand_v) // num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.qk_activation = 'l2'
        self.qk_norm = 'l2'
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.mem_slots = mem_slots
        self.mem_dim = mem_dim
        self.qk_activation = 'l2'
        self.qk_norm = 'l2'

class H1TitansModel(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.blocks = nn.ModuleList([H1TitansBlock(config) for _ in range(12)])
        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size, bias=False)
        
    def forward(self, input_ids, write_mem=False):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, write_mem)
        x = self.norm(x)
        return self.output(x)

class H1TitansBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = Attention(config)
        self.mem = Memory(config)
        self.mixer = nn.Linear(2 * config.d_model, config.d_model, bias=False)
        if config.use_gate:
            self.gate = nn.Linear(config.d_model, config.d_model)
        self.norm = RMSNorm(config.d_model)
        
    def forward(self, x, write_mem):
        a = self.attn(x)
        m = self.mem(x, write_mem)
        combined = torch.cat([a, m], dim=-1)
        combined = self.mixer(combined)
        if hasattr(self, 'gate'):
            combined = F.silu(self.gate(x)) * combined
        x = x + combined
        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.d_model, config.head_k * config.num_heads, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.head_k * config.num_heads, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.head_v * config.num_heads, bias=False)
        self.o_proj = nn.Linear(config.head_v * config.num_heads, config.d_model, bias=False)
        self.config = config
        if config.use_beta:
            self.b_proj = nn.Linear(config.d_model, config.num_heads, bias=False)
        else:
            self.b_proj = None
        
    def forward(self, x):
        B, T, C = x.size()
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.config.num_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.config.num_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.config.num_heads)
        
        q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
        if self.config.qk_norm == 'l2':
            q = l2norm(q).to(q.dtype)
            k = l2norm(k).to(k.dtype)
        
        beta = F.sigmoid(self.b_proj(x)) if self.b_proj else torch.ones(B, T, self.config.num_heads, device=x.device)
        beta = rearrange(beta, 'b t h -> b h t 1')
        
        q = rearrange(q, 'b h t d -> (b h) t d')
        k = rearrange(k, 'b h t d -> (b h) t d')
        v = rearrange(v, 'b h t d -> (b h) t d')
        
        o, _ = delta_rule_chunkwise(q, k, v, beta, chunk_size=32)
        o = rearrange(o, '(b h) t d -> b t h d', b=B, h=self.config.num_heads).contiguous()
        o = o.reshape(B, T, -1)
        o = self.o_proj(o)
        return o

class Memory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(config.mem_slots, config.mem_dim))
        self.q_proj = nn.Linear(config.d_model, config.mem_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.mem_dim, bias=False)
        self.out_proj = nn.Linear(config.mem_dim, config.d_model, bias=False)
        self.norm = RMSNorm(config.mem_dim)
        self.decay = 0.999
        
    def forward(self, x, write):
        B, T, C = x.size()
        q = self.norm(self.q_proj(x))  # (B, T, D)
        attn = F.softmax(q @ self.slots.T, dim=-1)  # (B*T, slots, )
        read = attn @ self.slots  # (B*T, D)
        read = read.reshape(B, T, -1)  # (B, T, D)
        out = self.out_proj(read) + x
        if write and not self.training:
            with torch.no_grad():
                v = self.v_proj(x).reshape(-1, self.slots.size(1))  # (B*T, D) -> (B*T, slots)
                q_update = self.q_proj(x).reshape(-1, self.slots.size(1))  # (B*T, D) -> (B*T, slots)
                attn_update = F.softmax(q_update @ self.slots.T, dim=-1)  # (B*T, slots, )
                # Update slots: EMA
                self.slots *= self.decay
                self.slots += (1 - self.decay) * (attn_update.T @ v).T
        return out

@register_model
def build_model(vocab_size, d_model=768, num_heads=12, **kwargs):
    config = Config(d_model, num_heads, **kwargs)
    return H1TitansModel(config, vocab_size)

# Example usage:
# model = build_model(vocab_size=50257, d_model=768, num_heads=12, use_beta=True, use_gate=True, mem_slots=64, mem_dim=64)
