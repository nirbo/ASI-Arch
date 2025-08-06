"""
Sample Linear Attention Architecture for ASI-Arch Testing
This is a basic linear attention model that serves as a starting template.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearAttention(nn.Module):
    """Simple Linear Attention Mechanism"""
    
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply feature map (ELU + 1 for positivity)
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Linear attention computation: O(n) complexity
        # KV = K^T @ V, then Q @ KV
        KV = torch.einsum('bshd,bshe->bhde', K, V)  # [batch, heads, head_dim, head_dim]
        K_sum = K.sum(dim=1, keepdim=True)  # Normalizing factor
        
        out = torch.einsum('bshd,bhde->bshe', Q, KV)  # [batch, seq_len, heads, head_dim]
        out = out / (torch.einsum('bshd,bthd->bsh', Q, K_sum).unsqueeze(-1) + 1e-6)
        
        # Reshape and project
        out = out.contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)

class FeedForward(nn.Module):
    """Feed Forward Network"""
    
    def __init__(self, embed_dim, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * embed_dim
            
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer Block with Linear Attention"""
    
    def __init__(self, embed_dim, num_heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        self.attention = LinearAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm architecture
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

class LinearAttentionModel(nn.Module):
    """Complete Linear Attention Language Model"""
    
    def __init__(self, vocab_size=32000, embed_dim=512, num_layers=6, num_heads=8, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits

# Main model class that the training script will look for
Model = LinearAttentionModel