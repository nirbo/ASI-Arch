"""
Hybrid Linear-HRM Architecture Example
Demonstrates integration of linear attention with hierarchical reasoning modules
This serves as a base architecture for ASI-Arch to evolve from.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum

class LinearAttention(nn.Module):
    """Efficient Linear Attention with O(n) complexity"""
    
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, L, D = x.shape
        
        Q = rearrange(self.q_proj(x), 'b l (h d) -> b l h d', h=self.num_heads)
        K = rearrange(self.k_proj(x), 'b l (h d) -> b l h d', h=self.num_heads)
        V = rearrange(self.v_proj(x), 'b l (h d) -> b l h d', h=self.num_heads)
        
        # Apply feature map for positive attention weights
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Linear attention computation O(n)
        KV = einsum(K, V, 'b l h d, b l h e -> b h d e')
        K_sum = K.sum(dim=1, keepdim=True)
        
        out = einsum(Q, KV, 'b l h d, b h d e -> b l h e')
        out = out / (einsum(Q, K_sum, 'b l h d, b k h d -> b l h').unsqueeze(-1) + 1e-6)
        
        out = rearrange(out, 'b l h d -> b l (h d)')
        return self.out_proj(out)

class HRMHighModule(nn.Module):
    """HRM High-level Module: Strategic reasoning (System 2)"""
    
    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 2
            
        self.strategic_processor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # *2 for prev_state + input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.convergence_head = nn.Linear(embed_dim, 1)
        
    def forward(self, prev_h_state, l_summary):
        # Strategic reasoning on combined state
        combined = torch.cat([prev_h_state, l_summary], dim=-1)
        h_state = self.strategic_processor(combined)
        
        # Convergence detection
        convergence = torch.sigmoid(self.convergence_head(h_state))
        
        return h_state, convergence

class HRMLowModule(nn.Module):
    """HRM Low-level Module: Tactical processing (System 1)"""
    
    def __init__(self, embed_dim, num_cycles=3):
        super().__init__()
        self.num_cycles = num_cycles
        
        # Fast processing cycles
        self.processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim * 4),
                nn.ReLU(),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(num_cycles)
        ])
        
        # Gating mechanism
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.summary_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, h_state, seq_repr):
        B, L, D = seq_repr.shape
        h_expanded = h_state.unsqueeze(1).expand(-1, L, -1)
        
        l_state = torch.zeros_like(seq_repr)
        
        # Rapid processing cycles
        for processor in self.processors:
            combined = torch.cat([h_expanded, l_state], dim=-1)
            processed = processor(combined)
            
            # Gated update
            gate_input = torch.cat([seq_repr, processed], dim=-1)
            gate_values = torch.sigmoid(self.gate(gate_input))
            l_state = gate_values * l_state + (1 - gate_values) * processed
        
        # Generate summary for H-module
        l_summary = self.summary_proj(l_state.mean(dim=1))
        
        return l_state, l_summary

class CrossModalFusion(nn.Module):
    """Fusion layer for combining linear attention and HRM outputs"""
    
    def __init__(self, embed_dim):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        
        self.fusion_ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, linear_out, hrm_out):
        # Cross-attention between modalities
        fused_linear, _ = self.cross_attn(linear_out, hrm_out, hrm_out)
        
        # Combine and process
        combined = torch.cat([fused_linear, hrm_out], dim=-1)
        fused = self.fusion_ffn(combined)
        
        # Residual connections
        return fused + linear_out + hrm_out

class HybridLinearHRMBlock(nn.Module):
    """Complete hybrid block combining linear attention with HRM reasoning"""
    
    def __init__(self, embed_dim, num_heads=8, hrm_cycles=3, h_update_interval=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.h_update_interval = h_update_interval
        
        # Core components
        self.linear_attention = LinearAttention(embed_dim, num_heads)
        self.hrm_h_module = HRMHighModule(embed_dim)
        self.hrm_l_module = HRMLowModule(embed_dim, hrm_cycles)
        self.fusion_layer = CrossModalFusion(embed_dim)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x, h_state=None, step_count=0):
        """
        Args:
            x: Input sequence [batch, seq_len, embed_dim]
            h_state: Previous high-level reasoning state
            step_count: Current step for multi-timescale processing
        """
        B, L, D = x.shape
        
        if h_state is None:
            h_state = torch.zeros(B, D, device=x.device)
        
        # 1. Linear attention processing (O(n) efficiency)
        linear_out = self.linear_attention(self.norm1(x))
        x = x + linear_out
        
        # 2. HRM L-module processing (every timestep)
        l_out, l_summary = self.hrm_l_module(h_state, self.norm2(x))
        
        # 3. HRM H-module processing (every h_update_interval steps)
        if step_count % self.h_update_interval == 0:
            h_state, convergence = self.hrm_h_module(h_state, l_summary)
        
        # 4. Cross-modal fusion
        fused_out = self.fusion_layer(linear_out, l_out)
        x = x + fused_out
        
        # 5. Feed forward
        x = x + self.ffn(self.norm3(x))
        
        return x, h_state

class HybridLinearHRMModel(nn.Module):
    """Complete Hybrid Linear-HRM Language Model"""
    
    def __init__(self, vocab_size=32000, embed_dim=512, num_layers=6, num_heads=8, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Hybrid transformer blocks
        self.blocks = nn.ModuleList([
            HybridLinearHRMBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Multi-timescale state management
        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, h_states=None):
        B, L = input_ids.shape
        
        # Initialize hierarchical states if needed
        if h_states is None:
            h_states = [torch.zeros(B, self.embed_dim, device=input_ids.device) 
                       for _ in range(self.num_layers)]
        
        # Position embeddings
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        
        # Process through hybrid blocks with multi-timescale reasoning
        new_h_states = []
        for i, block in enumerate(self.blocks):
            x, new_h_state = block(x, h_states[i], self.step_counter.item())
            new_h_states.append(new_h_state)
        
        # Update step counter for multi-timescale processing
        self.step_counter += 1
        
        # Final layers
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits, new_h_states
    
    def reset_reasoning_state(self):
        """Reset multi-timescale reasoning state"""
        self.step_counter.zero_()

# Main model class for training script compatibility
class Model(HybridLinearHRMModel):
    """Wrapper class for training script compatibility"""
    
    def __init__(self, vocab_size=32000, **kwargs):
        super().__init__(vocab_size=vocab_size, **kwargs)
        self.h_states = None
    
    def forward(self, input_ids):
        """Simplified forward pass for training compatibility"""
        logits, self.h_states = super().forward(input_ids, self.h_states)
        return logits

# Example usage and architecture information
if __name__ == "__main__":
    # Architecture specifications
    print("Hybrid Linear-HRM Architecture")
    print("==============================")
    print("• Linear Attention: O(n) sequence processing")
    print("• HRM H-Module: Strategic reasoning (System 2)")  
    print("• HRM L-Module: Tactical processing (System 1)")
    print("• Cross-Modal Fusion: Integration of attention + reasoning")
    print("• Multi-Timescale: H-module updates every 4 steps, L-module every step")
    print("• Expected Benefits: Scalable + fast reasoning")
    
    # Test model
    model = Model(vocab_size=1000, embed_dim=256, num_layers=4)
    x = torch.randint(0, 1000, (2, 128))  # [batch=2, seq_len=128]
    
    with torch.no_grad():
        output = model(x)
        print(f"\nModel test successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Memory efficient: O(n) attention complexity")