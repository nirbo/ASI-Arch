"""
HRM (Hierarchical Reasoning Module) Components for Linear↔HRM Hybrid Architectures
These components can be integrated with linear attention mechanisms for breakthrough architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum

class HighLevelReasoningModule(nn.Module):
    """
    HRM H-Module: Strategic, slow reasoning system (System 2)
    Updates every T timesteps for high-level planning and abstraction
    """
    
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 2
            
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Strategic reasoning layers
        self.input_transform = nn.Linear(embed_dim, hidden_dim)
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(2)  # Deeper reasoning
        ])
        
        # State management
        self.state_projection = nn.Linear(hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Convergence detection
        self.convergence_head = nn.Linear(embed_dim, 1)
        
    def forward(self, prev_h_state, l_state_summary):
        """
        Args:
            prev_h_state: Previous high-level state [batch, embed_dim]
            l_state_summary: Summary from L-module [batch, embed_dim]
        """
        # Combine previous strategic state with current tactical summary
        combined_input = prev_h_state + l_state_summary
        
        # Strategic reasoning
        h_repr = self.input_transform(combined_input).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        for layer in self.reasoning_layers:
            h_repr = layer(h_repr)
        
        # Project back to embed_dim
        h_state = self.layer_norm(self.state_projection(h_repr.squeeze(1)))
        
        # Convergence signal
        convergence = torch.sigmoid(self.convergence_head(h_state))
        
        return h_state, convergence

class LowLevelReasoningModule(nn.Module):
    """
    HRM L-Module: Tactical, fast reasoning system (System 1)  
    Updates every timestep for rapid, detailed computation
    """
    
    def __init__(self, embed_dim, num_cycles=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cycles = num_cycles
        
        # Rapid processing layers
        self.cycle_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim * 4),  # *2 for h_state + input
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(num_cycles)
        ])
        
        # State update mechanism
        self.state_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.update_gate = nn.Linear(embed_dim * 2, embed_dim)
        
        # Summary generation for H-module
        self.summary_layer = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, h_state, input_repr, max_cycles=None):
        """
        Args:
            h_state: Current high-level state [batch, embed_dim]
            input_repr: Input representation from linear attention [batch, seq_len, embed_dim]
        """
        if max_cycles is None:
            max_cycles = self.num_cycles
        
        batch_size, seq_len, embed_dim = input_repr.shape
        
        # Initialize L-state
        l_state = torch.zeros_like(input_repr)  # [batch, seq_len, embed_dim]
        h_state_expanded = h_state.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Rapid processing cycles
        for cycle in range(min(max_cycles, len(self.cycle_processor))):
            # Combine h_state with current l_state
            combined = torch.cat([h_state_expanded, l_state], dim=-1)  # [batch, seq_len, 2*embed_dim]
            
            # Process through cycle
            cycle_output = self.cycle_processor[cycle](combined)
            
            # Gated update
            combined_for_gates = torch.cat([input_repr, cycle_output], dim=-1)
            state_gate = torch.sigmoid(self.state_gate(combined_for_gates))
            update_gate = torch.sigmoid(self.update_gate(combined_for_gates))
            
            # Update L-state
            l_state = state_gate * l_state + update_gate * cycle_output
        
        # Generate summary for H-module
        l_summary = self.summary_layer(l_state.mean(dim=1))  # [batch, embed_dim]
        
        return l_state, l_summary

class LinearHRMFusion(nn.Module):
    """
    Cross-modal fusion layer combining linear attention and HRM reasoning
    """
    
    def __init__(self, embed_dim, fusion_dim=None):
        super().__init__()
        if fusion_dim is None:
            fusion_dim = embed_dim
            
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim
        
        # Cross-attention between modalities
        self.seq_to_reasoning = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        self.reasoning_to_seq = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        
        # Fusion layers
        self.fusion_transform = nn.Sequential(
            nn.Linear(embed_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, seq_repr, reasoning_repr):
        """
        Args:
            seq_repr: Linear attention output [batch, seq_len, embed_dim]
            reasoning_repr: HRM L-module output [batch, seq_len, embed_dim]
        """
        # Cross-modal attention
        seq_attended, _ = self.seq_to_reasoning(
            seq_repr, reasoning_repr, reasoning_repr
        )
        reasoning_attended, _ = self.reasoning_to_seq(
            reasoning_repr, seq_repr, seq_repr
        )
        
        # Fusion
        fused_input = torch.cat([seq_attended, reasoning_attended], dim=-1)
        fused_output = self.fusion_transform(fused_input)
        
        # Residual connection and projection
        output = self.output_projection(fused_output + seq_repr + reasoning_repr)
        
        return output

class AdaptiveComputationController(nn.Module):
    """
    Dynamic resource allocation controller for hybrid architectures
    """
    
    def __init__(self, embed_dim, max_cycles=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_cycles = max_cycles
        
        # Complexity estimation
        self.complexity_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Resource allocation
        self.resource_allocator = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim // 2),  # +1 for complexity
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 3),  # [h_cycles, l_cycles, fusion_weight]
            nn.Softmax(dim=-1)
        )
        
    def forward(self, input_repr):
        """
        Args:
            input_repr: Input representation [batch, seq_len, embed_dim]
        Returns:
            h_cycles, l_cycles, fusion_weight
        """
        # Estimate complexity
        complexity = self.complexity_estimator(input_repr.mean(dim=1))  # [batch, 1]
        
        # Allocate resources
        allocation_input = torch.cat([
            input_repr.mean(dim=1), complexity
        ], dim=-1)
        
        allocation = self.resource_allocator(allocation_input)  # [batch, 3]
        
        # Convert to cycles and weights
        h_cycles = torch.clamp(allocation[:, 0] * self.max_cycles, min=1, max=self.max_cycles)
        l_cycles = torch.clamp(allocation[:, 1] * self.max_cycles, min=1, max=self.max_cycles) 
        fusion_weight = allocation[:, 2]
        
        return h_cycles.int(), l_cycles.int(), fusion_weight

class HierarchicalLinearAttention(nn.Module):
    """
    Multi-scale linear attention with hierarchical processing
    """
    
    def __init__(self, embed_dim, num_heads=8, num_scales=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.head_dim = embed_dim // num_heads
        
        # Multi-scale projections
        self.scale_projections = nn.ModuleList([
            nn.ModuleDict({
                'q': nn.Linear(embed_dim, embed_dim),
                'k': nn.Linear(embed_dim, embed_dim), 
                'v': nn.Linear(embed_dim, embed_dim),
            })
            for _ in range(num_scales)
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(embed_dim * num_scales, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
    def linear_attention(self, Q, K, V):
        """Linear attention computation"""
        # Apply feature map (ELU + 1 for positivity)
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Linear attention: O(n) complexity
        KV = einsum(K, V, 'b s h d, b s h e -> b h d e')
        K_sum = K.sum(dim=1, keepdim=True)
        
        out = einsum(Q, KV, 'b s h d, b h d e -> b s h e')
        out = out / (einsum(Q, K_sum, 'b s h d, b t h d -> b s h').unsqueeze(-1) + 1e-6)
        
        return out
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        scale_outputs = []
        
        for scale_idx in range(self.num_scales):
            # Different pooling for different scales
            scale_factor = 2 ** scale_idx
            if scale_factor > 1:
                # Downsample for higher scales
                pooled_len = seq_len // scale_factor
                x_scale = F.avg_pool1d(
                    x.transpose(1, 2), kernel_size=scale_factor, stride=scale_factor
                ).transpose(1, 2)
            else:
                x_scale = x
                pooled_len = seq_len
            
            # Project to Q, K, V
            projections = self.scale_projections[scale_idx]
            Q = rearrange(projections['q'](x_scale), 'b s (h d) -> b s h d', h=self.num_heads)
            K = rearrange(projections['k'](x_scale), 'b s (h d) -> b s h d', h=self.num_heads)
            V = rearrange(projections['v'](x_scale), 'b s (h d) -> b s h d', h=self.num_heads)
            
            # Linear attention
            attn_out = self.linear_attention(Q, K, V)
            attn_out = rearrange(attn_out, 'b s h d -> b s (h d)')
            
            # Upsample if needed
            if scale_factor > 1:
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2), size=seq_len, mode='linear', align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(attn_out)
        
        # Fuse scales
        fused = self.scale_fusion(torch.cat(scale_outputs, dim=-1))
        return self.output_projection(fused)

# Utility functions for hybrid architectures

def create_hybrid_architecture_components(embed_dim, **kwargs):
    """Factory function to create standard hybrid components"""
    return {
        'linear_attention': HierarchicalLinearAttention(embed_dim, **kwargs),
        'h_module': HighLevelReasoningModule(embed_dim, **kwargs),
        'l_module': LowLevelReasoningModule(embed_dim, **kwargs),
        'fusion_layer': LinearHRMFusion(embed_dim, **kwargs),
        'adaptive_controller': AdaptiveComputationController(embed_dim, **kwargs)
    }

def multi_timescale_update_pattern(h_module, l_module, input_seq, h_update_interval=4):
    """
    Standard multi-timescale update pattern for hybrid architectures
    """
    h_state = torch.zeros(input_seq.shape[0], input_seq.shape[-1], device=input_seq.device)
    outputs = []
    
    for t in range(input_seq.shape[1]):
        current_input = input_seq[:, t:t+1, :]
        
        # L-module updates every timestep
        l_output, l_summary = l_module(h_state, current_input)
        
        # H-module updates every h_update_interval timesteps
        if t % h_update_interval == 0:
            h_state, convergence = h_module(h_state, l_summary)
        
        outputs.append(l_output)
    
    return torch.cat(outputs, dim=1)