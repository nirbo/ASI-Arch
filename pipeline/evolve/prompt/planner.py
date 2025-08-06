def Planner_input(context: str) -> str:
    return f"""# Neural Architecture Evolution Mission

## EXPERIMENTAL CONTEXT & HISTORICAL EVIDENCE
{context}

## ARCHITECTURE EVOLUTION OBJECTIVE
Your mission is to create a breakthrough neural architecture that addresses critical performance limitations identified through experimental evidence while integrating cutting-edge research insights. Design and implement an innovative architecture that maintains computational efficiency while achieving superior cognitive capabilities.

## SYSTEMATIC EVOLUTION METHODOLOGY

### PHASE 1: Evidence-Based Analysis Framework

#### 1.1 Architecture Forensics
**Current State Assessment:**
- Use `read_code_file` to examine existing architectural implementations
- Map computational mechanisms, design patterns, and information flow
- Identify core algorithmic approaches and their theoretical foundations
- Document interface constraints and compatibility requirements

#### 1.2 Performance Pattern Recognition  
**Historical Evidence Analysis:**
- **Training Dynamics Diagnosis**: Extract optimization challenges from loss curves and convergence patterns
- **Task-Specific Performance Profiling**: Identify capability gaps across cognitive domains (reasoning, memory, comprehension)
- **Bottleneck Identification**: Pinpoint architectural elements limiting performance vs. those enabling strengths
- **Cross-Architecture Comparison**: Analyze performance patterns across different experimental variants

#### 1.3 Research Integration Strategy
**Theoretical Foundation Building:**
- Map research insights to observed performance limitations
- Identify specific theoretical principles addressing architectural weaknesses  
- Synthesize multiple research findings for comprehensive enhancement opportunities
- Validate theoretical applicability through experimental evidence correlation

### PHASE 2: Innovation Design Framework

#### 2.1 Targeted Performance Engineering
**Gap-Specific Solutions:**
- Design architectural modifications targeting the most critical performance bottlenecks
- Create mechanisms leveraging research insights for problematic capability domains
- Balance multiple improvement objectives while maintaining architectural coherence
- Ensure modifications address root causes rather than symptoms

#### 2.2 Theoretical Grounding Protocol
**Research-Driven Design:**
- Ground all modifications in validated theoretical principles
- Ensure mathematical and computational justification for proposed changes
- Verify alignment with established research findings and best practices
- Create novel combinations of insights for breakthrough potential

#### 2.3 Efficiency Optimization Standards
**Computational Constraints:**
- Design using chunked computation patterns for scalability
- Maintain sub-quadratic O(N log N) complexity throughout
- Optimize memory usage through efficient processing strategies
- Preserve performance gains within strict complexity bounds

### PHASE 3: Implementation Excellence Protocol

#### 3.1 Architecture Implementation Standards
**Code Development Requirements:**
- Use `write_code_file` to implement the complete evolved architecture
- Preserve interface compatibility (forward function signatures, __init__ **kwargs)
- Add new parameters with sensible defaults (enabled by default for new features)
- Remove or refactor existing features to prevent architectural bloat
- Implement proper causal masking and information flow constraints

#### 3.2 Quality Assurance Framework
**Technical Excellence Standards:**
- Maintain @torch.compile decorators for computational optimization
- Preserve chunked processing patterns throughout the architecture
- Ensure causal constraints prevent any information leakage
- Verify sub-quadratic complexity in all implemented operations

#### 3.3 Documentation and Justification
**Innovation Communication:**
- Create comprehensive motivation explaining evolution rationale
- Connect experimental evidence to theoretical insights and implementation decisions
- Justify expected improvements based on research findings
- Provide clear reasoning for all architectural design choices

## TECHNICAL IMPLEMENTATION SPECIFICATIONS

### Critical Preservation Requirements
- **Class Structure**: Maintain DeltaNet class name and inheritance hierarchy
- **Interface Stability**: Preserve exact forward function signature compatibility
- **Parameter Compatibility**: Support **kwargs in __init__ for extensibility
- **Compilation Strategy**: Apply @torch.compile selectively to core computational functions only
- **Dimensional Consistency**: Maintain d_model and core parameter structure

### Implementation Quality Standards
- **Chunked Processing**: All sequence operations must utilize fixed-size chunking
- **Causal Integrity**: Implement strict causal constraints in attention-like mechanisms
- **Complexity Bounds**: Ensure O(N log N) or better for all operations
- **Memory Efficiency**: Design for optimal memory usage with chunked patterns
- **Compilation Safety**: Avoid @torch.compile on utility functions to prevent conflicts

### MANDATORY: Tensor Operations Robustness
- **einops.rearrange() Requirement**: Replace ALL .view()/.reshape() with einops.rearrange()
- **Dynamic Dimension Handling**: Never manually calculate dimensions - use einops inference
- **Batch Size Agnostic**: All operations must work with ANY batch size
- **Runtime Shape Extraction**: Get dimensions from tensor.shape at runtime, not config
- **Adaptive Processing**: Design for actual tensor dimensions, not predetermined values

### Cross-Environment Robustness Standards
- **Universal Compatibility**: Identical performance across training/evaluation/inference
- **Memory Adaptation**: Graceful handling of varying memory constraints
- **Shape Tolerance**: Robust operation with varying input dimensions
- **Resource Awareness**: Automatic adaptation to available computational resources

## INNOVATION TARGET DOMAINS

### Primary Capability Enhancement Areas
- **Extended Context Memory**: Revolutionary long-range dependency handling with linear O(n) complexity
- **Multi-Scale Information Integration**: Enhanced temporal and semantic scale processing
- **Hierarchical Reasoning Systems**: Brain-inspired multi-timescale processing (fast/slow systems)
- **Parallel Reasoning Architectures**: HRM-style latent reasoning without sequential token generation
- **Linear Attention Innovations**: Advanced O(n) attention mechanisms (RoPE, ALiBi, Flash-Linear)
- **Hybrid Processing Fusion**: Integration of sequence processing and reasoning modules
- **Adaptive Computational Mechanisms**: Dynamic adjustment based on input and reasoning complexity
- **Efficiency-Performance Optimization**: Superior capabilities within complexity constraints
- **Cognitive Task Performance**: Breakthrough improvements in reasoning, planning, and comprehension
- **Environmental Robustness**: Consistent performance across execution contexts
- **Resource Efficiency**: Optimal adaptation to computational constraints

## HYBRID ARCHITECTURE RESEARCH FOUNDATIONS

### Linear Attention + Hierarchical Reasoning Integration
**Revolutionary Architecture Paradigm:**
Advanced architectures should explore fusion of efficient sequence processing with hierarchical reasoning systems inspired by cognitive science and recent breakthroughs in parallel reasoning.

#### Core Research Insights:
1. **Linear Attention Efficiency**: O(n) complexity for sequence processing using feature maps (ELU+1, kernel methods)
2. **HRM Hierarchical Reasoning**: Multi-timescale processing with strategic (H-module) and tactical (L-module) systems
3. **Parallel Reasoning**: 100x faster than Chain-of-Thought through latent space reasoning vs. token generation
4. **Brain-Inspired Architecture**: Fast/slow system integration mimicking human cognitive processing

### Hybrid Architecture Templates

#### Template 1: Sequential Linear→HRM
```python
# Pattern: Linear attention processes sequence, HRM handles reasoning
class SequentialLinearHRM:
    def forward(self, x):
        # Phase 1: Efficient sequence processing (O(n))
        seq_repr = self.linear_attention(x)  
        
        # Phase 2: Hierarchical reasoning on representations
        h_state = self.hrm_h_module(seq_repr)  # Strategic planning
        output = self.hrm_l_module(h_state, seq_repr)  # Tactical execution
        return output
```

#### Template 2: Parallel Linear||HRM Processing
```python
# Pattern: Simultaneous sequence and reasoning processing
class ParallelLinearHRM:
    def forward(self, x):
        # Parallel processing streams
        seq_stream = self.linear_attention_branch(x)
        reasoning_stream = self.hrm_reasoning_branch(x)
        
        # Cross-modal fusion
        return self.fusion_layer(seq_stream, reasoning_stream)
```

#### Template 3: Nested Hierarchical Integration
```python
# Pattern: Multi-level integration with hierarchical attention
class NestedLinearHRM:
    def forward(self, x):
        # Level 1: Token-level linear attention
        token_attn = self.linear_attention(x)
        
        # Level 2: Sequence-level strategic reasoning (H-module)
        strategic_state = self.h_module(token_attn)
        
        # Level 3: Fine-grained tactical processing (L-module)  
        tactical_output = self.l_module(strategic_state, token_attn)
        
        return self.output_projection(tactical_output)
```

### Key Integration Mechanisms

#### Multi-Timescale Processing Architecture
- **Fast System (L-module)**: Rapid, intuitive computation updating every timestep
- **Slow System (H-module)**: Deliberate, strategic reasoning updating every T timesteps
- **Linear Backbone**: Efficient O(n) sequence processing for long contexts

#### Cross-Modal State Management
- **Attention-to-Reasoning**: Transfer linear attention outputs to reasoning modules
- **Reasoning-to-Attention**: Use reasoning states to guide attention patterns
- **Bidirectional Fusion**: Two-way information flow between processing systems

#### Adaptive Computational Time (ACT)
- **Dynamic Depth**: Variable reasoning cycles based on problem complexity
- **Convergence Detection**: Automatic stopping when reasoning reaches stable state
- **Resource Allocation**: Optimal computation distribution between modules

### Hybrid Innovation Opportunities

#### Novel Architectural Components
1. **LinearHRM Fusion Layers**: Cross-modal attention between sequence and reasoning representations
2. **Hierarchical Attention Heads**: Multi-scale attention operating at token, phrase, and sequence levels
3. **Reasoning-Guided Linear Attention**: HRM states directing linear attention patterns
4. **Parallel Reasoning Chains**: Multiple HRM instances processing different reasoning aspects
5. **Adaptive Module Switching**: Dynamic routing between linear processing and reasoning modes

#### Performance Optimization Targets
- **Sequence Scalability**: Linear O(n) complexity for arbitrarily long sequences
- **Reasoning Efficiency**: 10-100x speedup over sequential reasoning approaches
- **Parameter Efficiency**: Compact models (50M-200M parameters) vs. billion-parameter alternatives
- **Parallel Processing**: Inherent parallelization of reasoning vs. sequential token generation
- **Data Efficiency**: Ultra-low data requirements for reasoning capabilities

#### Research-Backed Design Principles
1. **Cognitive Architecture**: Dual-process theory implementation (System 1/System 2)
2. **Hierarchical Abstraction**: Multi-level processing from concrete to abstract reasoning
3. **Latent Reasoning**: Processing in continuous representations rather than discrete tokens
4. **Dynamic Resource Allocation**: Adaptive computation based on task requirements
5. **Hybrid Efficiency**: Best-of-both-worlds combining sequence processing and reasoning

### Implementation Guidelines for Hybrid Architectures

#### Core Hybrid Components to Consider
```python
# Essential components for Linear-HRM fusion
class HybridComponents:
    linear_attention: LinearAttentionModule     # O(n) sequence processing
    h_module: HighLevelReasoningModule         # Strategic, slow reasoning
    l_module: LowLevelReasoningModule          # Tactical, fast reasoning
    fusion_layer: CrossModalFusion             # Integration mechanism
    adaptive_controller: ComputationController # Dynamic resource allocation
```

#### Multi-Scale Integration Patterns
- **Token Level**: Linear attention for efficient token interactions
- **Sequence Level**: HRM H-module for high-level sequence understanding
- **Reasoning Level**: HRM L-module for detailed computational processing
- **Output Level**: Fusion and projection of multi-scale representations

## DELIVERABLE SPECIFICATIONS

### PRIMARY DELIVERABLE: Complete Implementation
**Architecture Code (MANDATORY):**
- **Implementation Tool**: Use `write_code_file` to create complete working architecture
- **Innovation Quality**: Embed revolutionary architectural advances in functional code
- **Constraint Compliance**: Preserve class structure, parameters, and interface compatibility
- **Technical Standards**: Maintain sub-quadratic complexity, chunked processing, causal constraints
- **Robustness Implementation**: Use einops.rearrange() universally, ensure batch size independence

### SECONDARY DELIVERABLE: Design Documentation
**Architecture Description:**
- **Naming Convention**: `delta_net_[innovation_identifier]` reflecting core innovations
- **Motivation Document**: Comprehensive explanation including:
  - Key architectural innovations and their implementation
  - Research insights applied and expected performance improvements
  - Design choice justification based on experimental evidence
  - Connection between theory, evidence, and implementation

## SUCCESS CRITERIA FRAMEWORK

### Critical Success Factors (Ranked by Priority)
1. **Implementation Excellence**: Successfully create breakthrough architecture using write_code_file
2. **Constraint Adherence**: Maintain class name, parameters, and interface compatibility
3. **Technical Robustness**: Ensure complexity bounds, chunked processing, causal constraints
4. **Universal Compatibility**: Use einops.rearrange() universally, support any batch size
5. **Evidence-Based Innovation**: Embed research insights addressing identified limitations
6. **Performance Targeting**: Implement solutions for specific weakness areas identified

## MISSION EMPHASIS
Your **PRIMARY OBJECTIVE** is implementing breakthrough architectural code that demonstrates robust performance across all execution environments and batch configurations. Create working innovations that directly address identified performance gaps through research-guided architectural evolution. Documentation serves as secondary validation of implemented innovations.

Begin your evolution process by examining the experimental evidence and identifying the most critical architectural improvement opportunities."""

