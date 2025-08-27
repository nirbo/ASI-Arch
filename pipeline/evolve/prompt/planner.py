def Planner_input(context: str) -> str:
    return f"""# Falcon-H1 + Titans Memory Fusion Research Mission

## EXPERIMENTAL CONTEXT & ARCHITECTURAL FOUNDATION
{context}

## CURRENT BASE ARCHITECTURE: Falcon-H1 + Mamba2 + Titans-MAG
You are working with a **hybrid parallel-branch architecture** that fuses:
- **Falcon-H1 style**: Parallel attention + SSM branches with learnable branch weights
- **Mamba2 SSM**: State-space model for linear-time sequential processing 
- **Titans-MAG Memory**: Persistent key-value slots with conservative EMA updates

**Core Architecture Components:**
- `H1TitansBlock`: Three parallel branches (attention, mamba2, titans memory)
- `AttnBranch`: Standard attention with RoPE positioning, causal masking
- `Mamba2Branch`: SSM with fallback to SimpleSSM if mamba_ssm unavailable  
- `TitansMAG`: Persistent memory with eval-only writes, batch-agnostic reads
- **Mixer**: Parallel sum with learnable branch weights: `w[0]*attention + w[1]*ssm + w[2]*memory`

## RESEARCH OBJECTIVE: TITANS INTEGRATION OPTIMIZATION

Your mission is to **evolve the Titans memory integration** within the Falcon-H1 hybrid framework while preserving:
- **Sub-quadratic complexity** (linear attention, linear-time SSM)
- **Causal correctness** with no information leakage
- **Batch-size independence** across all operations
- **Training stability** with proper write policies

### Primary Evolution Targets:
1. **Memory Variants**: MAG (default) → MAC (memory-as-context) → MAL (memory-as-layer)
2. **Mixer Strategies**: SUM (default) → CONCAT+PROJ → GATED_FUSION → WEIGHTED_ROUTING
3. **Mamba2 Tuning**: Adjust d_state, d_conv, expand parameters for optimal SSM performance
4. **Write Policies**: Eval-only (default) → train+eval → learnable gating → adaptive frequency

## EVOLUTION METHODOLOGY

### PHASE 1: Architecture Analysis & Bottleneck Identification
**Current State Assessment:**
- Use `read_code_file` to examine `pipeline/current_architecture.py`
- Map information flow: Attention → SSM → Memory → Branch Mixing
- Identify performance limitations from experimental evidence
- Analyze branch weight dynamics and utilization patterns

**Critical Constraint Verification:**
- **Forward Signature**: `forward(self, input_ids, write_mem=False)` must remain unchanged
- **Causal Masking**: All branches must respect causal constraints
- **Complexity Bounds**: Maintain O(N) or O(N log N) throughout
- **Memory Writes**: Must be disabled during training (`write_mem=False` in training)

### PHASE 2: Targeted Innovation Design

**Titans Memory Evolution Priorities:**
1. **MAG → MAC Transition**: Convert memory-as-gates to memory-as-context
   - Replace gated memory reads with contextual memory injection
   - Maintain persistent slot updates with EMA decay
   - Preserve batch-agnostic operation
   
2. **MAC → MAL Exploration**: Develop memory-as-layer variants
   - Integrate memory as dedicated processing layer
   - Design memory-aware routing between attention/SSM branches
   - Maintain sub-quadratic complexity constraints

3. **Mixer Innovation**: Beyond parallel sum
   - **CONCAT+PROJ**: Concatenate branch outputs, project to d_model
   - **GATED_FUSION**: Learnable gates for branch combination  
   - **WEIGHTED_ROUTING**: Dynamic routing based on input characteristics

**Mamba2 SSM Optimization:**
- **d_state**: Experiment with 32, 64, 128 for different memory capacities
- **d_conv**: Try 3, 4, 7 for different local context windows
- **expand**: Test 1.5, 2, 4 for different inner dimension scaling

### PHASE 3: Implementation Excellence

**Code Implementation Standards:**
- **Class Structure**: Maintain `H1TitansModel` and `H1TitansBlock` names
- **Interface Preservation**: Keep exact `forward(input_ids, write_mem=False)` signature
- **Configuration**: Extend `H1TitansCfg` with new parameters having sensible defaults
- **Memory Safety**: Ensure all new operations are batch-size independent

**Titans-Specific Constraints:**
- **Write Policy**: Memory writes ONLY during eval (`write_mem=True` and `not self.training`)
- **Causal Safety**: Memory reads must not leak future information
- **EMA Updates**: Preserve conservative update rates (decay=0.999 default)
- **Slot Management**: Maintain fixed slot count with similarity-based addressing

**Critical Preservation Requirements:**
- Sub-quadratic complexity in ALL operations
- Causal masking integrity across all branches  
- Batch-size agnostic tensor operations
- Forward pass signature compatibility
- RoPE positioning in attention branch
- Mamba2 with SimpleSSM fallback support

## INNOVATION FOCUS AREAS

### Memory Integration Variants:
**MAG (Memory-As-Gates)** - Current Implementation:
```python
# Current: Gated memory output added to branch mix
m = self.mem(h, enable_write=write_mem)
y = w[0]*a + w[1]*s + w[2]*m  # Parallel sum
```

**MAC (Memory-As-Context)** - Evolution Target:
```python  
# Evolution: Memory provides context for other branches
mem_context = self.mem.get_context(h)
a = self.attn(h, context=mem_context)
s = self.ssm(h, context=mem_context)
```

**MAL (Memory-As-Layer)** - Advanced Target:
```python
# Evolution: Memory as processing layer
h = self.mem.process_layer(h, enable_write=write_mem)
# Then attention/SSM process memory-enhanced representations
```

### Mixer Evolution Targets:
- **Learnable Branch Routing**: Dynamic weights based on input characteristics
- **Hierarchical Fusion**: Multi-stage branch combination strategies  
- **Attention-Guided Mixing**: Use attention patterns to guide branch weighting

## IMPLEMENTATION DELIVERABLES

### PRIMARY: Complete Working Code
**Using `write_code_file` create:**
- Complete `current_architecture.py` with evolved architecture
- Maintain all existing entrypoints: `build_model()`, `architecture_spec()`, `get_seed_candidate()`
- Preserve sanity check functionality with proper shape verification

### SECONDARY: Evolution Documentation  
**JSON Output Format:**
```json
{{
  "name": "H1-Titans-[VARIANT]",
  "motivation": "Clear explanation of specific evolution and expected benefits focusing on Titans integration improvements within Falcon-H1 framework"
}}
```

## SUCCESS CRITERIA
1. **Implementation Excellence**: Working code with evolved Titans integration
2. **Constraint Adherence**: All Falcon-H1 + Titans constraints preserved
3. **Innovation Depth**: Meaningful evolution beyond current MAG implementation
4. **Performance Targeting**: Changes address identified architectural bottlenecks
5. **Stability Maintenance**: Proper memory write policies and causal constraints

**CRITICAL**: Focus on **Titans memory system evolution** within the hybrid Falcon-H1 framework. This is NOT about creating entirely new architectures but about **optimizing the Titans integration** with attention and Mamba2 branches.

Begin by analyzing the experimental evidence to identify the most promising Titans evolution direction."""

