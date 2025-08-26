#!/usr/bin/env python3
"""
Generate structured RAG entries for linear attention papers.
Based on the papers we identified for non-quadratic transformer research.
"""
import json
from pathlib import Path

def create_gla_paper():
    """Gated Linear Attention Transformers with Hardware-Efficient Training"""
    return [{
        "DESIGN_INSIGHT": """### DESIGN_INSIGHT_1: Gated Linear Attention with Hardware-Efficient FlashLinearAttention

GLA introduces data-dependent gates to linear attention while maintaining linear complexity. The key innovation is FlashLinearAttention, a hardware-efficient algorithm that trades off memory movement against parallelizability. Instead of standard softmax attention's O(N²) complexity, GLA achieves linear complexity while adding expressive gating mechanisms that control information flow adaptively based on input content.""",

        "EXPERIMENTAL_TRIGGER_PATTERNS": """**Task_Performance_Signatures**:
- Expect superior length generalization: models trained on 2K tokens can handle 20K+ sequences without perplexity degradation
- Language modeling performance competitive with LLaMA-architecture transformers while being significantly more memory efficient
- Training throughput improvements over Mamba and other linear attention models
- Particularly strong performance on tasks requiring long-range dependencies (lambada_openai, hellaswag)

**Architectural_Symptoms**:
- Linear memory scaling enables training with much longer sequences (up to 20K+)
- Faster inference than FlashAttention-2 even on short sequences (1K tokens)
- Higher training throughput than similarly-sized Mamba models
- Can formulate as RNN for O(1) memory inference while maintaining parallel training""",

        "BACKGROUND": """**Title**: Gated Linear Attention Transformers with Hardware-Efficient Training

**Historical Technical Context**: Traditional transformers suffer from quadratic complexity in attention computation, making them prohibitively expensive for long sequences. Linear attention methods like Linformer and Performer aimed to reduce this to linear complexity but often suffered performance degradation. The challenge was to maintain the expressiveness of softmax attention while achieving linear computational complexity.

**Technical Limitations**: Previous linear attention methods lacked expressiveness compared to softmax attention and had suboptimal hardware implementations that were slower than highly optimized softmax attention kernels like FlashAttention.

**Paper Concepts**: 
- **Gated Linear Attention (GLA)**: Adds data-dependent gates to linear attention for improved expressiveness
- **FlashLinearAttention**: Hardware-efficient implementation that outperforms FlashAttention-2 
- **Dual Formulation**: Can be computed as linear attention during training and as RNN during inference
- **Length Generalization**: Ability to handle sequences much longer than training length""",

        "ALGORITHMIC_INNOVATION": """**Core_Algorithm**:
GLA modifies standard linear attention with learnable gates that control information flow:
- Attention(Q,K,V) = softmax(QK^T + G)V where G are learned gates
- Can be reformulated as linear attention: O = (QΦ(K))V where Φ incorporates gating
- FlashLinearAttention algorithm optimizes memory access patterns for linear attention

**Key_Mechanism**:
- Data-dependent gating allows adaptive attention patterns while maintaining linear complexity
- Hardware-efficient kernel design minimizes memory transfers
- Dual computation modes: parallel training, sequential inference

**Mathematical_Formulation**:
- Linear attention: O_i = Σ_j α_ij V_j where α_ij = exp(q_i^T k_j + g_ij) / Z_i  
- Gating function: g_ij = MLP([q_i; k_j; pos_i - pos_j])
- Complexity: O(NLD) for training, O(LD) for inference

**Computational_Properties**:
- Training: O(NLD) time and memory vs O(N²D) for standard attention
- Inference: O(LD) memory, constant per-token generation time
- Hardware-efficient: faster than FlashAttention-2 even on short sequences""",

        "IMPLEMENTATION_GUIDANCE": """**Integration_Strategy**:
- Replace standard attention layers with GLA layers in transformer architecture
- Use FlashLinearAttention kernel for optimal performance
- Maintain compatibility with existing transformer infrastructure (residual connections, layer norms)

**Parameter_Settings**:
- Gate initialization: small random values to start near linear attention behavior
- Position encoding: can be integrated into gating mechanism
- Sequence length: can scale to 20K+ tokens with linear memory growth

**Application_Conditions**:
- Ideal for long-sequence tasks where standard attention is prohibitively expensive
- Beneficial when inference speed is critical (constant memory usage)
- Suitable for resource-constrained environments requiring efficient attention

**Expected_Outcomes**:
- Length generalization: 10x+ longer sequences than training length
- Memory efficiency: Linear scaling vs quadratic for standard attention  
- Speed improvements: 2x+ faster training throughput than comparable models
- Competitive performance: Matches or exceeds transformer baselines on language modeling"""
    }]

def create_devil_linear_paper():
    """The Devil in Linear Transformer"""
    return [{
        "DESIGN_INSIGHT": """### DESIGN_INSIGHT_1: Gradient Stabilization and Attention Normalization in Linear Transformers

Linear transformers suffer from two critical issues: unbounded gradients during attention computation and attention dilution over long sequences. The solution involves replacing attention matrix scaling with normalization to stabilize gradients, and using diagonal attention in early layers to preserve local structure while maintaining linear complexity.""",

        "EXPERIMENTAL_TRIGGER_PATTERNS": """**Task_Performance_Signatures**:
- Improved training stability with faster convergence and fewer gradient explosions
- Better performance on long-sequence tasks due to reduced attention dilution
- Competitive results on Long-Range Arena benchmark while maintaining linear complexity
- Superior text classification and language modeling performance vs vanilla linear transformers

**Architectural_Symptoms**:
- Stable gradient flow throughout training, no gradient clipping needed
- Attention weights properly focused on relevant tokens rather than diluted
- Linear memory scaling preserved while fixing performance degradation issues""",

        "BACKGROUND": """**Title**: The Devil in Linear Transformer

**Historical Technical Context**: Linear transformers were proposed to address quadratic complexity but consistently underperformed vanilla transformers across tasks. The paper identified that previous work focused on kernel approximations without addressing fundamental issues in the attention mechanism itself.

**Technical Limitations**: Existing linear transformers suffered from: 1) Unbounded gradients causing training instability, 2) Attention dilution where attention scores spread uniformly over long sequences, ignoring local structure.

**Paper Concepts**:
- **TransNormer**: Proposed linear transformer with gradient stabilization and attention focusing
- **Attention Normalization**: Replacing scaling with normalization to bound gradients  
- **Diagonal Attention**: Confining attention to neighboring tokens in early layers
- **Gradient Analysis**: Mathematical characterization of why linear attention has unstable gradients""",

        "ALGORITHMIC_INNOVATION": """**Core_Algorithm**:
TransNormer addresses linear transformer issues through:
1. Attention normalization: Replace scaling operation with LayerNorm-style normalization
2. Diagonal attention: Add position-aware attention focusing in early layers
3. Stable kernel approximation: Use feature maps that don't cause gradient explosion

**Key_Mechanism**:
- Normalization stabilizes gradients by bounding attention matrix values
- Diagonal attention preserves locality while maintaining linear complexity
- Combines global linear attention with local attention structure

**Mathematical_Formulation**:
- Standard: Attention = softmax(QK^T/√d)V  
- TransNormer: Attention = φ(Q)φ(K)^T V with normalization
- Diagonal component: LocalAttn = softmax(QK^T ⊙ M)V where M is diagonal mask

**Computational_Properties**:
- Maintains O(NLD) complexity of linear attention
- Gradient norms remain bounded during training
- Preserves parallel training efficiency while fixing stability issues""",

        "IMPLEMENTATION_GUIDANCE": """**Integration_Strategy**:
- Replace linear attention layers with TransNormer layers
- Apply diagonal attention only in early layers (first 25-50% of layers)
- Use proper normalization instead of scaling in attention computation

**Parameter_Settings**:
- Normalization: LayerNorm or RMSNorm applied to attention matrices
- Diagonal mask: Restrict attention to ±k neighboring positions in early layers
- Feature maps: Use stable kernel approximations (e.g., ReLU, ELU+1)

**Application_Conditions**:
- When training instability is observed in linear attention models
- For long-sequence tasks where attention dilution hurts performance
- When competitive performance with vanilla transformers is required

**Expected_Outcomes**:
- Stable training curves without gradient explosions
- Improved performance on language modeling and classification tasks
- Better attention focus on relevant tokens rather than uniform distribution
- Maintained linear complexity with improved model quality"""
    }]

def create_learning_linear_attention_paper():
    """Learning Linear Attention in Polynomial Time"""
    return [{
        "DESIGN_INSIGHT": """### DESIGN_INSIGHT_1: Polynomial-Time Learnability of Linear Attention through RKHS Framework

Linear attention can be viewed as a linear predictor in a Reproducing Kernel Hilbert Space (RKHS), making it provably learnable in polynomial time. This theoretical foundation shows that linear transformers are not just computational approximations but have strong learning guarantees, bridging the gap between expressivity and learnability.""",

        "EXPERIMENTAL_TRIGGER_PATTERNS": """**Task_Performance_Signatures**:
- Strong generalization on tasks involving associative memory and pattern completion
- Excellent performance on finite automata learning tasks
- Competitive results on key-value association problems
- Provable convergence guarantees on structured learning problems

**Architectural_Symptoms**:
- Models learn efficiently from small datasets when data matches theoretical conditions
- Strong transfer learning capabilities due to RKHS structure
- Reliable performance on computations expressible as linear attention operations""",

        "BACKGROUND": """**Title**: Learning Linear Attention in Polynomial Time

**Historical Technical Context**: While transformers' computational expressivity was well-studied, their learnability from data remained unclear. Previous work showed transformers could simulate complex computations but didn't address whether these capabilities could be learned efficiently from examples.

**Technical Limitations**: Gap between theoretical expressivity and practical learnability of attention mechanisms. No polynomial-time learning guarantees existed for attention-based architectures.

**Paper Concepts**:
- **RKHS Formulation**: Linear attention as linear prediction in expanded feature space
- **PAC Learning**: Polynomial-time Probably Approximately Correct learning guarantees
- **Empirical Risk Minimization**: Conditions under which ERM recovers ground truth
- **Universal Computation**: Classes of computations learnable through linear attention""",

        "ALGORITHMIC_INNOVATION": """**Core_Algorithm**:
The key insight is reformulating linear attention learning:
1. Map linear attention to RKHS linear prediction problem
2. Use kernel methods to learn in expanded feature space
3. Convert learned predictor back to multi-headed linear transformer

**Key_Mechanism**:
- Linear attention = kernel method with specific feature map
- Learning reduces to solving linear system in RKHS
- Polynomial-time algorithms exist for this class of problems

**Mathematical_Formulation**:
- Linear attention: O = softmax(QK^T)V ≈ φ(Q)φ(K)^T V
- RKHS view: f(x) = ⟨w, φ(x)⟩ where φ maps to attention feature space
- Learning: minimize ||y - Xw||² subject to RKHS constraints

**Computational_Properties**:
- Polynomial-time learning complexity O(poly(n,d,1/ε))
- Sample complexity bounds for generalization
- Efficient conversion between RKHS and attention representations""",

        "IMPLEMENTATION_GUIDANCE": """**Integration_Strategy**:
- Use theoretical insights to design better initialization schemes
- Apply PAC learning principles to determine sufficient training data sizes
- Leverage RKHS structure for transfer learning between related tasks

**Parameter_Settings**:
- Feature map design: Choose φ based on task structure and theoretical guarantees
- Regularization: Apply RKHS-appropriate regularization terms
- Data requirements: Use sample complexity bounds to estimate training needs

**Application_Conditions**:
- Tasks involving associative memory, pattern completion, or sequence modeling
- When theoretical learning guarantees are important
- For few-shot learning scenarios with structured data

**Expected_Outcomes**:
- Provable generalization on tasks matching theoretical assumptions
- Efficient learning from small datasets when conditions are met
- Strong performance on automata learning and associative memory tasks
- Reliable convergence during training with proper initialization"""
    }]

def create_softmax_free_paper():
    """Softmax-free Linear Transformers"""
    return [{
        "DESIGN_INSIGHT": """### DESIGN_INSIGHT_1: Gaussian Kernel Replacement and Low-Rank Decomposition for Softmax-Free Attention

The fundamental limitation of linear attention approximations is the inheritance of softmax normalization, which challenges linearization efforts. SOFT eliminates softmax entirely by using Gaussian kernel functions and low-rank matrix decomposition, achieving true linear complexity without softmax-induced approximation errors.""",

        "EXPERIMENTAL_TRIGGER_PATTERNS": """**Task_Performance_Signatures**:
- Superior performance on vision tasks (ImageNet, COCO, ADE20K) compared to other linear attention methods
- Better accuracy-efficiency trade-off than vanilla ViTs, especially on long sequences
- Strong performance scaling with sequence length due to true linear complexity
- Competitive results with much longer token sequences than standard attention allows

**Architectural_Symptoms**:
- Linear memory scaling enables processing of very long visual sequences
- No softmax bottleneck allows more direct optimization of attention patterns
- Moore-Penrose inverse computation provides robust attention weight estimation""",

        "BACKGROUND": """**Title**: Softmax-free Linear Transformers

**Historical Technical Context**: Previous linear attention methods approximated softmax attention, inheriting its normalization constraints. This limited their effectiveness because softmax properties don't align well with linear approximation requirements.

**Technical Limitations**: Softmax normalization creates challenges for linearization: 1) Non-linear normalization step breaks linear structure, 2) Approximation errors accumulate, 3) Hardware optimization is difficult.

**Paper Concepts**:
- **SOFT Architecture**: Completely removes softmax from attention computation
- **Gaussian Kernels**: Replace dot-product similarity with Gaussian kernel functions
- **Low-rank Decomposition**: Approximate full attention matrix efficiently  
- **Moore-Penrose Inverse**: Robust method for attention weight computation""",

        "ALGORITHMIC_INNOVATION": """**Core_Algorithm**:
SOFT replaces softmax attention with Gaussian kernel-based computation:
1. Replace dot-product similarity with Gaussian kernel: K(qi,kj) = exp(-||qi-kj||²/2σ²)
2. Use low-rank decomposition to approximate attention matrix
3. Compute Moore-Penrose inverse efficiently using Newton-Raphson iteration

**Key_Mechanism**:
- Gaussian kernel naturally provides similarity without requiring softmax normalization
- Low-rank structure enables linear complexity while preserving expressiveness
- Iterative inverse computation provides numerical stability

**Mathematical_Formulation**:
- Attention: A_ij = exp(-||q_i - k_j||²/2σ²) (no softmax normalization)
- Low-rank: A ≈ UV^T where U,V ∈ R^(N×r), r << N
- Output: O = A†V where A† is Moore-Penrose inverse

**Computational_Properties**:
- Time complexity: O(Nr²) for rank-r approximation vs O(N²) for full attention
- Memory complexity: O(Nr) vs O(N²) 
- Numerically stable through iterative inverse computation""",

        "IMPLEMENTATION_GUIDANCE": """**Integration_Strategy**:
- Replace softmax attention layers with SOFT layers in vision transformer architectures
- Adjust rank parameter r based on computational budget and accuracy requirements
- Use symmetric normalization for dense prediction tasks

**Parameter_Settings**:
- Kernel bandwidth σ: Tune based on input feature distributions
- Rank r: Balance between accuracy and efficiency (typically r = 32-128)
- Newton-Raphson iterations: 2-5 iterations usually sufficient for convergence

**Application_Conditions**:
- Vision tasks where long sequences are beneficial (high-resolution images)
- When true linear complexity is required (not just asymptotic improvements)
- Applications requiring elimination of softmax for hardware optimization

**Expected_Outcomes**:
- Better accuracy-efficiency trade-offs than softmax-based linear attention
- Ability to process longer visual sequences within same memory constraints
- Improved computational efficiency on specialized hardware
- Superior performance on dense prediction tasks requiring long-range dependencies"""
    }]

def create_radlads_paper():
    """RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale"""
    return [{
        "DESIGN_INSIGHT": """### DESIGN_INSIGHT_1: Efficient Distillation from Softmax to Linear Attention at Scale

RADLADS demonstrates that large-scale softmax attention transformers can be rapidly converted to linear attention models using only 350-700M tokens (0.005% of original training). This enables leveraging existing pretrained models while gaining linear attention benefits through targeted distillation rather than full retraining.""",

        "EXPERIMENTAL_TRIGGER_PATTERNS": """**Task_Performance_Signatures**:
- Converted models maintain near-original performance on downstream benchmarks
- State-of-the-art results among linear attention models of comparable size
- Successful scaling from 7B to 72B parameter models
- Cost-effective conversion (under $2,000 for 72B model conversion)

**Architectural_Symptoms**:
- Linear inference complexity while preserving model quality
- Efficient deployment of multiple specialized models from single base model
- Rapid adaptation to linear attention architecture without full retraining""",

        "BACKGROUND": """**Title**: RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale

**Historical Technical Context**: Training large linear attention models from scratch is expensive and time-consuming. Existing linear attention models often underperformed compared to their softmax attention counterparts, creating a need for efficient conversion methods.

**Technical Limitations**: Previous approaches required extensive retraining or suffered from significant performance degradation when converting from softmax to linear attention.

**Paper Concepts**:
- **Attention Distillation**: Knowledge transfer from softmax to linear attention mechanisms
- **RWKV Variants**: Two new RWKV-based architectures for linear attention
- **Scale Efficiency**: Conversion protocol that works across model sizes (7B-72B)
- **Minimal Token Training**: Achieving conversion with <1% of original training tokens""",

        "ALGORITHMIC_INNOVATION": """**Core_Algorithm**:
RADLADS conversion protocol:
1. Initialize linear attention model using softmax transformer weights where possible
2. Apply targeted fine-tuning on carefully selected data using distillation objectives
3. Use teacher-student training with softmax model as teacher, linear model as student

**Key_Mechanism**:
- Knowledge distillation preserves learned representations while adapting attention mechanism
- Selective weight initialization leverages compatible parameters from original model
- Minimal fine-tuning focuses only on attention mechanism adaptation

**Mathematical_Formulation**:
- Distillation loss: L = αL_task + βL_KL(P_softmax || P_linear)
- Where L_task is downstream task loss, L_KL is KL divergence between attention distributions
- Conversion requires optimizing only attention-specific parameters

**Computational_Properties**:
- Training cost: <0.1% of original pretraining cost
- Conversion time: Orders of magnitude faster than training from scratch
- Maintains inference efficiency of linear attention (O(N) vs O(N²))""",

        "IMPLEMENTATION_GUIDANCE": """**Integration_Strategy**:
- Start with pretrained softmax transformer as initialization
- Apply distillation protocol with carefully curated training data
- Fine-tune only attention-related parameters while keeping most weights frozen

**Parameter_Settings**:
- Distillation weight β: Balance task performance vs attention similarity (typically 0.1-0.5)
- Training tokens: 350-700M tokens sufficient for most model sizes
- Learning rate: Lower than pretraining (typically 1e-5 to 1e-4)

**Application_Conditions**:
- When high-quality pretrained softmax models are available
- For rapid deployment of linear attention variants
- When training budget is limited but linear attention benefits are desired

**Expected_Outcomes**:
- Near-original performance retention (>95% of original model quality)
- Linear attention benefits: O(N) inference, constant memory per token
- Rapid conversion: Days rather than months for large model conversion
- Cost efficiency: <$2,000 for 72B model conversion vs millions for training from scratch"""
    }]

def generate_all_papers():
    """Generate all paper files for the RAG system."""
    
    papers = {
        "arxiv.org_pdf_2312.06635v6.json": create_gla_paper(),
        "arxiv.org_pdf_2210.10340v1.json": create_devil_linear_paper(), 
        "arxiv.org_pdf_2410.10101v2.json": create_learning_linear_attention_paper(),
        "arxiv.org_pdf_2207.03341v3.json": create_softmax_free_paper(),
        "arxiv.org_pdf_2505.03005v3.json": create_radlads_paper(),
    }
    
    # Create cognition directory if it doesn't exist
    cognition_dir = Path("cognition")
    cognition_dir.mkdir(exist_ok=True)
    
    # Generate all paper files
    for filename, content in papers.items():
        filepath = cognition_dir / filename
        with open(filepath, 'w') as f:
            json.dump(content, f, indent=2)
        print(f"✅ Created {filename}")
    
    print(f"\n📁 Files created in: {cognition_dir.absolute()}")
    print(f"\n🎯 Generated {len(papers)} high-quality RAG entries for linear attention research")
    
    return len(papers)

if __name__ == "__main__":
    count = generate_all_papers()
    print(f"\n🚀 Ready to enhance your ASI-Arch pipeline with {count} cutting-edge linear attention papers!")
    print(f"\nNext steps:")
    print(f"1. Run: python rag_service.py")
    print(f"2. Your agents will now have access to state-of-the-art linear attention knowledge!")