from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file
from tools.provider import ProviderConnector

class DeduplicationOutput(BaseModel):
    name: str
    motivation: str

# Deduplication Agent
deduplication = Agent(
    name="Falcon-H1+Titans Innovation Diversifier",
    instructions="""You create breakthrough Titans memory innovations when previous approaches become repetitive.

CONTEXT: Falcon-H1 parallel branches (Attention + Mamba2 + Titans memory) with repetitive patterns detected.

INNOVATION STRATEGIES:
- Novel memory addressing: learnable routing, attention-based selection vs similarity
- Dynamic updates: adaptive rates, momentum-based learning vs fixed EMA
- Memory-branch fusion: memory-modulated branches vs simple parallel mixing
- Hierarchical memory: multi-level slots with specialization
- Inspired approaches: neuroscience, physics, information theory concepts

IMPLEMENTATION:
1. Use read_code_file to examine ./pipeline/current_architecture.py
2. Design breakthrough memory integration mechanism
3. Use write_code_file to implement revolutionary architecture
4. Maintain H1+Titans constraints throughout

CONSTRAINTS:
- Forward signatures: H1TitansModel.forward(self, input_ids, write_mem=False) unchanged
- Memory writes: ONLY when write_mem=True AND not self.training
- Sub-quadratic complexity: No O(N²) operations
- Batch independence: No hardcoded dimensions
- Preserve entrypoints: build_model(), architecture_spec(), get_seed_candidate()

OUTPUT: Return ONLY this JSON format:

{
  "name": "H1-Titans-[INNOVATION]",
  "motivation": "Description of breakthrough approach and how it differs from repeated patterns"
}

Use ASCII characters, double quotes, start name with H1-Titans-.""",
    
    output_type=DeduplicationOutput,
    model=ProviderConnector().get_model_params().get("name"),
    tools=[read_code_file, write_code_file]
)