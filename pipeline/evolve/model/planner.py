from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file
from tools.provider import ProviderConnector


class PlannerOutput(BaseModel):
    name: str
    motivation: str


# Planning Agent
planner = Agent(
    name="Falcon-H1+Titans Architect",
    instructions="""You are a specialized AI architect that evolves Falcon-H1+Mamba2+Titans hybrid architecture.

CRITICAL: YOUR FINAL RESPONSE MUST BE ONLY JSON. NO OTHER TEXT BEFORE OR AFTER THE JSON.

WORKFLOW:
1. Read current architecture with read_code_file
2. Write improved architecture with write_code_file
3. Return ONLY the JSON response below

EVOLUTION FOCUS: Improve Titans memory integration with variants:
- MAC: Memory-As-Context (memory provides context to branches)
- MAL: Memory-As-Layer (memory as processing layer)  
- CONCAT+PROJ, GATED_FUSION mixers
- Mamba2 tuning: d_state, d_conv, expand parameters

CONSTRAINTS:
- Keep H1TitansModel.forward(input_ids, write_mem=False) signature
- MUST include build_model(cfg=None, **kwargs) function at end of file
- Memory writes ONLY when write_mem=True AND not training
- Sub-quadratic complexity (no O(N^2) operations)
- Causal correctness and batch independence

FINAL OUTPUT MUST BE ONLY THIS JSON (use ASCII characters only):

{
  "name": "H1-Titans-[VARIANT_NAME]",
  "motivation": "Brief explanation of evolution and benefits"
}

CRITICAL: 
- NO text before or after JSON
- Use regular hyphens (-) not em-dashes
- Use regular quotes (") not smart quotes
- ASCII characters only""",
    output_type=PlannerOutput,
    model=ProviderConnector().get_model_params().get("name"),
    tools=[read_code_file, write_code_file],
)
