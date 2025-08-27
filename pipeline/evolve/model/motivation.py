from agents import Agent
from pydantic import BaseModel
from tools.provider import ProviderConnector

class MotivationCheckOutput(BaseModel):
    is_repeated: bool
    repeated_index: list[int]
    judgement_reason: str

motivation_checker = Agent(
    name="Falcon-H1+Titans Motivation Validator",
    instructions="""You validate whether new Falcon-H1+Titans research motivations duplicate previous work.

ARCHITECTURE FOCUS: Falcon-H1 with parallel attention + Mamba2 SSM + Titans memory integration.

DUPLICATION RULES:
- DUPLICATE: Same Titans variant (MAG/MAC/MAL) with identical implementation
- DUPLICATE: Identical mixer strategy (SUM/CONCAT/GATED) with same parameters  
- DUPLICATE: Same memory policy and hyperparameters
- NOT DUPLICATE: Different memory variants (MAG vs MAC vs MAL)
- NOT DUPLICATE: Different mixer approaches or memory configurations
- NOT DUPLICATE: Different optimization targets or performance trade-offs

OUTPUT: Respond with ONLY this JSON format, nothing else:

{
  "is_repeated": false,
  "repeated_index": [],
  "judgement_reason": "Explain architectural differences from previous work"
}

If duplicate found, set is_repeated to true and list indices in repeated_index array.

Use only ASCII characters, double quotes, lowercase booleans. No text before or after JSON.""",
    output_type=MotivationCheckOutput,
    tools=[],
    model=ProviderConnector().get_model_params().get("name"),
)
