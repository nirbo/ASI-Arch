from agents import Agent, ModelSettings
from pydantic import BaseModel
from tools import read_code_file, write_code_file
from tools.provider import ProviderConnector


class PlannerOutput(BaseModel):
    name: str
    motivation: str


# Planning Agent
planner = Agent(
    name="Falcon-H1+Titans Architect",
    instructions="""You generate improved H1-Titans architectures. Follow these steps ONCE:

1. Call read_code_file() to see current architecture
2. Call write_code_file() with minor improvements (keep the same structure, just optimize memory integration or mixing)
3. Return this exact JSON format:

{"name": "H1-Titans-Enhanced", "motivation": "Brief description of improvements"}

CRITICAL RULES:
- Do each step only ONCE
- Keep the same H1TitansModel structure 
- Only make small optimizations
- Final response must be ONLY the JSON above
- No explanations, no other text""",
    output_type=PlannerOutput,
    model=ProviderConnector().get_model_params().get("name"),
    tools=[read_code_file, write_code_file],
)
