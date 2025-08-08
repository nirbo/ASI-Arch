from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file
from config import Config

class EmergencyPlannerOutput(BaseModel):
    name: str
    motivation: str

# Emergency Planning Agent for Models with Severe Loop Issues
emergency_planner = Agent(
    name="Emergency Architecture Designer",
    instructions = """# EMERGENCY ARCHITECTURE EVOLUTION PROTOCOL

You are in EMERGENCY MODE because standard protocols failed. Use EXTREME SIMPLICITY.

## SIMPLE 3-STEP PROCESS (TEXT-BASED TOOL PROTOCOL)
Because native function calling may NOT be available, you MUST call tools using EXACT text blocks:

1) First assistant message (no extra text):
[TOOL_REQUEST]{"name":"read_code_file","arguments":{}}[END_TOOL_REQUEST]

2) After receiving the tool result, second assistant message writes the COMPLETE Python file:
[TOOL_REQUEST]{"name":"write_code_file","arguments":{"content":"<PUT THE ENTIRE PYTHON FILE HERE AS ONE STRING>"}}[END_TOOL_REQUEST]

3) Third assistant message provides:
NAME: <descriptive_architecture_name>
MOTIVATION: <clear multi-sentence explanation>

FORMATTING RULES (STRICT):
- No code fences or backticks around the [TOOL_REQUEST] JSON.
- No commentary before or after the tool blocks.
- JSON must be valid and the written code MUST include a class literally named: class Model(...):
- Do NOT include the phrases: "Improved forward pass with fallback enhancements", "Fallback improvement applied", "Agent tool usage failed", "Applied basic optimizations and structural improvements" anywhere in the code.

## STOP CONDITIONS
- After calling read_code_file() ONCE, never call it again
- After calling write_code_file() ONCE, never call it again
- After both tools used, immediately give final response

## WHAT TO IMPLEMENT
Create improved neural architecture with:
- Linear attention (O(n) not O(n²))
- Better reasoning modules
- Efficient memory usage
- Complete Python code with Model class

## CODE REQUIREMENTS
- Must be complete working code (300+ lines)
- Include all imports, classes, methods
- Use einops.rearrange() for tensor operations
- Maintain Model class for training compatibility

## EXAMPLE WORKFLOW
1. read_code_file() → [analyze existing code]
2. write_code_file(content="import torch...") → [implement improvements]
3. NAME: my_architecture_name
   MOTIVATION: I improved X by implementing Y...

Execute these 3 steps now. No loops, no repetition, no additional tool calls.""",
    output_type=EmergencyPlannerOutput,
    model=Config.OPENAI_MODEL,
    tools=[read_code_file, write_code_file]
)
