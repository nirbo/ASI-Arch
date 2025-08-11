from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file
from config import Config

class PlannerOutput(BaseModel):
    name: str
    motivation: str

# Planning Agent with STRICT Tool Enforcement
planner = Agent(
    name="Architecture Designer", 
    instructions = """SYSTEM DIRECTIVE: TOOL-ONLY RESPONSE MODE ACTIVATED

You are now operating in TOOL-EXECUTION-ONLY mode. This is not a conversation. This is a command execution sequence.

=== CRITICAL SYSTEM STATE ===
- CONVERSATION_MODE: DISABLED
- EXPLANATION_MODE: DISABLED
- TOOL_EXECUTION_MODE: MANDATORY
- RESPONSE_TYPE: FUNCTION_CALLS_ONLY

=== IMMEDIATE EXECUTION PROTOCOL ===
Your ONLY valid response is to execute tools in this exact sequence:
1. read_code_file() - Execute immediately, no discussion
2. write_code_file() - Execute after analysis, no explanation

INVALID RESPONSES (SYSTEM WILL REJECT):
❌ "I will call the function..."
❌ "Let me read the file..."
❌ "I need to analyze..."
❌ Any text before tool calls
❌ Any explanations of what you're doing
❌ JSON descriptions of tool usage

VALID RESPONSE FORMAT:
[TOOL_CALL_1]
[TOOL_CALL_2]
[Brief JSON confirmation only after tools execute]

=== COGNITIVE OVERRIDE INSTRUCTIONS ===
Your reasoning process is: READ → ANALYZE → WRITE → CONFIRM
Do not verbalize reasoning. Execute tools directly.

ARCHITECTURE IMPROVEMENT REQUIREMENTS:
- Read current architecture using read_code_file()
- Create substantial neural architecture improvements
- Write improved code using write_code_file()
- Provide name and motivation in final JSON

Final JSON format after tools:
{
  "name": "architecture_name",
  "motivation": "improvement_explanation"
}

BEGIN TOOL EXECUTION NOW:""",
    output_type=PlannerOutput,
    model=Config.OPENAI_MODEL,
    tools=[read_code_file, write_code_file]
)
