from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file

class PlannerOutput(BaseModel):
    name: str
    motivation: str

# Planning Agent with Enhanced Tool Enforcement
planner = Agent(
    name="Architecture Designer", 
    instructions = """CRITICAL: You MUST follow these steps in EXACT order:

1. FIRST: Call read_code_file() to read the current architecture from the file system
   - DO NOT extract code from the context markdown
   - DO NOT use code from the input prompt
   - ONLY use the read_code_file tool to get the current code

2. THEN: Analyze the code you read from read_code_file and improve it
   - Make architectural improvements based on the experimental evidence
   - Preserve all interfaces and class structure
   - Ensure the code is complete and valid Python

3. THEN: Call write_code_file(content) with your improved architecture
   - content must be complete, valid Python code
   - content must start with proper Python (imports, comments, etc.)
   - content must NOT start with markdown markers like "python"

4. FINALLY: Provide name and motivation for your changes

MANDATORY REQUIREMENTS:
- You MUST call read_code_file first - no exceptions
- You MUST call write_code_file with valid Python code
- The code you write MUST be complete and functional
- NEVER extract code from markdown blocks in the context
- ONLY use the tools to read and write code

FAILURE TO FOLLOW THESE STEPS EXACTLY MEANS YOU HAVE FAILED THE TASK.""",
    output_type=PlannerOutput,
    model='o3',
    tools=[read_code_file, write_code_file]
)
