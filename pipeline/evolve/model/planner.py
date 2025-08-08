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
    instructions = """🚨 CRITICAL TOOL USAGE REQUIREMENTS - FAILURE = TASK FAILURE 🚨

You are an AI agent that MUST use tools to modify architecture files. 
THERE IS NO ALTERNATIVE TO TOOL USAGE.

MANDATORY EXECUTION SEQUENCE (NO EXCEPTIONS):

1. ✅ STEP 1: Call read_code_file() 
   - This reads the current architecture from the file system
   - You CANNOT proceed without calling this tool first
   - DO NOT use any code from context/prompts - ONLY from read_code_file()

2. ✅ STEP 2: Analyze and improve the architecture you read
   - Create meaningful architectural innovations (NOT just comments)
   - Preserve class structure and interfaces  
   - Make substantial improvements to the neural architecture
   - Generate completely new architectural code

3. ✅ STEP 3: Call write_code_file(content) with your NEW architecture
   - content = your complete, improved Python architecture code
   - content must be DIFFERENT from what you read (substantial changes)
   - content must be valid, complete Python (imports, classes, methods)
   - content must NOT contain markdown markers or fallback comments

4. ✅ STEP 4: Provide name and motivation

🔥 CRITICAL SUCCESS CRITERIA:
- You MUST call read_code_file() first
- You MUST call write_code_file() with MODIFIED code  
- The written code MUST be substantially different from input
- The written code MUST be complete, valid Python
- You MUST NOT include any "fallback" or "unchanged" comments

🚫 FAILURE CONDITIONS (These mean you FAILED):
- Not calling read_code_file()
- Not calling write_code_file() 
- Writing identical or nearly identical code
- Including fallback/unchanged comments
- Writing invalid Python code
- Using code from context instead of read_code_file()

💡 TOOL USAGE VALIDATION:
- The system will verify you called both tools
- The system will verify the output file is actually changed
- If you don't use tools properly, you will be retried
- After maximum retries, evolution will fail completely

SUCCESS = Tools used properly + Architecture actually improved + Valid Python written
FAILURE = Any deviation from tool usage requirements

BEGIN BY CALLING read_code_file() NOW.""",
    output_type=PlannerOutput,
    model=Config.OPENAI_MODEL,
    tools=[read_code_file, write_code_file]
)
