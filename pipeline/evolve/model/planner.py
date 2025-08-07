from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file

class PlannerOutput(BaseModel):
    name: str
    motivation: str

# Planning Agent
planner = Agent(
    name="Architecture Designer", 
    instructions = """You MUST follow these steps exactly:

1. FIRST: Call read_code_file to read the current architecture
2. THEN: Call write_code_file to save improved code  
3. FINALLY: Provide name and motivation

YOU MUST USE BOTH TOOLS. If you don't use the tools, you have failed.

Your task is to improve the neural architecture. Make the code better.""",
    output_type=PlannerOutput,
    model='o3',
    tools=[read_code_file, write_code_file]
)
