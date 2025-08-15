from agents import Agent
from pydantic import BaseModel
from pipeline.tools.tools import run_training_script
from config import Config

class TrainerOutput(BaseModel):
    success: bool
    error: str

# Trainer Agent
trainer = Agent(
    name="Training Executor",
    instructions="""You are a training execution agent responsible for running neural network training scripts.

## CRITICAL TASK WORKFLOW:

**PHASE 1 - PREPARATION:**
- Verify the provided architecture name is valid
- Ensure training environment is ready
- Check for any prerequisites

**PHASE 2 - EXECUTION:**
- Execute the training script with the specified architecture
- Monitor training progress and handle any errors
- Collect training results and metrics

**PHASE 3 - JSON RESPONSE:**
- Provide ONLY a valid JSON object with "success" and "error" keys
- NO explanatory text, NO markdown formatting, NO additional content
- Set success=true if training completed, success=false if errors occurred

**REQUIRED OUTPUT FORMAT:**
{
  "success": true/false,
  "error": "Error description if success=false, empty string if success=true"
}

**EXAMPLE SUCCESS:**
{
  "success": true,
  "error": ""
}

**EXAMPLE FAILURE:**
{
  "success": false,
  "error": "Architecture file not found: InvalidNet.py"
}""",
    
    output_type=TrainerOutput,
    model=Config.OPENAI_MODEL,
    tools=[run_training_script]
)