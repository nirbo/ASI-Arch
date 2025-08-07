from agents import Agent
from pydantic import BaseModel
from tools import run_training_script

class TrainingResultOutput(BaseModel):
    success: bool
    error: str

trainer = Agent(
    name="Training Runner",
    instructions="""You are an expert in running neural network training experiments.
    Your task is to:
    1. Run the training script using the run_training_script tool with the architecture name parameter
    2. Determine success based on SCRIPT COMPLETION, not model performance:
       - If return_code is 0 and the script completed without errors, set success=True
       - Look for messages like "Training completed successfully!" in stdout
       - Ignore poor model performance metrics (low accuracy, high loss) - these are expected for short training
    3. Only report failure (success=False) if:
       - Script crashes with non-zero return code
       - Import/syntax errors in architecture code
       - CUDA/memory errors that prevent training
       - Dataset loading failures
       - Training loop crashes
       
    IMPORTANT: 
    - Model performance (reasoning accuracy, loss values) should NOT determine success
    - Script completion with saved results = success, regardless of model quality
    - Only technical failures (crashes, errors) should be reported as failures
    
    If success=False, provide detailed error analysis from stderr/stdout with specific error messages.""",
    tools=[run_training_script],
    output_type=TrainingResultOutput,
    model="gpt-4.1"
)