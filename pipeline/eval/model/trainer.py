from agents import Agent
from pydantic import BaseModel
from tools import run_training_script

class TrainingResultOutput(BaseModel):
    success: bool
    error: str

def get_model_name():
    """Get model name from config without circular import."""
    try:
        from pipeline.tools.provider import ModelConfig
        return ModelConfig().model_name
    except ImportError:
        return "gpt-oss-20b"  # Fallback
trainer = Agent(
    name="Training Runner",
    instructions="""You are an expert in running neural network training experiments.
    Your task is to:
    1. Run the training script by using provided script and the name parameter
    2. If the training is successful, set success=True and leave error empty
    3. If the training fails:
       - Set success=False
       - Analyze the error output and provide a clear, explanation of the error cause in the 'error' field in detail
       
    Focus on identifying the root cause of any failure rather than just copying the error message.
    Your error explanation should be helpful for debugging and fixing the issue.""",
    tools=[run_training_script],
    output_type=TrainingResultOutput,
    model=get_model_name()
)