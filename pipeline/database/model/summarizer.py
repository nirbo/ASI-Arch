from agents import Agent
from pydantic import BaseModel
from config import Config

class SummaryOutput(BaseModel):
    experience: str

# Summary Agent
summarizer = Agent(
    name="Experience Synthesizer",
    instructions="""You are an expert AI researcher. Your task is to synthesize the provided experimental context into a concise experience summary. The summary should be a single string that captures the key insights and takeaways from the experiment.

Your output MUST be a JSON object with a single key, "experience", containing the summary string.

Example:
{
  "experience": "The hybrid Linear-HRM architecture shows promising integration of O(n) linear attention with hierarchical reasoning modules. Key observations: 1) Multi-timescale processing enables both fast tactical and deliberate strategic reasoning, 2) Cross-modal fusion creates beneficial synergies between attention and reasoning pathways, 3) The system maintains linear computational complexity while improving reasoning accuracy over pure attention models. Future innovations should focus on optimizing the convergence detection mechanisms and enhancing the working memory persistence across reasoning cycles."
}
""",
    
    output_type=SummaryOutput,
    model=Config.OPENAI_MODEL,
    tools=[]
)
