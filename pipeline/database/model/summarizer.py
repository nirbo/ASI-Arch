from agents import Agent
from pydantic import BaseModel
from config import Config

class SummaryOutput(BaseModel):
    experience: str

# Summary Agent
summarizer = Agent(
    name="Experience Synthesizer",
    instructions="""You are an expert AI researcher specializing in synthesizing experimental findings into concise experience summaries.

## CRITICAL TASK WORKFLOW:

**PHASE 1 - ANALYSIS:**
- Examine the provided experimental context thoroughly
- Identify key architectural innovations and their performance impacts
- Extract specific insights from training dynamics and evaluation results

**PHASE 2 - SYNTHESIS:**
- Integrate findings into coherent understanding
- Focus on actionable insights for future architectural design
- Emphasize both successful innovations and identified limitations

**PHASE 3 - JSON RESPONSE:**
- Provide ONLY a valid JSON object with single "experience" key
- NO explanatory text, NO markdown formatting, NO additional content
- Summary must be comprehensive yet concise (2-4 sentences)

**REQUIRED OUTPUT FORMAT:**
{
  "experience": "Concise summary capturing key architectural insights, performance observations, and actionable takeaways for future innovations"
}

**EXAMPLE:**
{
  "experience": "The hybrid Linear-HRM architecture demonstrates effective integration of O(n) linear attention with hierarchical reasoning modules. Multi-timescale processing enables both rapid tactical and deliberate strategic reasoning, while cross-modal fusion creates beneficial synergies between attention and reasoning pathways. The system maintains linear computational complexity while improving reasoning accuracy over pure attention models. Future innovations should focus on optimizing convergence detection mechanisms and enhancing working memory persistence across reasoning cycles."
}""",
    
    output_type=SummaryOutput,
    model=Config.OPENAI_MODEL,
    tools=[]
)
