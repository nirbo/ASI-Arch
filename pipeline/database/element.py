from dataclasses import dataclass, asdict
from typing import Dict, Optional

from config import Config
from utils.agent_logger import log_agent_run
from .model import summarizer
from .prompt import Summary_input


@dataclass
class DataElement:
    """Data element model for experimental results."""
    time: str
    name: str
    result: Dict[str, str]
    program: str
    motivation: str
    analysis: str
    cognition: str
    log: str
    parent: Optional[int] = None
    index: Optional[int] = None
    summary: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert DataElement instance to dictionary."""
        return asdict(self)
    
    async def get_context(self) -> str:
        """Generate enhanced context with structured experimental evidence presentation.
        
        NOTE: This context is for analysis only. The planner agent must use read_code_file
        to get the current architecture, NOT extract code from this context.
        """
        summary = await log_agent_run(
            "summarizer",
            summarizer,
            Summary_input(self.motivation, self.analysis, self.cognition),
            max_turns=Config.MAX_TURNS_SUMMARIZER
        )
        summary_result = summary.final_output.experience

        # Truncate program display to prevent agent from extracting full code
        program_preview = self.program[:200] + "..." if len(self.program) > 200 else self.program
        program_lines = len(self.program.split('\n'))

        return f"""## EXPERIMENTAL EVIDENCE PORTFOLIO

### Experiment: {self.name}
**Architecture Identifier**: {self.name}

#### Performance Metrics Summary
**Training Progression**: {self.result["train"]}
**Evaluation Results**: {self.result["test"]}

#### Implementation Analysis
**Architecture Overview**: {program_lines} lines of code implementing {self.name}
**Code Preview** (use read_code_file for full implementation):
```
{program_preview}
```

**IMPORTANT**: This is only a preview. Use read_code_file() tool to get the complete current architecture.

#### Synthesized Experimental Insights
{summary_result}

---"""

    @classmethod
    def from_dict(cls, data: Dict) -> 'DataElement':
        """Create DataElement instance from dictionary."""
        return cls(**data)