from dataclasses import dataclass, asdict
from typing import Dict, Optional

from utils.agent_logger import log_agent_run
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
        """Generate enhanced context with structured experimental evidence presentation."""
        # Lazy import to avoid circular dependency
        from .model import summarizer
        
        summary = await log_agent_run(
            "summarizer",
            summarizer,
            Summary_input(self.motivation, self.analysis, self.cognition),
            max_turns=30
        )
        summary_result = summary.final_output.experience

        # Extract key architecture details without full code to preserve research insights
        # while keeping context manageable
        program_lines = self.program.split('\n')
        key_lines = []
        
        # Extract class definitions and key architectural elements
        for line in program_lines[:50]:  # First 50 lines usually contain key architecture info
            if any(keyword in line.lower() for keyword in ['class ', 'def forward', 'def __init__', 'titans', 'memory', 'attention']):
                key_lines.append(line.strip())
        
        architecture_summary = '\n'.join(key_lines[:10])  # Max 10 key lines
        
        return f"""## EXPERIMENTAL EVIDENCE PORTFOLIO

### Experiment: {self.name}
**Architecture Identifier**: {self.name}

#### Performance Metrics Summary
**Training Progression**: {self.result["train"]}
**Evaluation Results**: {self.result["test"]}

#### Key Architecture Elements
```python
{architecture_summary}
... [Full implementation available but truncated for context efficiency]
```

#### Synthesized Experimental Insights
{summary_result}

---"""

    @classmethod
    def from_dict(cls, data: Dict) -> 'DataElement':
        """Create DataElement instance from dictionary."""
        return cls(**data)