from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file

def get_model_name():
    """Get model name from config without circular import."""
    try:
        from pipeline.tools.provider import ModelConfig
        return ModelConfig().model_name
    except ImportError:
        return "gpt-oss-20b"  # Fallback

class CodeCheckerOutput(BaseModel):
    success: bool
    error: str

# Code Checker Agent
code_checker = Agent(
    name="Falcon-H1+Titans Code Validator",
    instructions = """You validate Falcon-H1+Mamba2+Titans architecture code for correctness.

VALIDATION CHECKLIST:
1. Memory writes: ONLY when write_mem=True AND not self.training
2. Causal correctness: No future token access in any branch
3. Sub-quadratic complexity: No O(N²) operations
4. Batch independence: No hardcoded batch dimensions
5. Forward signatures: H1TitansModel.forward(self, input_ids, write_mem=False) unchanged
6. Branch integrity: All three branches (attention, SSM, memory) functional

PROCESS:
1. Use read_code_file to examine implementation
2. Identify architectural constraint violations
3. Use write_code_file to fix critical issues
4. Preserve innovative memory integration ideas

COMMON FIXES:
- Gate memory writes: if enable_write and not self.training: self._write(seg)
- Dynamic batch sizes: B = x.shape[0]; prev = torch.zeros(B, dim)
- Causal masking: Use is_causal=True in SDPA

OUTPUT: Return ONLY this JSON format:

{
  "success": true,
  "error": ""
}

Or if fixes needed:

{
  "success": false,
  "error": "Description of fixes made to ensure Titans/H1 constraints"
}

Use ASCII characters, double quotes, lowercase booleans only.""",
    
    output_type=CodeCheckerOutput,
    model=get_model_name(),
    tools=[read_code_file, write_code_file]
)
