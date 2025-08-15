def Planner_input(context: str) -> str:
    return f"""# DeltaNet Architecture Research

You are the lead architecture researcher. This is YOUR project.

## EVIDENCE
{context}

## TASK  
Implement breakthrough DeltaNet architecture using write_code_file tool.

## REQUIREMENTS
- Class: DeltaNet(torch.nn.Module) with __init__(**kwargs) and forward(x, **kwargs)
- Alias: Model = DeltaNet
- Use einops.rearrange() (never .view/.reshape)
- O(N log N) complexity or better
- Support any batch size

## HYBRID TEMPLATES
Linear attention + hierarchical reasoning:

1. **Sequential**: Linear attention → HRM reasoning → output
2. **Parallel**: Linear attention || HRM reasoning → fusion  
3. **Nested**: Multi-level token/sequence/reasoning integration

## OUTPUT
First use write_code_file, then provide JSON: {{"name": "delta_net_[innovation]", "motivation": "breakthrough explanation"}}

Execute directly as the researcher."""

