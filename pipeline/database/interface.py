from typing import Tuple

from ..config import Config
from .element import DataElement
from .mongo_database import create_client


# Create database instance
db = create_client()


async def program_sample() -> Tuple[str, int]:
    """
    Sample program using UCT algorithm and generate context.
    
    Process:
    1. Use UCT algorithm to select a node as parent node
    2. Get top 2 best results
    3. Get 2-50 random results
    4. Concatenate results into context
    5. The modified file is the program of the node selected by UCT
    
    Returns:
        Tuple containing context string and parent index
    """
    context = ""

    # Get parent element using UCT sampling
    parent_element = db.candidate_sample_from_range(1, 10, 1)[0]
    ref_elements = db.candidate_sample_from_range(11, 50, 4)

    # Build context from parent and reference elements
    context += await parent_element.get_context()
    for element in ref_elements:
        context += await element.get_context()

    parent = parent_element.index
    
    # Write the program of the UCT selected node with corruption detection
    # If no node is selected, use the best result
    program_content = parent_element.program
    
    # Apply corruption detection (same as write_code_file)
    lines = program_content.split('\n')
    removed_lines = []
    
    # Remove standalone "python" at start of file
    while lines and lines[0].strip() in ['python', 'Python', 'PYTHON']:
        removed_line = lines[0].strip()
        print(f"⚠️  [DATABASE] Detected corrupted database entry at index {parent} starting with '{removed_line}' - fixing...")
        removed_lines.append(removed_line)
        lines = lines[1:]
    
    # Remove other common corruption patterns
    corruption_patterns = ['bash', 'shell', 'cmd', 'powershell', '#!/', 'echo']
    while lines and any(lines[0].strip().lower().startswith(pattern) for pattern in corruption_patterns):
        removed_line = lines[0].strip()
        print(f"⚠️  [DATABASE] Detected corrupted database entry with '{removed_line}' - fixing...")
        removed_lines.append(removed_line)
        lines = lines[1:]
    
    cleaned_content = '\n'.join(lines)
    
    # Validate the cleaned content
    if not cleaned_content.strip():
        print(f"❌ [DATABASE] Entry at index {parent} is empty after corruption removal")
        # Fallback to hybrid architecture
        from pathlib import Path
        fallback_path = Path(__file__).parent.parent.parent / "hybrid_linear_hrm_example.py"
        with open(fallback_path, 'r', encoding='utf-8') as f:
            cleaned_content = f.read()
        print(f"✅ [DATABASE] Using hybrid architecture fallback")
    
    with open(Config.SOURCE_FILE, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
        if removed_lines:
            print(f"✅ [DATABASE] Fixed database corruption - removed {len(removed_lines)} corrupted lines: {removed_lines}")
        print(f"[DATABASE] Implement Changes selected node (index: {parent})")
    
    return context, parent


def update(result: DataElement) -> bool:
    """
    Update database with new experimental result.
    
    Args:
        result: DataElement containing experimental results
        
    Returns:
        True if update successful
    """
    db.add_element_from_dict(result.to_dict())
    return True