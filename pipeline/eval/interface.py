import os
from typing import Tuple

from ..config import Config
from ..utils.agent_logger import log_agent_run
from .model import debugger, trainer
from .prompts import Debugger_input


async def evaluation(name: str, motivation: str) -> bool:
    """
    Evaluate training performance for a given experiment.
    
    Args:
        name: Experiment name
        motivation: Experiment motivation
        
    Returns:
        True if training successful, False otherwise
    """
    success, error_msg = await run_training(name, motivation)
    if not success:
        print(f"Training failed: {error_msg}")
        return False
    save(name)
    return True


async def run_training(name: str, motivation: str) -> Tuple[bool, str]:
    """
    Run training script with debugging retry mechanism.
    
    Args:
        name: Experiment name
        motivation: Experiment motivation
        
    Returns:
        Tuple of (success_flag, error_message)
    """
    try:
        debug = False
        previous_error = ""
        
        for attempt in range(Config.MAX_DEBUG_ATTEMPT):
            if debug:
                debug_result = await log_agent_run(
                    "debugger",
                    debugger,
                    Debugger_input(motivation, previous_error),
                    max_turns=Config.MAX_TURNS_DEBUGGER
                )
                
                changes_made = debug_result.final_output.changes_made
                print(f"Debug changes for {name}: {changes_made}")

            train_result = await log_agent_run(
                "trainer",
                trainer,
                f"""Please run the training script for architecture: {name}
                Use the run_training_script tool with the architecture name as parameter.
                Return success=True only if the training completes successfully.""",
                max_turns=Config.MAX_TURNS_TRAINER
            )
            
            if train_result.final_output.success:
                print(f"Training successful for {name}")
                return True, ""
            else:
                debug = True
                # Read debug file content as detailed error information
                try:
                    # If debug file doesn't exist, create an empty file
                    if not os.path.exists(Config.DEBUG_FILE):
                        with open(Config.DEBUG_FILE, 'w', encoding='utf-8') as f:
                            f.write("")

                    with open(Config.DEBUG_FILE, 'r', encoding='utf-8') as f:
                        debug_content = f.read()
                    previous_error = f"Training failed. Debug info:\n{debug_content}"
                except Exception as e:
                    previous_error = (
                        f"Training failed. Cannot read debug file {Config.DEBUG_FILE}: {str(e)}"
                    )
                
                print(f"Training failed for {name} (attempt {attempt + 1}): {previous_error}")
                
                # If this is the last attempt, return failure
                if attempt == Config.MAX_DEBUG_ATTEMPT - 1:
                    return False, (
                        f"Training failed after {Config.MAX_DEBUG_ATTEMPT} attempts. "
                        f"Final error: {previous_error}"
                    )
                
                continue
                
    except Exception as e:
        error_msg = f"Unexpected error during training: {str(e)}"
        print(error_msg)
        return False, error_msg


def save(name: str) -> None:
    """
    Save source file content to code pool with given name.
    
    Args:
        name: File name to save as
    """
    import os
    
    # Ensure pool directory exists
    os.makedirs(Config.CODE_POOL, exist_ok=True)
    
    with open(Config.SOURCE_FILE, "r", encoding='utf-8') as f:
        content = f.read()
    
    # Fix corruption pattern before saving to pool
    lines = content.split('\n')
    if lines and lines[0].strip() == 'python':
        print(f"⚠️  Detected corrupted architecture file starting with 'python' - fixing before saving to pool...")
        content = '\n'.join(lines[1:])  # Remove first line
        print(f"✅ Fixed architecture file corruption before saving to pool")
    
    with open(f"{Config.CODE_POOL}/{name}.py", "w", encoding='utf-8') as f:
        f.write(content)