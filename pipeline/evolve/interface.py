from .prompt import Planner_input, Motivation_checker_input, Deduplication_input, CodeChecker_input
from .model import planner, motivation_checker, deduplication, code_checker
from agents import exceptions, set_tracing_disabled
from typing import List, Tuple
from config import Config
from ..database.mongo_database import create_client
from ..utils.agent_logger import log_agent_run, log_info, log_warning, log_error
import re

def parse_planner_response(response_text: str) -> Tuple[str, str]:
    """Parse plain text response from planner into name and motivation."""
    # Look for "NAME: something" and "MOTIVATION: something" patterns
    name_match = re.search(r'NAME:\s*(.+)', response_text, re.IGNORECASE)
    motivation_match = re.search(r'MOTIVATION:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
    
    if name_match and motivation_match:
        name = name_match.group(1).strip()
        motivation = motivation_match.group(1).strip()
        return name, motivation
    
    # Fallback: try to extract from any structured format
    lines = response_text.split('\n')
    name = "evolved_architecture"  # Default fallback
    motivation = response_text[:1000]  # First 1000 chars
    
    # Try to find lines that look like names
    for line in lines:
        if any(word in line.lower() for word in ['architecture', 'model', 'network', 'design']):
            clean_line = re.sub(r'[^\w\s_]', '', line).strip()
            if len(clean_line) > 3 and len(clean_line) < 50:
                name = clean_line.lower().replace(' ', '_')
                break
    
    return name, motivation

async def evolve(context: str) -> Tuple[str, str]:
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        with open(Config.SOURCE_FILE, 'r') as f:
            original_source = f.read()
            
        name, motivation = await gen(context)
        
        if await check_code_correctness(motivation):
            return name, motivation

        with open(Config.SOURCE_FILE, 'w') as f:
            f.write(original_source)
        print("Try new motivations")
    return "Failed", "evolve error"
    
async def gen(context: str) -> Tuple[str, str]:
    # Save original file content
    with open(Config.SOURCE_FILE, 'r') as f:
        original_source = f.read()
        
    repeated_result = None
    motivation = None
    
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        try:
            # Restore original file
            with open(Config.SOURCE_FILE, 'w') as f:
                f.write(original_source)
            
            # Verify file was written correctly
            with open(Config.SOURCE_FILE, 'r') as f:
                restored_content = f.read()
            
            if restored_content != original_source:
                print(f"⚠️  File restoration failed on attempt {attempt + 1}")
                continue
            
            message = f"🔄 Evolution attempt {attempt + 1}"
            print(message)
            log_info(message)
            
            # Use different prompt based on whether it's repeated
            plan = None
            if attempt == 0 or repeated_result is None:
                input = Planner_input(context)
                plan = await log_agent_run("planner", planner, input, max_turns=Config.MAX_TURNS_PLANNER)
            else:
                repeated_context = get_repeated_context(repeated_result.repeated_index)
                input = Deduplication_input(context, repeated_context)
                plan = await log_agent_run("deduplication", deduplication, input, max_turns=Config.MAX_TURNS_DEDUPLICATION)
            
            # Validate that the agent actually provided output
            if not plan:
                print(f"❌ Agent failed to provide output on attempt {attempt + 1}")
                repeated_result = None  # Reset for non-repetition retry
                continue
                
            # Parse plain text response (since output_type is disabled for tool calling)
            response_text = str(plan)
            name, motivation = parse_planner_response(response_text)
            
            # Try to extract from final_output if available
            if hasattr(plan, 'final_output') and plan.final_output:
                try:
                    if hasattr(plan.final_output, 'name') and hasattr(plan.final_output, 'motivation'):
                        name = plan.final_output.name
                        motivation = plan.final_output.motivation
                except:
                    pass  # Use parsed values as fallback
            
            message = f"🧠 Agent plan: {name} - {motivation[:100]}{'...' if len(motivation) > 100 else ''}"
            print(message)
            log_info(message)
            
            # Validate that we got meaningful output
            if not name or not motivation or name.strip() == "" or motivation.strip() == "":
                message = f"❌ Agent provided empty name or motivation on attempt {attempt + 1}"
                print(message)
                log_warning(message)
                repeated_result = None  # Reset for non-repetition retry
                continue
            
            # CRITICAL: Validate that the file was actually modified (agent MUST use write_code_file)
            with open(Config.SOURCE_FILE, 'r') as f:
                final_content = f.read()
            
            # Check if agent actually used write_code_file tool
            tool_usage_success = final_content != original_source
            meaningful_changes = len(final_content) != len(original_source) or final_content != original_source
            
            if not tool_usage_success:
                message = f"❌ Agent failed to use write_code_file tool on attempt {attempt + 1}"
                print(message)
                log_warning(message)
                
                # FORCE RETRY - DO NOT use fallback for tool usage failures
                if attempt < Config.MAX_RETRY_ATTEMPTS - 1:
                    retry_message = f"🔄 Forcing retry {attempt + 2} - agent MUST use write_code_file"
                    print(retry_message)
                    log_info(retry_message)
                    repeated_result = None  # Reset for non-repetition retry
                    continue
                else:
                    error_message = f"❌ EVOLUTION FAILURE: Agent consistently failed to use tools after {Config.MAX_RETRY_ATTEMPTS} attempts"
                    print(error_message)
                    log_error(error_message)
                    raise Exception(f"Evolution failed: Agent refused to use write_code_file tool after {Config.MAX_RETRY_ATTEMPTS} attempts")
            
            # Validate the changes are meaningful (not just fallback comments)
            if 'Fallback improvement applied' in final_content or 'Agent tool usage failed' in final_content:
                print(f"⚠️  WARNING: Detected fallback artifacts in agent output on attempt {attempt + 1}")
                print(f"   This suggests agent copied fallback content instead of creating new architecture")
                if attempt < Config.MAX_RETRY_ATTEMPTS - 1:
                    print(f"   🔄 Forcing retry {attempt + 2} - agent must create original architecture")
                    repeated_result = None  # Reset for non-repetition retry
                    continue
                    
            message = f"✅ Agent successfully generated new architecture ({len(final_content)} chars)"
            print(message)
            log_info(message)
            
            repeated_result = await check_repeated_motivation(motivation)
            if repeated_result.is_repeated:
                print(f"Attempt {attempt + 1}: Motivation repeated, index is {repeated_result.repeated_index}")
                if attempt == Config.MAX_RETRY_ATTEMPTS - 1:
                    raise Exception("Maximum retry attempts reached, unable to generate non-repeated motivation")
                continue
            else:
                print(f"Attempt {attempt + 1}: Motivation not repeated, continue execution")
                print(f"Generated name: {name}")
                print(f"Generated motivation length: {len(motivation)} characters")
                return name, motivation
                
        except exceptions.MaxTurnsExceeded as e:
            print(f"Attempt {attempt + 1} exceeded maximum dialogue turns")
        except Exception as e:
            print(f"Attempt {attempt + 1} error: {e}")
            raise e

async def check_code_correctness(motivation) -> bool:
    """Check code correctness"""
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        try:
            code_checker_result = await log_agent_run(
                "code_checker",
                code_checker,
                CodeChecker_input(motivation=motivation),
                max_turns=Config.MAX_TURNS_CODE_CHECKER
            )
            
            if code_checker_result.final_output.success:
                print("Code checker passed - code looks correct")
                return True
            else:
                error_msg = code_checker_result.final_output.error
                print(f"Code checker found issues: {error_msg}")
                if attempt == Config.MAX_RETRY_ATTEMPTS - 1:
                    print("Reaching checking limits")
                    return False
                continue
                
        except exceptions.MaxTurnsExceeded as e:
            print("Code checker exceeded maximum turns")
            return False
        except Exception as e:
            print(f"Code checker error: {e}")
            return False

async def check_repeated_motivation(motivation: str):
    client = create_client()
    similar_elements = client.search_similar_motivations(motivation)
    context = similar_motivation_context(similar_elements)
    input = Motivation_checker_input(context, motivation)
    repeated_result = await log_agent_run("motivation_checker", motivation_checker, input, max_turns=Config.MAX_TURNS_MOTIVATION_CHECKER)
    return repeated_result.final_output


def similar_motivation_context(similar_elements: list) -> str:
    """
    Generate structured context from similar motivation elements
    """
    if not similar_elements:
        return "No previous motivations found for comparison."
    
    context = "### PREVIOUS RESEARCH MOTIVATIONS\n\n"
    
    for i, element in enumerate(similar_elements, 1):
        context += f"**Reference #{i} (Index: {element.index})**\n"
        context += f"```\n{element.motivation}\n```\n\n"
    
    context += f"**Total Previous Motivations**: {len(similar_elements)}\n"
    context += "**Analysis Scope**: Compare target motivation against each reference above\n"
    
    return context

def get_repeated_context(repeated_index: list[int]) -> str:
    """
    Generate structured context from repeated motivation experiments
    """
    client = create_client()
    repeated_elements = [client.get_elements_by_index(index) for index in repeated_index]
    
    if not repeated_elements:
        return "No repeated experimental context available."
    
    structured_context = "### REPEATED EXPERIMENTAL PATTERNS ANALYSIS\n\n"
    
    for i, element in enumerate(repeated_elements, 1):
        structured_context += f"**Experiment #{i} - Index {element.index}**\n"
        structured_context += f"```\n{element.motivation}\n```\n\n"
    
    structured_context += f"**Pattern Analysis Summary:**\n"
    structured_context += f"- **Total Repeated Experiments**: {len(repeated_elements)}\n"
    structured_context += f"- **Innovation Challenge**: Break free from these established pattern spaces\n"
    structured_context += f"- **Differentiation Requirement**: Implement orthogonal approaches that explore fundamentally different design principles\n\n"
    
    structured_context += f"**Key Insight**: The above experiments represent exhausted design spaces. Your task is to identify and implement approaches that operate on completely different mathematical, biological, or physical principles to achieve breakthrough innovation.\n"
    
    return structured_context

def apply_fallback_improvements(original_code: str, motivation: str) -> str:
    """DEPRECATED: Fallback system disabled to force proper agent tool usage.
    
    This function previously applied superficial changes when agents failed to use tools.
    It has been disabled to force agents to properly use write_code_file.
    Evolution system now requires agents to actually modify architectures.
    """
    print("❌ FALLBACK SYSTEM DISABLED")
    print("   Evolution agents MUST use write_code_file tool")
    print("   Fallback improvements are no longer applied")
    print("   This failure indicates agent tool usage problems")
    
    # Return original code unchanged - force proper tool usage
    return original_code
