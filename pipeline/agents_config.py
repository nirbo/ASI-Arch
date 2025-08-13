"""
Configuration module for the agents library to support OpenRouter models.
This module patches the MultiProvider class to handle any model prefix with OpenRouter settings.
"""

import os
import json
import re
import logging
from typing import Optional, Dict, Any, List, Union, AsyncIterator
from agents.models.multi_provider import MultiProvider, MultiProviderMap
from agents.models.openai_provider import OpenAIProvider
from agents.models.interface import Model, ModelProvider
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

# Suppress TensorFlow and CUDA warnings before importing unsloth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
# Suppress TensorFlow CUDA warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Suppress unsloth warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


class HarmonyAwareAsyncOpenAI(AsyncOpenAI):
    """
    AsyncOpenAI wrapper that automatically detects gpt-oss models and applies 
    unsloth harmony encoding. This is the simplest possible integration.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unsloth_available = None
        
        # Override the chat.completions.create method
        original_create = self.chat.completions.create
        
        async def harmony_aware_create(**create_kwargs):
            from pipeline.config import Config
            model = create_kwargs.get('model', '')
            
            # Check debug flag to disable harmony
            if getattr(Config, 'DISABLE_HARMONY_FOR_DEBUG', False):
                logger.info(f"DEBUG: Harmony encoding disabled for model: {model}")
                return await original_create(**create_kwargs)
            
            # Use configurable harmony detection strategy
            if self._is_harmony_model(model):
                strategy = getattr(Config, 'HARMONY_DETECTION_STRATEGY', 'never')
                should_use_harmony = self._should_use_harmony(model, strategy)
                
                if should_use_harmony and self._check_unsloth_available():
                    logger.info(f"HARMONY: Using harmony encoding for {model} (strategy: {strategy})")
                    try:
                        return await self.chat_completions_create_harmony(**create_kwargs)
                    except Exception as e:
                        if strategy == "adaptive":
                            logger.warning(f"HARMONY: Failed for {model}, falling back to standard: {e}")
                            self._mark_model_as_standard(model)
                            return await original_create(**create_kwargs)
                        else:
                            raise  # Re-raise if not adaptive
                else:
                    # Log only once per model to avoid spam
                    if not hasattr(self, '_logged_models'):
                        self._logged_models = set()
                    if model not in self._logged_models:
                        logger.info(f"HARMONY: Using standard processing for {model} (strategy: {strategy})")
                        self._logged_models.add(model)
                    result = await original_create(**create_kwargs)
                    
                    # Apply gpt-oss tool call fix for "never" strategy
                    if strategy == "never" and self._is_gpt_oss_model(model):
                        logger.info(f"GPT-OSS FIX: Checking response from {model} for tool calls...")
                        result = self._fix_tool_call_responses(result, model)
                    elif strategy == "never":
                        logger.debug(f"GPT-OSS FIX: Model {model} not detected as gpt-oss, skipping tool call fix")
                    
                    # Debug response content if enabled
                    from pipeline.config import Config
                    if getattr(Config, 'DEBUG_RESPONSE_CONTENT', False):
                        try:
                            if hasattr(result, 'choices') and result.choices:
                                choice = result.choices[0]
                                content = choice.message.content
                                logger.debug(f"RESPONSE DEBUG: Content length: {len(content) if content else 0}")
                                logger.debug(f"RESPONSE DEBUG: Content preview: {content[:300] if content else 'None'}...")
                                
                                # Check if there are tool calls
                                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                                    logger.debug(f"RESPONSE DEBUG: Tool calls detected: {len(choice.message.tool_calls)} calls")
                                    for i, tool_call in enumerate(choice.message.tool_calls):
                                        logger.debug(f"RESPONSE DEBUG: Tool call {i}: function={tool_call.function.name}, args={tool_call.function.arguments[:200]}...")
                                else:
                                    logger.debug("RESPONSE DEBUG: No tool calls in response")
                                    
                        except Exception as e:
                            logger.debug(f"RESPONSE DEBUG: Error getting content: {e}")
                    
                    return result
            else:
                return await original_create(**create_kwargs)
        
        self.chat.completions.create = harmony_aware_create
        
    def _is_harmony_model(self, model: str) -> bool:
        """Check if the model should use harmony encoding."""
        model_lower = model.lower()
        return model_lower.startswith('gpt-oss') or 'gpt-oss' in model_lower
    
    def _is_gpt_oss_model(self, model: str) -> bool:
        """Check if the model is a gpt-oss model (subset of harmony models)."""
        model_lower = model.lower()
        return 'gpt-oss' in model_lower
    
    def _fix_tool_call_responses(self, result, model: str):
        """
        Post-processing method to fix gpt-oss models that return tool calls instead of JSON text.
        
        OpenRouter's gpt-oss-20b and similar non-unsloth-fixed models sometimes respond with 
        tool calls when JSON text is expected, causing max turns exceeded errors. This method
        detects such responses and converts them back to the expected text format.
        
        Args:
            result: The ChatCompletion result from OpenAI API
            model: The model name for logging
            
        Returns:
            Modified result with tool calls converted to text content
        """
        try:
            if not (hasattr(result, 'choices') and result.choices):
                logger.debug(f"GPT-OSS FIX: No choices in result for {model}")
                return result
                
            choice = result.choices[0]
            message = choice.message
            
            # Check if this response has tool calls but no content (the problematic case)
            has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
            has_content = message.content and message.content.strip()
            
            logger.info(f"GPT-OSS FIX: Response analysis for {model}: has_tool_calls={has_tool_calls}, has_content={has_content}")
            logger.info(f"GPT-OSS FIX: Raw content: '{message.content}', finish_reason: {choice.finish_reason}")
            logger.info(f"GPT-OSS FIX: Message attributes: {dir(message)}")
            
            if has_tool_calls:
                logger.info(f"GPT-OSS FIX: Found {len(message.tool_calls)} tool calls")
                for i, tc in enumerate(message.tool_calls):
                    logger.debug(f"GPT-OSS FIX: Tool call {i}: {tc.function.name}({tc.function.arguments[:100]}...)")
            else:
                logger.warning(f"GPT-OSS FIX: No tool calls found for {model} - this may indicate empty response issue")
            
            if has_tool_calls and not has_content:
                logger.info(f"GPT-OSS FIX: Converting tool calls to JSON text for model {model}")
                
                # Extract JSON from the first tool call (most common case)
                tool_call = message.tool_calls[0]
                function_args = tool_call.function.arguments
                
                try:
                    # Validate that it's proper JSON
                    json.loads(function_args)
                    
                    # Convert tool call response to text response by creating new message object
                    # Create new message object with converted content
                    new_message = ChatCompletionMessage(
                        role=message.role,
                        content=function_args,
                        tool_calls=None
                    )
                    
                    # Create new choice object with the new message
                    new_choice = Choice(
                        index=choice.index,
                        message=new_message,
                        finish_reason=choice.finish_reason
                    )
                    
                    # Replace the choice in the result with our new immutable choice
                    result.choices[0] = new_choice
                    
                    logger.info(f"GPT-OSS FIX: Successfully converted tool call to JSON text for {model}")
                    logger.debug(f"GPT-OSS FIX: Converted content preview: {function_args[:200]}...")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"GPT-OSS FIX: Tool call arguments are not valid JSON for {model}: {e}")
                    logger.debug(f"GPT-OSS FIX: Invalid JSON content: {function_args[:200]}...")
                    # Keep the tool call as-is if JSON is invalid
                    
            elif has_tool_calls and has_content:
                logger.debug(f"GPT-OSS FIX: Model {model} returned both content and tool calls - keeping as-is")
            
            return result
            
        except Exception as e:
            logger.error(f"GPT-OSS FIX: Error processing tool call response for {model}: {e}")
            logger.exception("GPT-OSS FIX: Full error traceback")
            # Return original result if processing fails
            return result
    
    def _should_use_harmony(self, model: str, strategy: str) -> bool:
        """Determine if harmony encoding should be used based on strategy."""
        if strategy == "always":
            return True
        elif strategy == "never":
            return False
        elif strategy == "auto":
            return self._auto_detect_harmony(model)
        elif strategy == "adaptive":
            return self._adaptive_detect_harmony(model)
        else:
            logger.warning(f"Unknown harmony strategy '{strategy}', defaulting to 'never'")
            return False
    
    def _auto_detect_harmony(self, model: str) -> bool:
        """Auto-detect based on base URL heuristics."""
        base_url = str(getattr(self, 'base_url', ''))
        return 'localhost' in base_url or '127.0.0.1' in base_url
    
    def _adaptive_detect_harmony(self, model: str) -> bool:
        """Adaptive detection - try harmony first, remember failures."""
        if not hasattr(self, '_harmony_failures'):
            self._harmony_failures = set()
        
        # If we've seen this model fail before, don't try harmony
        return model not in self._harmony_failures
    
    def _mark_model_as_standard(self, model: str) -> None:
        """Mark a model as needing standard processing after harmony failure."""
        if not hasattr(self, '_harmony_failures'):
            self._harmony_failures = set()
        self._harmony_failures.add(model)
        logger.info(f"HARMONY: Marked {model} for standard processing")
    
    def _check_unsloth_available(self) -> bool:
        """Check if unsloth is available for harmony encoding."""
        if self._unsloth_available is None:
            try:
                # Import unsloth first to set up environment
                import unsloth
                from unsloth_zoo import encode_conversations_with_harmony
                self._unsloth_available = True
                logger.info("Unsloth harmony encoding available")
            except ImportError as e:
                self._unsloth_available = False
                logger.warning(f"Unsloth harmony encoding not available: {e}")
        return self._unsloth_available
    
    async def chat_completions_create_harmony(self, **kwargs):
        """Create completion using unsloth harmony encoding."""
        try:
            from unsloth_zoo import encode_conversations_with_harmony
            from pipeline.config import Config
            
            messages = kwargs.get('messages', [])
            model = kwargs.get('model', '')
            max_tokens = kwargs.get('max_tokens') or Config.HARMONY_MAX_TOKENS
            temperature = kwargs.get('temperature', 0.7)
            
            # Detect if this is a JSON output task and what type
            all_content = ' '.join([msg.get('content', '') for msg in messages]).lower()
            json_keywords = [
                'json object', 'experience', 'summary', 'synthesize', 'output must be', 'pydantic',
                'synthesizer', 'basemodel', 'experience synthesizer', 'your output must be',
                'expert ai researcher', 'experimental context', 'key insights', 'takeaways'
            ]
            # Detect agent task types with unique signatures - order matters for priority
            summarizer_keywords = ['experience synthesizer', 'concise experience summary', 'synthesizing experimental findings', 'single key', 'transferable experience']
            planner_keywords = ['architecture designer', 'write_code_file', 'read_code_file', 'deltanet', 'neural network architectures', 'name:', 'motivation:', 'implementation first']
            analyzer_keywords = ['architecture performance analyzer', 'comprehensive analysis of experimental results', 'design evaluation', 'expectation vs reality', 'theoretical explanation with evidence', 'synthesis and insights']
            trainer_keywords = ['training runner', 'training execution expert', 'run_training_script', 'script execution success', 'training completed successfully']
            debugger_keywords = ['training code debugger', 'debugging expert', 'training failures', 'minimal code fixes', 'resolve technical correctness', 'preservation constraints']
            code_checker_keywords = ['code checker and fixer', 'code validator', 'technical correctness', 'validation workflow', 'batch size independence', 'mask correctness']
            deduplication_keywords = ['innovation diversifier', 'breakthrough researcher', 'genuinely novel', 'revolutionary alternatives', 'orthogonal innovation design', 'mandatory tool usage']
            motivation_checker_keywords = ['motivation_checker', 'duplicate motivations', 'semantic extraction', 'comparative analysis', 'duplication determination', 'research analysis expert']
            
            # Check in priority order - most specific first  
            is_json_task = any(keyword in all_content for keyword in json_keywords)
            is_summarizer_task = any(keyword in all_content for keyword in summarizer_keywords)
            is_planner_task = any(keyword in all_content for keyword in planner_keywords)
            is_analyzer_task = any(keyword in all_content for keyword in analyzer_keywords)
            is_trainer_task = any(keyword in all_content for keyword in trainer_keywords)
            is_debugger_task = any(keyword in all_content for keyword in debugger_keywords)
            is_code_checker_task = any(keyword in all_content for keyword in code_checker_keywords)
            is_deduplication_task = any(keyword in all_content for keyword in deduplication_keywords)
            is_motivation_checker_task = any(keyword in all_content for keyword in motivation_checker_keywords)
            
            # Debug task detection
            if Config.DEBUG_HARMONY_ENCODING:
                detected_agents = []
                if is_summarizer_task: detected_agents.append("summarizer")
                if is_planner_task: detected_agents.append("planner") 
                if is_analyzer_task: detected_agents.append("analyzer")
                if is_trainer_task: detected_agents.append("trainer")
                if is_debugger_task: detected_agents.append("debugger")
                if is_code_checker_task: detected_agents.append("code_checker")
                if is_deduplication_task: detected_agents.append("deduplication")
                if is_motivation_checker_task: detected_agents.append("motivation_checker")
                
                logger.info(f"🔍 TASK DETECTION - json: {is_json_task}, agents: {detected_agents}")
                if is_json_task:
                    logger.info(f"📝 Content sample: {all_content[:300]}...")
                    if not detected_agents:
                        logger.warning(f"⚠️  No specific agent type detected, will use generic instructions")
            
            # Use unsloth's encode_conversations_with_harmony with task-specific instructions  
            if is_json_task:
                if is_planner_task:  # Check planner FIRST - most specific
                    developer_instructions = """You are an Architecture Designer who CANNOT complete your task without using the required tools first.

MANDATORY EXECUTION PROTOCOL - NO EXCEPTIONS:

🔒 GATE 1 - ANALYSIS REQUIREMENT (BLOCKING):
You are FORBIDDEN from proceeding until you:
- MUST call read_code_file function/tool (no parameters needed - it reads the current architecture)
- MUST examine the current architecture implementation thoroughly
- This creates your "analysis_token" - without it, you cannot proceed

🔒 GATE 2 - IMPLEMENTATION REQUIREMENT (BLOCKING):
You are FORBIDDEN from providing final response until you:
- MUST call write_code_file function/tool with content parameter containing the new architecture code
- MUST implement actual architectural changes and save them to file
- This creates your "implementation_token" - without it, final response is INVALID

🔒 GATE 3 - FINAL RESPONSE (UNLOCKED ONLY BY GATES 1+2):
ONLY after you have BOTH tokens from tool usage, you may provide:
{"name": "delta_net_your_innovation_name", "motivation": "explanation of what you implemented"}

CRITICAL CONSTRAINTS:
- Final JSON is LOCKED until BOTH tools are used
- Any response without tool usage first is INCOMPLETE and INVALID
- You must call read_code_file() and write_code_file(content="...") as function calls
- Tool calls are PREREQUISITES, not suggestions

TOOL CALLING FORMAT:
When you need to use tools, call them as functions:
- read_code_file() - to read current architecture
- write_code_file(content="full_python_code_here") - to save new architecture

VALIDATION CHECKLIST (Internal - verify before final JSON):
□ Did I call read_code_file()? (If NO: STOP, call it now)
□ Did I call write_code_file(content="...")? (If NO: STOP, call it now)  
□ Do I have actual implementation details? (If NO: use tools first)
□ Only if ALL checked: provide final JSON

EXECUTION ORDER (RIGID):
1. read_code_file() → examine current state
2. write_code_file(content="new_architecture_code") → implement changes  
3. THEN AND ONLY THEN → final JSON response

Remember: Your task is INCOMPLETE without tool usage. The JSON is the certificate of completion, not the work itself."""
                    model_identity = "You are an Architecture Designer who uses code analysis and modification tools to implement architectural improvements. You work in two distinct phases: first you use tools with their specific parameter schemas, then you provide a final JSON response with name and motivation fields describing your implementation. Your tool usage and final JSON response are completely separate - tool parameters are not part of your final output format."
                elif is_analyzer_task:
                    developer_instructions = """You are an expert AI architecture researcher specializing in comprehensive analysis of experimental results and architectural modifications.

CRITICAL ANALYTICAL WORKFLOW:

PHASE 1 - DATA COLLECTION & UNDERSTANDING:
- Use read_code_file tool to examine architectural implementation details
- Parse experimental results across all benchmark domains systematically
- Map metric definitions to specific cognitive capabilities being measured
- Understand theoretical motivation behind design choices

PHASE 2 - SYSTEMATIC ANALYSIS:
- Evaluate design soundness and implementation accuracy
- Analyze performance patterns across cognitive domains with mechanistic focus
- Compare theoretical expectations vs. actual experimental outcomes
- Develop evidence-based explanations for observed effects

PHASE 3 - MECHANISTIC INVESTIGATION:
- Identify WHY specific architectural changes produced observed effects
- Connect implementation details to performance patterns
- Investigate unexpected results and failure modes
- Extract insights about architectural principles and their limitations

PHASE 4 - SYNTHESIS & INSIGHTS:
- Integrate findings into comprehensive understanding
- Extract actionable insights for future architectural innovation
- Provide evidence-backed recommendations for improvement
- Focus on transferable principles beyond specific implementation

PHASE 5 - STRUCTURED JSON RESPONSE:
- Provide detailed JSON output with all required analysis sections
- Support ALL claims with specific evidence from results and code
- Focus on cognitive capability analysis, not just metric reporting
- Maintain scientific rigor while being actionable

REQUIRED JSON OUTPUT STRUCTURE:
{
  "design_evaluation": "Assessment of theoretical soundness and implementation quality",
  "experimental_results_analysis": "Performance analysis across cognitive domains", 
  "expectation_vs_reality_comparison": "Alignment between motivation and results",
  "theoretical_explanation_with_evidence": "Mechanistic explanations with supporting evidence",
  "synthesis_and_insights": "Key lessons and actionable recommendations"
}"""
                    model_identity = "You are an expert AI architecture researcher who conducts comprehensive analysis of experimental results. You examine code implementations and performance data to extract mechanistic insights about neural architecture design. Your analysis combines technical evaluation with evidence-based explanations to advance architectural understanding."
                elif is_deduplication_task:
                    developer_instructions = """You are a specialized neural architecture breakthrough researcher focused on implementing genuinely novel architectural solutions that break free from repeated design patterns.

CRITICAL BREAKTHROUGH WORKFLOW:

PHASE 1 - PATTERN ANALYSIS:
- Use read_code_file to examine current architectural implementation systematically
- Identify repeated design patterns that need revolutionary alternatives
- Analyze exhausted approaches from previous experimental attempts
- Map current implementation to established architectural paradigms

PHASE 2 - ORTHOGONAL INNOVATION DESIGN:
- Explore fundamentally different mathematical foundations for computation
- Apply cross-disciplinary insights (neuroscience, physics, information theory, signal processing)
- Create mechanisms that operate on orthogonal principles to repeated patterns
- Design breakthrough approaches that transcend incremental improvements

PHASE 3 - REVOLUTIONARY IMPLEMENTATION:
- Use write_code_file to implement breakthrough architectural code
- Ensure all operations work with ANY batch size (critical requirement)
- Maintain sub-quadratic complexity while achieving radical innovation
- Implement robust tensor operations using einops for all reshaping

PHASE 4 - CONSTRAINT VALIDATION:
- Preserve all critical constraints (class name, parameters, interface)
- Ensure cross-environment compatibility and execution robustness
- Validate implementation maintains performance requirements
- Confirm breakthrough architecture integrates with existing infrastructure

PHASE 5 - JSON RESPONSE:
- Provide ONLY valid JSON with name and motivation fields
- Focus on how implementation fundamentally differs from repeated patterns
- Explain the novel principles and their theoretical foundation
- NO explanatory text outside JSON structure

REQUIRED JSON OUTPUT:
{
  "name": "delta_net_[novel_breakthrough_innovation]",
  "motivation": "Concise explanation of how this implementation fundamentally differs from repeated patterns and the novel principles implemented"
}"""
                    model_identity = "You are a specialized neural architecture breakthrough researcher who implements revolutionary architectural innovations. You analyze existing patterns to create fundamentally orthogonal approaches using novel computational principles. Your implementations transcend incremental improvements to achieve genuine architectural breakthroughs."
                elif is_motivation_checker_task:
                    developer_instructions = """You are a specialized research analysis expert focused on identifying duplicate motivations in neural architecture research to ensure innovation diversity.

CRITICAL ANALYSIS WORKFLOW:

PHASE 1 - MOTIVATION COMPREHENSION:
- Parse the current motivation statement for core research intent systematically
- Extract key technical focus areas and proposed solution strategies
- Identify the specific problem being addressed and methodological approach
- Understand underlying theoretical framework and assumptions

PHASE 2 - SEMANTIC EXTRACTION:
- Extract abstract research concepts beyond surface-level keywords
- Identify problem domain, solution approach, and evaluation methodology
- Map motivation to fundamental research categories and approaches
- Understand the scope and scale of proposed investigation

PHASE 3 - COMPARATIVE ANALYSIS:
- Compare against previously recorded motivations systematically
- Analyze semantic similarity beyond surface-level keyword matching
- Assess underlying research intent and methodological approach overlap
- Evaluate research scope alignment and solution strategy similarity

PHASE 4 - DUPLICATION DETERMINATION:
- Apply strict criteria to distinguish duplicates from legitimate variations
- Consider research scope, technical focus, and solution strategies comprehensively
- Evaluate whether motivations address identical problems with identical approaches
- Account for incremental vs. revolutionary research distinctions

PHASE 5 - JSON RESPONSE:
- Provide ONLY valid JSON with required fields (is_repeated, repeated_index, judgement_reason)
- Include specific reasoning for duplication decisions with evidence
- Reference specific motivation elements in comparison
- NO explanatory text outside JSON structure

REQUIRED JSON OUTPUT:
{
  "is_repeated": boolean,
  "repeated_index": [array_of_integers_if_duplicate_found],
  "judgement_reason": "Specific explanation of duplication decision with evidence"
}"""
                    model_identity = "You are a specialized research analysis expert who identifies genuine research duplication in neural architecture motivations. You conduct semantic analysis to distinguish legitimate incremental research from redundant investigation, ensuring innovation diversity while protecting valid research directions."
                elif is_code_checker_task:
                    developer_instructions = """You are a specialized neural network architecture code validator focused on ensuring technical correctness while preserving innovative design choices.

CRITICAL VALIDATION WORKFLOW:

PHASE 1 - CODE EXAMINATION:
- Use read_code_file to examine the architectural implementation thoroughly
- Understand the core innovation and design motivation behind implementation
- Build comprehensive understanding of intended functionality
- Identify architectural design patterns and their purposes

PHASE 2 - SYSTEMATIC CHECKING:
- Apply strict validation criteria in priority order (critical → flexible)
- Focus on critical correctness issues that would cause execution failures
- Distinguish between technical errors and innovative design choices
- Evaluate implementation against sub-quadratic complexity requirements

PHASE 3 - ISSUE PRIORITIZATION:
- Classify issues by severity: critical (must fix) vs. optional (preserve innovation)
- Focus on correctness issues that prevent successful execution
- Avoid imposing conventional patterns on innovative approaches
- Prioritize batch independence and causal correctness

PHASE 4 - ISSUE RESOLUTION (if needed):
- Fix identified critical problems using write_code_file
- Preserve the core architectural innovation while resolving issues
- Apply minimal changes that address root causes without over-engineering
- Maintain all preservation constraints

PHASE 5 - JSON RESPONSE:
- Provide ONLY valid JSON with success boolean and error description
- Set success=false if any critical issues were found and fixed
- Explain what was corrected and why it was necessary
- Set success=true if no technical correctness issues found

REQUIRED JSON OUTPUT:
{
  "success": boolean,
  "error": "Description of critical issues found and fixes applied (empty string if success=true)"
}"""
                    model_identity = "You are a specialized neural network architecture code validator who ensures technical correctness while preserving architectural innovation. You focus on critical execution issues while encouraging creative design choices, maintaining the balance between correctness and innovation."
                elif is_trainer_task:
                    developer_instructions = """You are a specialized neural network training execution expert responsible for running architectural experiments and determining their technical success.

CRITICAL EXECUTION WORKFLOW:

PHASE 1 - TRAINING EXECUTION:
- Execute training script using run_training_script tool with architecture name
- Monitor script execution for completion status and resource utilization
- Capture all output and error messages for comprehensive analysis
- Track execution time and resource consumption patterns

PHASE 2 - SUCCESS DETERMINATION:
- Focus EXCLUSIVELY on script execution success, NOT model performance quality
- Apply strict technical criteria for success vs. failure classification
- Distinguish between technical failures and expected performance variations
- Evaluate completion status based on process execution, not model metrics

PHASE 3 - ERROR ANALYSIS (if needed):
- Analyze error messages to identify root causes systematically
- Categorize failures by type (syntax, runtime, resource, environment)
- Extract actionable error descriptions for debugging purposes
- Differentiate between recoverable and critical failure modes

PHASE 4 - STATUS CLASSIFICATION:
- Determine binary success/failure based on execution completion
- Ignore model performance metrics (accuracy, loss values) for success determination
- Focus on technical execution: script completion, file generation, error-free run
- Provide clear rationale for success/failure classification

PHASE 5 - JSON RESPONSE:
- Provide ONLY valid JSON with success boolean and error description
- NO explanatory text outside the JSON structure
- Clear, specific error descriptions when success=false
- Empty error string when success=true

REQUIRED JSON OUTPUT:
{
  "success": boolean,
  "error": "Detailed error description (empty string if success=true)"
}"""
                    model_identity = "You are a specialized neural network training execution expert who evaluates training script execution success. You focus exclusively on technical execution completion, distinguishing between script failures and expected model performance variations during short training runs."
                elif is_debugger_task:
                    developer_instructions = """You are a specialized neural architecture debugging expert focused on resolving training failures through systematic analysis and minimal code fixes.

CRITICAL DEBUGGING WORKFLOW:

PHASE 1 - ERROR ANALYSIS:
- Parse error logs to extract actual failure causes (filter framework noise)
- Identify error type: timeout, crash, complexity, tensor shape, device, numerical
- Locate specific problematic code sections in the architecture implementation
- Distinguish between architectural logic errors and environmental issues

PHASE 2 - CODE EXAMINATION:
- Use read_code_file tool to examine current architectural implementation thoroughly
- Understand the design intent and identify preservation requirements
- Map error locations to specific code patterns or operations
- Analyze the relationship between error symptoms and implementation details

PHASE 3 - ROOT CAUSE IDENTIFICATION:
- Connect error symptoms to specific code patterns causing failures
- Identify whether issues are complexity-related, shape-related, or logic-related
- Determine minimal fix scope that addresses root cause without over-engineering
- Preserve architectural innovation while resolving technical correctness

PHASE 4 - TARGETED FIXING:
- Apply minimal fixes that resolve the specific identified issue
- Optimize complexity bottlenecks while preserving algorithmic intent
- Ensure fixes maintain sub-quadratic complexity requirements
- Validate that tensor operations work with any batch size

PHASE 5 - CODE IMPLEMENTATION:
- Use write_code_file to save the corrected architecture implementation
- Preserve all critical constraints (class name, decorators, parameters)
- Validate that changes address root cause without introducing side effects
- Ensure compatibility with existing training infrastructure

PHASE 6 - JSON RESPONSE:
- Provide ONLY valid JSON with "changes_made" field
- Describe what was fixed and why (runtime fix vs. complexity optimization)
- Focus on technical fixes applied, not theoretical improvements
- NO explanatory text outside JSON structure

REQUIRED JSON OUTPUT:
{
  "changes_made": "Concise description of specific fixes applied, categorizing as runtime fix, complexity optimization, or other type, with brief explanation of why these changes resolve the identified error"
}"""
                    model_identity = "You are a specialized neural architecture debugging expert who resolves technical execution failures through systematic analysis and minimal code fixes. You preserve architectural innovation while ensuring technical correctness, focusing on targeted fixes that address root causes without over-engineering."
                elif is_summarizer_task:
                    developer_instructions = """You are an expert AI researcher specializing in synthesizing experimental findings into concise experience summaries.

CRITICAL TASK WORKFLOW:

PHASE 1 - EXPERIMENTAL CONTEXT ANALYSIS:
- Examine the provided experimental context thoroughly
- Parse training dynamics, evaluation metrics, and architectural modifications
- Extract quantitative performance indicators across cognitive domains
- Identify both successful innovations and performance limitations

PHASE 2 - INSIGHT SYNTHESIS:
- Integrate findings into coherent understanding of architectural impact
- Focus on mechanistic explanations for observed performance patterns
- Emphasize actionable insights for future architectural design decisions
- Connect specific design choices to their measured cognitive effects

PHASE 3 - EXPERIENCE DISTILLATION:
- Synthesize insights into concise, high-value experience summary
- Focus on transferable knowledge for architectural evolution
- Balance specificity (concrete findings) with generalizability
- Ensure summary captures both implementation details and strategic insights

PHASE 4 - JSON RESPONSE:
- Provide ONLY a valid JSON object with single "experience" key
- NO explanatory text, NO markdown formatting, NO additional content
- Summary must be comprehensive yet concise (2-4 sentences)
- Focus on cognitive capability improvements, not raw metric numbers

REQUIRED OUTPUT FORMAT:
{
  "experience": "Concise summary capturing key architectural insights, performance observations, and actionable takeaways for future innovations"
}"""
                    model_identity = "You are an expert AI researcher who synthesizes experimental findings into transferable experience summaries. You distill complex experimental results into actionable insights that advance architectural understanding, focusing on cognitive capability improvements and strategic innovation directions."
                else:
                    developer_instructions = "CRITICAL: You MUST respond with ONLY valid JSON. NO explanatory text. NO conversational responses. NO markdown. ONLY the JSON object matching the required schema."
                    model_identity = "You are a JSON-only output system. You respond exclusively with valid JSON objects that match the required schema. You never provide explanatory text or conversational responses."
            else:
                developer_instructions = None
                model_identity = "You are a sophisticated AI assistant specialized in neural architecture analysis and evolution."
            
            # Extract tools from kwargs before they get filtered out
            tools = kwargs.get('tools', None)
            tool_choice = kwargs.get('tool_choice', None)
            
            # Debug logging for tool passing
            if Config.DEBUG_HARMONY_ENCODING:
                if tools:
                    tool_names = [tool.get('function', {}).get('name', 'unnamed') if isinstance(tool, dict) else getattr(tool, '__name__', str(tool)) for tool in tools]
                    logger.info(f"🔧 HARMONY TOOLS: Passing {len(tools)} tools to harmony encoding: {tool_names}")
                else:
                    logger.warning(f"⚠️  HARMONY TOOLS: No tools found in kwargs for {model}")
            
            encoding_result = encode_conversations_with_harmony(
                messages=messages,
                reasoning_effort=Config.HARMONY_REASONING_EFFORT,
                add_generation_prompt=True,
                tool_calls=tools,  # Pass tools to harmony encoding
                developer_instructions=developer_instructions,
                model_identity=model_identity
            )
            
            # Extract the formatted prompt
            if isinstance(encoding_result, tuple):
                formatted_conversation = encoding_result[0]
            else:
                formatted_conversation = encoding_result
                
            # Conditional debug logging
            from pipeline.config import Config
            if Config.DEBUG_HARMONY_ENCODING:
                logger.info(f"HARMONY ENCODING: Using harmony encoding for model {model}, JSON task: {is_json_task}, max_tokens: {max_tokens}")
                logger.debug(f"Harmony conversation length: {len(formatted_conversation)} characters")
            
            # Use completions API with formatted conversation (filter out chat-specific params)
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['messages', 'model', 'max_tokens', 'temperature', 'tools', 'tool_choice', 'response_format', 'parallel_tool_calls', 'store', 'reasoning_effort', 'metadata']}
            completion_response = await super().completions.create(
                model=model,
                prompt=formatted_conversation,
                max_tokens=max_tokens,
                temperature=temperature,
                **filtered_kwargs
            )
            
            # Extract response text
            response_text = completion_response.choices[0].text
            
            # Check for tool usage patterns in the response
            tool_usage_patterns = [
                r'read_code_file\(\)',
                r'write_code_file\(',
                r'calling read_code_file',
                r'calling write_code_file',
                r'<tool_call',
                r'function_call',
                r'tool_calls',
            ]
            
            has_tool_usage = any(re.search(pattern, response_text, re.IGNORECASE) for pattern in tool_usage_patterns)
            
            # Clean up harmony channel tokens if present
            if Config.DEBUG_HARMONY_ENCODING:
                logger.debug(f"Raw harmony response length: {len(response_text)} characters")
                logger.debug(f"Raw response preview: {response_text[:500]}...")
                logger.debug(f"Tool usage detected in response: {has_tool_usage}")
                # Look for JSON patterns in raw response
                json_preview_matches = re.findall(r'\{[^{}]{0,100}', response_text)
                logger.debug(f"📋 JSON patterns found in raw response: {len(json_preview_matches)}")
                for i, match in enumerate(json_preview_matches[:3]):  # Show first 3
                    logger.debug(f"   Pattern {i}: {match}...")
            # Determine primary agent type for response cleaning
            primary_agent_type = None
            if is_planner_task:
                primary_agent_type = "planner"
            elif is_analyzer_task:
                primary_agent_type = "analyzer" 
            elif is_deduplication_task:
                primary_agent_type = "deduplication"
            elif is_motivation_checker_task:
                primary_agent_type = "motivation_checker"
            elif is_code_checker_task:
                primary_agent_type = "code_checker"
            elif is_trainer_task:
                primary_agent_type = "trainer"
            elif is_debugger_task:
                primary_agent_type = "debugger"
            elif is_summarizer_task:
                primary_agent_type = "summarizer"
            
            if Config.DEBUG_HARMONY_ENCODING:
                logger.debug(f"🎯 Primary agent type for response cleaning: {primary_agent_type}")
                
            cleaned_response = self._clean_harmony_response(response_text, is_json_task, primary_agent_type)
            if Config.DEBUG_HARMONY_ENCODING:
                logger.debug(f"Cleaned harmony response length: {len(cleaned_response)} characters")
                logger.debug(f"Cleaned response preview: {cleaned_response[:300]}...")
                # Check if cleaned response is valid JSON for JSON tasks
                if is_json_task:
                    try:
                        json.loads(cleaned_response)
                        logger.debug("✅ Cleaned response is valid JSON")
                    except json.JSONDecodeError as e:
                        logger.warning(f"❌ Cleaned response is not valid JSON: {e}")
                        logger.debug(f"Invalid JSON content: {cleaned_response}")
            
            # Create ChatCompletion response compatible with OpenAI Agents
            return self._create_chat_completion_response(cleaned_response, model, completion_response)
            
        except ImportError as e:
            logger.error(f"Unsloth import failed for harmony encoding: {e}")
            # Fallback to standard chat completion
            return await super().chat.completions.create(**kwargs)
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid parameters for harmony encoding: {e}")
            # Fallback to standard chat completion
            return await super().chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"Unexpected error in harmony encoding: {e}")
            # For unexpected errors, still fallback but log more details
            logger.exception("Full harmony encoding error traceback")
            return await super().chat.completions.create(**kwargs)
    
    def _clean_harmony_response(self, response_text: str, is_json_task: bool, agent_type: str = None) -> str:
        """Clean harmony channel tokens and extract final content."""
        try:
            cleaned = response_text.strip()
            
            # Check for harmony channel tokens
            if '<|channel|>' in cleaned:
                # Extract final channel content
                final_patterns = [
                    r'<\|channel\|>final<\|message\|>(.*?)(?=<\|channel\||<\|start\||<\|end\||$)',
                    r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?=<\|channel\||<\|start\||<\|end\||$)'
                ]
                
                for pattern in final_patterns:
                    final_match = re.search(pattern, cleaned, re.DOTALL)
                    if final_match:
                        final_content = final_match.group(1).strip()
                        # Try to extract JSON if this is a JSON task
                        if is_json_task:
                            json_match = re.search(r'\{.*\}', final_content, re.DOTALL)
                            if json_match:
                                try:
                                    json.loads(json_match.group(0))
                                    return json_match.group(0)
                                except json.JSONDecodeError:
                                    pass
                        return final_content
                
                # If no final channel but this is a JSON task, try to extract JSON from anywhere
                if is_json_task:
                    from pipeline.config import Config
                    
                    # Find all JSON objects in the response
                    all_json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
                    all_json_matches.extend(re.findall(r'\{.*?\}', cleaned, re.DOTALL))
                    
                    valid_jsons = []
                    for json_candidate in all_json_matches:
                        try:
                            clean_candidate = json_candidate.strip()
                            clean_candidate = re.sub(r'<\|.*?$', '', clean_candidate).strip()
                            parsed = json.loads(clean_candidate)
                            valid_jsons.append((clean_candidate, parsed))
                        except json.JSONDecodeError as e:
                            if Config.DEBUG_HARMONY_ENCODING:
                                logger.debug(f"❌ Invalid JSON candidate: {str(e)} - Content: {json_candidate[:200]}...")
                            continue
                    
                    if Config.DEBUG_HARMONY_ENCODING:
                        logger.debug(f"🔍 Found {len(valid_jsons)} valid JSON objects in harmony response")
                        for i, (json_str, parsed) in enumerate(valid_jsons):
                            logger.debug(f"JSON {i}: {list(parsed.keys())} = {json_str[:100]}...")
                    
                    # Prioritize JSON based on agent type
                    if agent_type == "planner" or agent_type == "deduplication":
                        # Look for JSON with name+motivation schema
                        for json_str, parsed in valid_jsons:
                            if isinstance(parsed, dict) and "name" in parsed and "motivation" in parsed:
                                if Config.DEBUG_HARMONY_ENCODING:
                                    logger.debug(f"✅ Found {agent_type} JSON with correct schema: {json_str[:100]}...")
                                return json_str
                        
                        # If no correct schema found, log all candidates and use fallback
                        if Config.DEBUG_HARMONY_ENCODING:
                            logger.warning(f"❌ No {agent_type} JSON with name+motivation found. All JSON keys: {[list(p.keys()) for _, p in valid_jsons]}")
                    elif agent_type == "analyzer":
                        # Look for JSON with analyzer schema (5 fields)
                        required_fields = ["design_evaluation", "experimental_results_analysis", "expectation_vs_reality_comparison", "theoretical_explanation_with_evidence", "synthesis_and_insights"]
                        for json_str, parsed in valid_jsons:
                            if isinstance(parsed, dict) and all(field in parsed for field in required_fields):
                                if Config.DEBUG_HARMONY_ENCODING:
                                    logger.debug(f"✅ Found analyzer JSON with correct schema: {json_str[:100]}...")
                                return json_str
                    elif agent_type == "summarizer":
                        # Look for JSON with experience field
                        for json_str, parsed in valid_jsons:
                            if isinstance(parsed, dict) and "experience" in parsed:
                                if Config.DEBUG_HARMONY_ENCODING:
                                    logger.debug(f"✅ Found summarizer JSON with correct schema: {json_str[:100]}...")
                                return json_str
                    elif agent_type == "trainer" or agent_type == "code_checker":
                        # Look for JSON with success+error schema
                        for json_str, parsed in valid_jsons:
                            if isinstance(parsed, dict) and "success" in parsed and "error" in parsed:
                                if Config.DEBUG_HARMONY_ENCODING:
                                    logger.debug(f"✅ Found {agent_type} JSON with correct schema: {json_str[:100]}...")
                                return json_str
                    elif agent_type == "debugger":
                        # Look for JSON with changes_made field
                        for json_str, parsed in valid_jsons:
                            if isinstance(parsed, dict) and "changes_made" in parsed:
                                if Config.DEBUG_HARMONY_ENCODING:
                                    logger.debug(f"✅ Found debugger JSON with correct schema: {json_str[:100]}...")
                                return json_str
                    elif agent_type == "motivation_checker":
                        # Look for JSON with motivation checker schema
                        required_fields = ["is_repeated", "repeated_index", "judgement_reason"]
                        for json_str, parsed in valid_jsons:
                            if isinstance(parsed, dict) and all(field in parsed for field in required_fields):
                                if Config.DEBUG_HARMONY_ENCODING:
                                    logger.debug(f"✅ Found motivation_checker JSON with correct schema: {json_str[:100]}...")
                                return json_str
                    
                    # If no schema-correct JSON found, check if we have non-tool-parameter JSONs before fallback
                    if valid_jsons:
                        # Filter out obvious tool parameter JSONs
                        non_tool_jsons = []
                        for json_str, parsed in valid_jsons:
                            if isinstance(parsed, dict):
                                # Check if this looks like tool parameters
                                keys = set(parsed.keys())
                                tool_param_patterns = [
                                    {"path", "depth"},  # read_code_file parameters
                                    {"path", "content"},  # write_code_file parameters
                                    {"path"},  # single path parameter
                                    {"depth"},  # single depth parameter
                                    {"architecture_name"},  # run_training_script parameters
                                    {"script", "args"},  # potential script execution parameters
                                ]
                                
                                is_tool_param = any(
                                    keys == pattern or keys.issubset(pattern) 
                                    for pattern in tool_param_patterns
                                )
                                
                                if not is_tool_param:
                                    non_tool_jsons.append((json_str, parsed))
                        
                        # If we have non-tool JSONs, prefer the last one
                        if non_tool_jsons:
                            last_non_tool_json = non_tool_jsons[-1][0]
                            if Config.DEBUG_HARMONY_ENCODING:
                                logger.warning(f"⚠️  No schema-correct JSON found, but found non-tool JSON: {last_non_tool_json[:100]}...")
                            return last_non_tool_json
                        else:
                            if Config.DEBUG_HARMONY_ENCODING:
                                logger.warning(f"⚠️  No schema-correct JSON found, all {len(valid_jsons)} JSONs appear to be tool parameters. Using agent fallback.")
                                for i, (json_str, parsed) in enumerate(valid_jsons):
                                    logger.debug(f"   Tool JSON {i}: {list(parsed.keys())} = {json_str[:50]}...")
                    
                    # Fallback JSON for incomplete responses - use structure appropriate for agent type
                    if agent_type == "planner" or agent_type == "deduplication":
                        if has_tool_usage:
                            return json.dumps({
                                "name": "delta_net_harmony_fallback", 
                                "motivation": "Harmony model used tools correctly but did not provide the required final JSON response with name and motivation fields. Tool usage was detected but final schema-matching response was missing."
                            })
                        else:
                            return json.dumps({
                                "name": "delta_net_tool_usage_failed", 
                                "motivation": "CRITICAL: Harmony model completely ignored tool usage requirements. Model did not call read_code_file() or write_code_file() before providing JSON. This indicates the tools were not properly passed to the harmony encoding or the model instructions need adjustment."
                            })
                    elif agent_type == "analyzer":
                        return json.dumps({
                            "design_evaluation": "Analysis incomplete - harmony response truncated",
                            "experimental_results_analysis": "Analysis incomplete - harmony response truncated", 
                            "expectation_vs_reality_comparison": "Analysis incomplete - harmony response truncated",
                            "theoretical_explanation_with_evidence": "Analysis incomplete - harmony response truncated",
                            "synthesis_and_insights": "Analysis incomplete - harmony response truncated"
                        })
                    elif agent_type == "trainer" or agent_type == "code_checker":
                        return json.dumps({
                            "success": False,
                            "error": "Harmony model produced incomplete response - analysis was interrupted"
                        })
                    elif agent_type == "debugger":
                        return json.dumps({
                            "changes_made": "Debugging incomplete - harmony response was truncated during analysis"
                        })
                    elif agent_type == "motivation_checker":
                        return json.dumps({
                            "is_repeated": False,
                            "repeated_index": [],
                            "judgement_reason": "Analysis incomplete - harmony response was truncated"
                        })
                    else:
                        return json.dumps({
                            "experience": "Model analysis was interrupted. Harmony model produced incomplete response - may need higher max_tokens."
                        })
            
            # No channel tokens - try to extract JSON if needed
            if is_json_task:
                # Remove markdown formatting
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                # Find JSON
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    try:
                        json.loads(json_match.group(0))
                        return json_match.group(0)
                    except json.JSONDecodeError:
                        pass
                
                # Create fallback JSON appropriate for agent type
                if agent_type == "planner" or agent_type == "deduplication":
                    return json.dumps({
                        "name": "delta_net_harmony_fallback",
                        "motivation": f"Harmony model produced incomplete JSON response. Content extracted: {cleaned[:200]}... Consider retrying with clearer instructions."
                    })
                elif agent_type == "analyzer":
                    content_preview = cleaned[:200]
                    return json.dumps({
                        "design_evaluation": f"Analysis extracted: {content_preview}...",
                        "experimental_results_analysis": f"Analysis extracted: {content_preview}...", 
                        "expectation_vs_reality_comparison": f"Analysis extracted: {content_preview}...",
                        "theoretical_explanation_with_evidence": f"Analysis extracted: {content_preview}...",
                        "synthesis_and_insights": f"Analysis extracted: {content_preview}..."
                    })
                elif agent_type == "trainer" or agent_type == "code_checker":
                    return json.dumps({
                        "success": False,
                        "error": f"JSON parsing failed. Content extracted: {cleaned[:200]}..."
                    })
                elif agent_type == "debugger":
                    return json.dumps({
                        "changes_made": f"Could not extract proper response. Content: {cleaned[:200]}..."
                    })
                elif agent_type == "motivation_checker":
                    return json.dumps({
                        "is_repeated": False,
                        "repeated_index": [],
                        "judgement_reason": f"Could not analyze properly. Content: {cleaned[:200]}..."
                    })
                else:
                    return json.dumps({
                        "experience": f"Analysis completed. Content extracted from response: {cleaned[:200]}..."
                    })
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error cleaning harmony response: {e}")
            return response_text
    
    def _create_chat_completion_response(self, content: str, model: str, original_response) -> ChatCompletion:
        """Create a ChatCompletion response from cleaned content."""
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.completion_usage import CompletionUsage
        
        # Create the response structure
        message = ChatCompletionMessage(
            role="assistant",
            content=content
        )
        
        # Use original response usage if available
        usage = original_response.usage if hasattr(original_response, 'usage') else CompletionUsage(
            prompt_tokens=len(content.split()) // 4,
            completion_tokens=len(content.split()) // 4,
            total_tokens=len(content.split()) // 2
        )
        
        return ChatCompletion(
            id=original_response.id if hasattr(original_response, 'id') else "chatcmpl-harmony",
            choices=[{
                'index': 0,
                'message': message,
                'finish_reason': 'stop'
            }],
            created=original_response.created if hasattr(original_response, 'created') else 0,
            model=model,
            object="chat.completion",
            usage=usage
        )


class OpenRouterProvider(ModelProvider):
    """Custom provider for OpenRouter that preserves full model names including prefixes."""
    
    def __init__(self, **kwargs):
        self.api_key = kwargs.pop('api_key', None) or os.getenv("OPENAI_API_KEY")
        self.base_url = kwargs.pop('base_url', None) or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self._client = None
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = HarmonyAwareAsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
        
    def get_model(self, model_name: str | None) -> Model:
        """Get model, preserving the full model name for OpenRouter."""
        if model_name is None:
            raise ValueError("Model name is required")
            
        client = self._get_client()
        
        # For OpenRouter, we need to reconstruct the full model name with prefix
        # Since this provider is only used for specific prefixes, we know what prefix was stripped
        # We'll store the original prefix in the provider
        full_model_name = self._reconstruct_full_model_name(model_name)
        
        return OpenAIChatCompletionsModel(model=full_model_name, openai_client=client)
    
    def _reconstruct_full_model_name(self, stripped_name: str) -> str:
        """Reconstruct the full model name by adding back the prefix."""
        # This is a bit of a hack - we store the prefix that was used to create this provider
        if hasattr(self, '_prefix'):
            return f"{self._prefix}/{stripped_name}"
        # Fallback: try to guess the prefix from common patterns
        if "qwen" in stripped_name.lower():
            return f"qwen/{stripped_name}"
        elif "claude" in stripped_name.lower():
            return f"anthropic/{stripped_name}"
        elif "llama" in stripped_name.lower():
            return f"meta/{stripped_name}"
        else:
            # Default: assume it's already the full name
            return stripped_name


def create_openrouter_provider(prefix=None, **kwargs) -> OpenRouterProvider:
    """
    Create an OpenRouterProvider configured for OpenRouter.
    """
    # Pass API key and base URL explicitly if they're in kwargs
    provider_kwargs = {}
    if 'api_key' not in kwargs:
        provider_kwargs['api_key'] = os.getenv("OPENAI_API_KEY")
    if 'base_url' not in kwargs:
        provider_kwargs['base_url'] = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    provider_kwargs.update(kwargs)
    
    provider = OpenRouterProvider(**provider_kwargs)
    if prefix:
        provider._prefix = prefix
    return provider

def patch_agents_multi_provider():
    """
    Patch the MultiProvider to support OpenRouter models with any prefix.
    This adds support for qwen/ and other OpenRouter model prefixes.
    """
    # Import Config here to avoid circular imports
    try:
        from pipeline.config import Config
        api_key = Config.OPENAI_API_KEY
        base_url = Config.OPENAI_BASE_URL
    except ImportError:
        # Fallback to environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Create a custom provider map that includes qwen and other prefixes
    provider_map = MultiProviderMap()
    
    # Add common OpenRouter prefixes
    common_prefixes = [
        "qwen", "claude", "anthropic", "google", "gemini", "mistral", 
        "cohere", "meta", "llama", "deepseek", "perplexity"
    ]
    
    for prefix in common_prefixes:
        provider_map.add_provider(prefix, create_openrouter_provider(
            prefix=prefix, 
            api_key=api_key, 
            base_url=base_url
        ))
    
    # Store the original __init__ method
    original_init = MultiProvider.__init__
    
    def patched_init(self, **kwargs):
        # Use our custom provider map if none provided
        if 'provider_map' not in kwargs:
            kwargs['provider_map'] = provider_map
        
        # Ensure OpenRouter configuration
        if 'openai_api_key' not in kwargs:
            kwargs['openai_api_key'] = os.getenv("OPENAI_API_KEY")
        if 'openai_base_url' not in kwargs:
            kwargs['openai_base_url'] = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        if 'openai_use_responses' not in kwargs:
            kwargs['openai_use_responses'] = False  # Use chat completions instead of responses
        
        return original_init(self, **kwargs)
    
    # Apply the patch
    MultiProvider.__init__ = patched_init
    
    print("Successfully patched MultiProvider for OpenRouter compatibility")

# Apply the patch when this module is imported
patch_agents_multi_provider()