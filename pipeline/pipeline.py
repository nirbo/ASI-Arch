import asyncio
import sys
import os
import logging

# Suppress TensorFlow and CUDA warnings BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Suppress CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reduce noise from HTTP libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

# Import agents configuration before any agents are created
from pipeline.agents_config import patch_agents_multi_provider

# Apply the patch to handle any model prefix with OpenRouter
patch_agents_multi_provider()

from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI

from pipeline.analyse import analyse
from pipeline.config import Config
from pipeline.database import program_sample, update
from pipeline.eval import evaluation
from pipeline.evolve import evolve
from pipeline.utils.agent_logger import end_pipeline, log_error, log_info, log_step, log_warning, start_pipeline
from pipeline.model_adapters import ModelClientManager, ModelAdapterFactory, get_service_manager

# Initialize the unified model client manager
client_manager = ModelClientManager(
    api_key=Config.OPENAI_API_KEY,
    base_url=Config.OPENAI_BASE_URL,
    default_model=Config.OPENAI_MODEL,
    force_harmony=Config.FORCE_HARMONY_MODE
)

# Harmony detection is simplified - only gpt-oss models use harmony (via unsloth_zoo directly)

# For backward compatibility with agents library, create a wrapper that looks like AsyncOpenAI



class AsyncOpenAICompatWrapper:
    """Wrapper to maintain compatibility with agents library expectations."""
    
    def __init__(self, client_manager: ModelClientManager):
        self.client_manager = client_manager
        
        # Use the existing chat manager from ModelClientManager
        self.chat = client_manager.chat
        
        # Mirror important AsyncOpenAI properties for compatibility
        self.api_key = client_manager.api_key
        self.base_url = client_manager.base_url
        
        # For compatibility - agents library may still check for responses attribute
        # but should use chat.completions.create when API is set to "chat_completions"
        self.responses = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

# Create the compatibility wrapper
client = AsyncOpenAICompatWrapper(client_manager)

set_default_openai_client(client)
set_default_openai_api("chat_completions") 

set_tracing_disabled(True)


async def run_single_experiment() -> bool:
    """Run single experiment loop - using pipeline categorized logging."""
    # Start a new pipeline process
    pipeline_id = start_pipeline("experiment")
    
    try:
        # Step 1: Program sampling
        log_step("Program Sampling", "Start sampling program from database")
        context, parent = await program_sample()
        log_info(f"Program sampling completed, context length: {len(str(context))}")
        
        # Step 2: Evolution
        log_step("Program Evolution", "Start evolving new program")
        name, motivation = await evolve(context)
        if name == "Failed":
            log_error("Program evolution failed")
            end_pipeline(False, "Evolution failed")
            return False
        log_info(f"Program evolution successful, generated program: {name}")
        log_info(f"Evolution motivation: {motivation}")
        
        # Step 3: Evaluation
        log_step("Program Evaluation", f"Start evaluating program {name}")
        success = await evaluation(name, motivation)
        if not success:
            log_error(f"Program {name} evaluation failed")
            end_pipeline(False, "Evaluation failed")
            return False
        log_info(f"Program {name} evaluation successful")
        
        # Step 4: Analysis
        log_step("Result Analysis", f"Start analyzing program {name} results")
        result = await analyse(name, motivation, parent=parent)
        log_info(f"Analysis completed, result: {result}")
        
        # Step 5: Update database
        log_step("Database Update", "Update results to database")
        update(result)
        log_info("Database update completed")
        
        # Successfully complete pipeline
        log_info("Experiment pipeline completed successfully")
        end_pipeline(True, f"Experiment completed successfully, program: {name}, result: {result}")
        return True
        
    except KeyboardInterrupt:
        log_warning("User interrupted experiment")
        end_pipeline(False, "User interrupted experiment")
        return False
    except Exception as e:
        log_error(f"Experiment pipeline unexpected error: {str(e)}")
        end_pipeline(False, f"Unexpected error: {str(e)}")
        return False


async def main():
    """Main function - continuous experiment execution with service management."""
    set_tracing_disabled(True)
    
    log_info("Starting continuous experiment pipeline...")
    
    # Initialize service status tracking
    service_status = await client_manager.get_service_status()
    if service_status:
        log_info(f"Detected harmony services: {list(service_status.keys())}")
    
    # Run plot.py first
    log_info("Running plot scripts...")
    log_info("Plot scripts completed")
    
    experiment_count = 0
    # Only initialize service manager for harmony models that actually need it
    service_manager = None
    if ModelAdapterFactory._is_harmony_model(Config.OPENAI_MODEL):
        service_manager = get_service_manager()
    
    try:
        while True:
            # Check if shutdown was requested via signal (only for harmony models)
            if service_manager and service_manager.is_shutdown_requested():
                log_warning("Shutdown requested via signal, stopping pipeline")
                break
                
            try:
                experiment_count += 1
                log_info(f"Starting experiment {experiment_count}")
                
                success = await run_single_experiment()
                if success:
                    log_info(f"Experiment {experiment_count} completed successfully, starting next experiment...")
                else:
                    log_warning(f"Experiment {experiment_count} failed, retrying in {Config.RETRY_INTERVAL} seconds...")
                    # Check for shutdown during sleep
                    for _ in range(Config.RETRY_INTERVAL):
                        if service_manager and service_manager.is_shutdown_requested():
                            log_warning("Shutdown requested during retry wait")
                            break
                        await asyncio.sleep(1)
                    
            except KeyboardInterrupt:
                log_warning("Continuous experiment interrupted by user")
                # Ensure signal is propagated to service manager
                if service_manager:
                    await service_manager.request_shutdown()
                break
            except Exception as e:
                log_error(f"Main loop unexpected error: {e}")
                log_info("Retrying in 60 seconds...")
                # Check for shutdown during sleep
                for _ in range(60):
                    if service_manager and service_manager.is_shutdown_requested():
                        log_warning("Shutdown requested during error retry wait")
                        break
                    await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        log_warning("Main pipeline interrupted by user")
        # Ensure signal is propagated to service manager
        if service_manager:
            await service_manager.request_shutdown()
    
    finally:
        # Cleanup harmony services when shutting down
        log_info("Shutting down pipeline, cleaning up services...")
        try:
            if client_manager:
                cleanup_success = await client_manager.shutdown_managed_services()
                if cleanup_success:
                    log_info("Harmony services shut down successfully")
                else:
                    log_warning("Some harmony services may not have shut down cleanly")
        except Exception as e:
            log_error(f"Error during service cleanup: {e}")
        
        log_info("Pipeline shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())