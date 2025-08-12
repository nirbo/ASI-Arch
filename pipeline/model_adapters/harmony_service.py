"""
Harmony Service Manager

Manages the lifecycle of harmony services for gpt-oss models.
Provides automatic start/stop functionality, health checking, and configuration management.

Features:
- Automatic service detection and startup when harmony models are needed
- Health checking with retries and timeout handling
- Graceful shutdown with proper resource cleanup
- Configuration management for service parameters
- Process management with subprocess handling
- Concurrent request handling to the same service
- Service failure detection and fallback behavior
"""

import asyncio
import logging
import subprocess
import time
import signal
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import json

# Import configuration
try:
    from ..config import Config
except ImportError:
    # Handle when running from different contexts
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import Config

# Import aiohttp for HTTP requests to harmony service
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

logger = logging.getLogger(__name__)


@dataclass
class HarmonyServiceConfig:
    """Configuration for a harmony service instance."""
    
    # Service identification
    model: str
    host: str = "localhost"
    port: int = 8080
    
    # Service startup parameters
    command: List[str] = field(default_factory=list)  # Command to start service
    working_dir: Optional[str] = None  # Working directory for service
    environment: Dict[str, str] = field(default_factory=dict)  # Environment variables
    
    # Service behavior
    startup_timeout: float = 120.0  # Maximum time to wait for service startup
    health_check_timeout: float = 10.0  # Timeout for individual health checks
    health_check_interval: float = 1.0  # Interval between health checks during startup
    shutdown_timeout: float = 30.0  # Maximum time to wait for graceful shutdown
    
    # Service URLs
    base_url: Optional[str] = None  # Computed from host/port if not provided
    health_endpoint: str = "/health"  # Health check endpoint
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.base_url is None:
            self.base_url = f"http://{self.host}:{self.port}"
    
    @property
    def health_url(self) -> str:
        """Get the health check URL."""
        return f"{self.base_url.rstrip('/')}{self.health_endpoint}"


class HarmonyServiceManager:
    """
    Manages harmony service lifecycle for automatic start/stop functionality.
    
    This manager handles:
    - Detecting when harmony services are needed
    - Starting services on-demand with proper configuration
    - Health checking services to ensure readiness
    - Managing multiple concurrent requests to the same service
    - Graceful shutdown with resource cleanup
    - Service failure detection and recovery
    """
    
    def __init__(self):
        """Initialize the harmony service manager."""
        self._running_services: Dict[str, "ServiceInstance"] = {}  # model -> ServiceInstance
        self._service_configs: Dict[str, HarmonyServiceConfig] = {}  # model -> config
        self._startup_locks: Dict[str, asyncio.Lock] = {}  # model -> lock for startup
        self._shutdown_in_progress: Set[str] = set()  # models being shut down
        self._global_shutdown = False
        
        # Register signal handlers for cleanup
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (OSError, ValueError):
            # Signal handling might not be available in all contexts
            pass
        
        # Only log initialization if we actually expect to use harmony services
        # Check if any models in the current configuration might need harmony services
        try:
            from ..config import Config
            if hasattr(Config, 'OPENAI_MODEL') and Config.OPENAI_MODEL:
                # Import factory here to avoid circular imports
                from .factory import ModelAdapterFactory
                if ModelAdapterFactory._is_harmony_model(Config.OPENAI_MODEL):
                    logger.info("Initialized HarmonyServiceManager for harmony model")
                else:
                    logger.debug("Initialized HarmonyServiceManager for signal handling (no harmony services needed)")
            else:
                logger.debug("Initialized HarmonyServiceManager (no model configured)")
        except ImportError:
            logger.debug("Initialized HarmonyServiceManager")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self._global_shutdown = True
        
        # Schedule cleanup to run in the event loop instead of doing it in signal handler
        # This prevents blocking and hanging issues
        if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop() is not None:
            # Schedule shutdown in the running event loop
            try:
                loop = asyncio.get_running_loop()
                if not loop.is_closed():
                    loop.call_soon_threadsafe(self._schedule_shutdown)
                    logger.info("Scheduled shutdown in event loop")
                else:
                    logger.warning("Event loop is closed, cannot schedule shutdown")
            except RuntimeError:
                logger.warning("No running event loop found for signal handler")
        else:
            logger.info("No event loop running, shutdown flag set for main pipeline to handle")
    
    def _schedule_shutdown(self):
        """Schedule shutdown task in event loop."""
        try:
            # Create a shutdown task if none exists
            if not hasattr(self, '_shutdown_task') or self._shutdown_task is None or self._shutdown_task.done():
                self._shutdown_task = asyncio.create_task(self._signal_initiated_shutdown())
                logger.info("Created shutdown task from signal handler")
        except Exception as e:
            logger.error(f"Error scheduling shutdown task: {e}")
    
    async def _signal_initiated_shutdown(self):
        """Shutdown initiated by signal handler."""
        try:
            logger.info("Starting signal-initiated shutdown process")
            
            # Set a maximum time for the entire shutdown process
            shutdown_start = time.time()
            max_total_shutdown_time = 15.0  # Maximum 15 seconds for entire shutdown
            
            try:
                await asyncio.wait_for(
                    self.shutdown_all_services(),
                    timeout=max_total_shutdown_time
                )
                elapsed = time.time() - shutdown_start
                logger.info(f"Signal-initiated service shutdown completed in {elapsed:.1f}s")
                
            except asyncio.TimeoutError:
                elapsed = time.time() - shutdown_start
                logger.error(f"Signal-initiated shutdown timed out after {elapsed:.1f}s")
                # Force kill everything
                await self._force_kill_remaining_services()
                
            # Schedule main process exit after a brief delay to allow logging to flush
            asyncio.get_event_loop().call_later(0.5, self._force_exit)
            
        except Exception as e:
            logger.error(f"Error in signal-initiated shutdown: {e}")
            # Still try to exit
            asyncio.get_event_loop().call_later(1.0, self._force_exit)
    
    def _force_exit(self):
        """Force exit the process if normal shutdown doesn't work."""
        logger.info("Forcing process exit after service cleanup")
        try:
            # Give any pending operations a moment to complete
            import sys
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during forced exit: {e}")
            # Last resort
            os._exit(0)
    
    async def request_shutdown(self):
        """Request shutdown from main application loop."""
        self._global_shutdown = True
        logger.info("Shutdown requested, main application should handle cleanup")
    
    def register_service_config(self, config: HarmonyServiceConfig) -> None:
        """
        Register configuration for a harmony service.
        
        Args:
            config: Service configuration to register
        """
        self._service_configs[config.model] = config
        logger.info(f"Registered service config for model: {config.model}")
    
    def get_default_config(self, model: str, port: int = 8080) -> HarmonyServiceConfig:
        """
        Create a default configuration for a harmony service.
        
        Args:
            model: Model name (e.g., "gpt-oss-20b")
            port: Port to run the service on
            
        Returns:
            Default HarmonyServiceConfig for the model
        """
        # Use command from config if available
        if Config.HARMONY_SERVICE_COMMAND:
            command = [
                arg.format(
                    model=model,
                    port=str(port),
                    host=Config.HARMONY_SERVICE_HOST
                ) 
                for arg in Config.HARMONY_SERVICE_COMMAND
            ]
        else:
            # Fallback to generic command
            command = [
                "python", "-m", "harmony_service",  # Hypothetical harmony service module
                "--model", model,
                "--host", Config.HARMONY_SERVICE_HOST,
                "--port", str(port),
                "--timeout", "300"
            ]
        
        return HarmonyServiceConfig(
            model=model,
            host=Config.HARMONY_SERVICE_HOST,
            port=port,
            command=command,
            working_dir=Config.HARMONY_SERVICE_WORKING_DIR or None,
            environment=Config.HARMONY_SERVICE_ENVIRONMENT.copy(),
            startup_timeout=Config.HARMONY_SERVICE_STARTUP_TIMEOUT,
            health_check_timeout=Config.HARMONY_SERVICE_HEALTH_TIMEOUT,
            health_check_interval=1.0,
            shutdown_timeout=Config.HARMONY_SERVICE_SHUTDOWN_TIMEOUT
        )
    
    async def ensure_service_running(self, model: str) -> Tuple[bool, str]:
        """
        Ensure that a harmony service is running for the given model.
        
        This method:
        1. Checks if service is already running and healthy
        2. If not, starts the service with proper configuration
        3. Waits for service to become ready with health checks
        4. Returns success status and connection URL
        
        Args:
            model: Model name that needs harmony service
            
        Returns:
            Tuple of (success: bool, base_url: str)
            If success is False, base_url contains error message
        """
        if self._global_shutdown:
            return False, "Service manager is shutting down"
        
        # Check if service is already running and healthy
        if model in self._running_services:
            service = self._running_services[model]
            if await service.is_healthy():
                logger.debug(f"Service for {model} already running and healthy")
                return True, service.config.base_url
            else:
                logger.warning(f"Service for {model} exists but unhealthy, restarting")
                await self._stop_service(model)
        
        # Ensure we have a startup lock for this model
        if model not in self._startup_locks:
            self._startup_locks[model] = asyncio.Lock()
        
        # Use lock to prevent concurrent startup of the same service
        async with self._startup_locks[model]:
            # Double-check service isn't running after acquiring lock
            if model in self._running_services:
                service = self._running_services[model]
                if await service.is_healthy():
                    logger.debug(f"Service for {model} started by another coroutine")
                    return True, service.config.base_url
            
            # Start the service
            success, message = await self._start_service(model)
            if success and model in self._running_services:
                return True, self._running_services[model].config.base_url
            else:
                return False, message
    
    async def _start_service(self, model: str) -> Tuple[bool, str]:
        """
        Start a harmony service for the given model.
        
        Args:
            model: Model name to start service for
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Get or create configuration
            if model in self._service_configs:
                config = self._service_configs[model]
                logger.info(f"Using registered config for model: {model}")
            else:
                # Find an available port
                port = await self._find_available_port()
                config = self.get_default_config(model, port)
                logger.info(f"Created default config for model: {model} on port {port}")
            
            # Check if the port is actually available
            if not await self._is_port_available(config.port):
                # Try to find another port
                new_port = await self._find_available_port(start_port=config.port + 1)
                logger.warning(f"Port {config.port} not available, using {new_port}")
                config.port = new_port
                config.base_url = f"http://{config.host}:{new_port}"
            
            logger.info(f"Starting harmony service for model {model} on {config.host}:{config.port}")
            
            # Start the service process
            service_instance = await self._launch_service_process(config)
            if service_instance is None:
                return False, "Failed to launch service process"
            
            # Wait for service to become healthy
            logger.info(f"Waiting for service {model} to become ready...")
            healthy = await self._wait_for_service_health(service_instance)
            
            if healthy:
                # Store the running service
                self._running_services[model] = service_instance
                logger.info(f"Service for model {model} started successfully at {config.base_url}")
                return True, f"Service started at {config.base_url}"
            else:
                # Service failed to become healthy, clean up
                await service_instance.stop()
                return False, f"Service failed to become healthy within {config.startup_timeout}s"
        
        except Exception as e:
            logger.error(f"Error starting service for model {model}: {str(e)}")
            return False, f"Service startup error: {str(e)}"
    
    async def _launch_service_process(self, config: HarmonyServiceConfig) -> Optional["ServiceInstance"]:
        """
        Launch the harmony service process.
        
        Args:
            config: Service configuration
            
        Returns:
            ServiceInstance if successful, None otherwise
        """
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(config.environment)
            
            # Set working directory
            working_dir = config.working_dir or os.getcwd()
            
            logger.debug(f"Launching service with command: {' '.join(config.command)}")
            logger.debug(f"Working directory: {working_dir}")
            
            # Start the process
            process = await asyncio.create_subprocess_exec(
                *config.command,
                cwd=working_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None  # Create process group on Unix
            )
            
            # Store manager reference in config for service instance
            config._manager = self
            
            # Create service instance
            service_instance = ServiceInstance(config, process)
            
            # Start monitoring task
            service_instance.start_monitoring()
            
            logger.debug(f"Service process launched with PID: {process.pid}")
            return service_instance
        
        except Exception as e:
            logger.error(f"Failed to launch service process: {str(e)}")
            return None
    
    async def _wait_for_service_health(self, service: "ServiceInstance") -> bool:
        """
        Wait for a service to become healthy.
        
        Args:
            service: Service instance to check
            
        Returns:
            True if service became healthy within timeout
        """
        config = service.config
        start_time = time.time()
        
        logger.debug(f"Checking service health at {config.health_url}")
        
        while time.time() - start_time < config.startup_timeout:
            if self._global_shutdown:
                logger.info("Global shutdown detected, stopping health check")
                return False
            
            # Check if process is still running
            if service.process.returncode is not None:
                logger.error(f"Service process exited with code {service.process.returncode}")
                return False
            
            # Perform health check
            if await service.is_healthy():
                elapsed = time.time() - start_time
                logger.info(f"Service became healthy after {elapsed:.1f}s")
                return True
            
            # Wait before next check
            await asyncio.sleep(config.health_check_interval)
        
        logger.error(f"Service failed to become healthy within {config.startup_timeout}s")
        return False
    
    async def _stop_service(self, model: str) -> bool:
        """
        Stop a harmony service for the given model.
        
        Args:
            model: Model name to stop service for
            
        Returns:
            True if service stopped successfully
        """
        if model not in self._running_services:
            logger.debug(f"No service running for model: {model}")
            return True
        
        if model in self._shutdown_in_progress:
            logger.debug(f"Service for {model} already shutting down")
            return True
        
        self._shutdown_in_progress.add(model)
        
        try:
            service = self._running_services[model]
            logger.info(f"Stopping service for model: {model}")
            
            success = await service.stop()
            
            # Remove from running services
            del self._running_services[model]
            
            if success:
                logger.info(f"Service for model {model} stopped successfully")
            else:
                logger.warning(f"Service for model {model} may not have stopped cleanly")
            
            return success
        
        except Exception as e:
            logger.error(f"Error stopping service for model {model}: {str(e)}")
            return False
        
        finally:
            self._shutdown_in_progress.discard(model)
    
    async def shutdown_all_services(self) -> bool:
        """
        Shutdown all running services gracefully.
        
        Returns:
            True if all services stopped successfully
        """
        if not self._running_services:
            logger.info("No services running, shutdown complete")
            return True
        
        logger.info(f"Shutting down {len(self._running_services)} services...")
        
        # Stop all services concurrently
        shutdown_tasks = []
        for model in list(self._running_services.keys()):
            task = asyncio.create_task(self._stop_service(model))
            shutdown_tasks.append(task)
        
        # Wait for all shutdowns to complete with timeout
        try:
            # Use an even shorter timeout for more responsive shutdown, especially during signal handling
            if self._global_shutdown:
                max_shutdown_timeout = 8.0  # Shorter timeout for signal-initiated shutdown
            else:
                max_shutdown_timeout = min(Config.HARMONY_SERVICE_SHUTDOWN_TIMEOUT, 15.0)
            
            logger.info(f"Waiting up to {max_shutdown_timeout}s for services to shutdown gracefully")
            results = await asyncio.wait_for(
                asyncio.gather(*shutdown_tasks, return_exceptions=True),
                timeout=max_shutdown_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Service shutdown timed out after {max_shutdown_timeout}s, forcing termination")
            # Force kill any remaining processes
            await self._force_kill_remaining_services()
            return False
        
        # Check results
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error during service shutdown: {result}")
            elif result:
                success_count += 1
        
        all_success = success_count == len(results)
        logger.info(f"Service shutdown complete: {success_count}/{len(results)} successful")
        return all_success
    
    async def _force_kill_remaining_services(self):
        """Force kill any remaining service processes."""
        force_kill_tasks = []
        
        for model, service in list(self._running_services.items()):
            force_kill_tasks.append(self._force_kill_single_service(model, service))
        
        if force_kill_tasks:
            logger.warning(f"Force killing {len(force_kill_tasks)} remaining services")
            # Run force kills concurrently with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*force_kill_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.error("Force kill operations timed out")
        
        # Clear all services
        self._running_services.clear()
    
    async def _force_kill_single_service(self, model: str, service):
        """Force kill a single service process."""
        try:
            if service.process.returncode is None:
                logger.warning(f"Force killing service process {service.process.pid} for model {model}")
                
                if os.name != 'nt':
                    try:
                        os.killpg(os.getpgid(service.process.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        try:
                            service.process.kill()
                        except ProcessLookupError:
                            pass  # Process already dead
                else:
                    try:
                        service.process.kill()
                    except ProcessLookupError:
                        pass  # Process already dead
                
                # Give it a moment to die with shorter timeout
                try:
                    await asyncio.wait_for(service.process.wait(), timeout=1.0)
                    logger.debug(f"Service process {service.process.pid} terminated")
                except asyncio.TimeoutError:
                    logger.error(f"Failed to kill service process {service.process.pid} after SIGKILL")
            
            # Remove from running services
            if model in self._running_services:
                del self._running_services[model]
                
        except Exception as e:
            logger.error(f"Error force killing service {model}: {e}")
    
    async def get_service_status(self, model: str) -> Dict:
        """
        Get status information for a service.
        
        Args:
            model: Model name to check
            
        Returns:
            Dictionary with service status information
        """
        if model not in self._running_services:
            return {
                "running": False,
                "model": model,
                "message": "Service not running"
            }
        
        service = self._running_services[model]
        is_healthy = await service.is_healthy()
        
        return {
            "running": True,
            "healthy": is_healthy,
            "model": model,
            "base_url": service.config.base_url,
            "pid": service.process.pid,
            "uptime": time.time() - service.start_time
        }
    
    async def get_all_service_status(self) -> Dict[str, Dict]:
        """
        Get status for all managed services.
        
        Returns:
            Dictionary mapping model names to status dictionaries
        """
        status_dict = {}
        
        # Get status for running services
        for model in self._running_services:
            status_dict[model] = await self.get_service_status(model)
        
        return status_dict
    
    async def _find_available_port(self, start_port: int = 8080) -> int:
        """
        Find an available port starting from the given port.
        
        Args:
            start_port: Port to start searching from
            
        Returns:
            Available port number
        """
        port = start_port
        max_attempts = 100
        
        for _ in range(max_attempts):
            if await self._is_port_available(port):
                return port
            port += 1
        
        # Fallback: return start_port and let the service fail if it can't bind
        logger.warning(f"Could not find available port after {max_attempts} attempts")
        return start_port
    
    async def _is_port_available(self, port: int, host: str = "localhost") -> bool:
        """
        Check if a port is available for binding.
        
        Args:
            port: Port number to check
            host: Host to check (default: localhost)
            
        Returns:
            True if port is available
        """
        try:
            # Try to create a socket and bind to the port
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result = sock.bind((host, port))
            sock.close()
            return True
        except OSError:
            return False
    
    def is_shutdown_requested(self) -> bool:
        """
        Check if shutdown was requested via signal.
        
        Returns:
            True if shutdown was requested
        """
        return self._global_shutdown
    
    def is_harmony_model_needed(self, model: str) -> bool:
        """
        Check if a model requires harmony service.
        
        This integrates with the ModelAdapterFactory pattern detection.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model needs harmony service
        """
        from .factory import ModelAdapterFactory
        return ModelAdapterFactory._is_harmony_model(model)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.shutdown_all_services()


class ServiceInstance:
    """
    Represents a running harmony service instance.
    
    Manages the service process, health checking, and cleanup.
    """
    
    def __init__(self, config: HarmonyServiceConfig, process: asyncio.subprocess.Process):
        """
        Initialize service instance.
        
        Args:
            config: Service configuration
            process: The running subprocess
        """
        self.config = config
        self.process = process
        self.start_time = time.time()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[float] = None
        self._health_check_cache: Optional[bool] = None
        self._health_check_cache_duration = 5.0  # Cache health check results for 5 seconds
        
        # Store reference to manager for shutdown state checking
        if hasattr(config, '_manager'):
            self._manager = config._manager
        else:
            self._manager = None
    
    def start_monitoring(self):
        """Start background monitoring of the service process."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_process())
    
    async def _monitor_process(self):
        """Monitor the service process for unexpected termination."""
        try:
            await self.process.wait()
            logger.warning(f"Service process for {self.config.model} terminated unexpectedly")
        except asyncio.CancelledError:
            logger.debug(f"Monitoring cancelled for {self.config.model}")
        except Exception as e:
            logger.error(f"Error monitoring service process: {e}")
    
    async def is_healthy(self) -> bool:
        """
        Check if the service is healthy and responsive.
        
        Uses caching to avoid overwhelming the service with health checks.
        
        Returns:
            True if service is healthy
        """
        # Check cache first
        now = time.time()
        if (self._last_health_check is not None and 
            self._health_check_cache is not None and
            now - self._last_health_check < self._health_check_cache_duration):
            return self._health_check_cache
        
        # Perform actual health check
        try:
            # Use shorter timeout for health checks during shutdown
            timeout_duration = self.config.health_check_timeout
            if self._manager and getattr(self._manager, '_global_shutdown', False):
                timeout_duration = min(timeout_duration, 2.0)  # Shorter timeout during shutdown
            
            timeout = aiohttp.ClientTimeout(total=timeout_duration, connect=timeout_duration/2)
            
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=1, limit_per_host=1)
            ) as session:
                async with session.get(self.config.health_url) as response:
                    is_healthy = response.status == 200
            
            # Update cache
            self._last_health_check = now
            self._health_check_cache = is_healthy
            
            return is_healthy
        
        except Exception as e:
            logger.debug(f"Health check failed for {self.config.model}: {e}")
            
            # Update cache with failure
            self._last_health_check = now
            self._health_check_cache = False
            
            return False
    
    async def stop(self) -> bool:
        """
        Stop the service process gracefully.
        
        Returns:
            True if stopped successfully
        """
        try:
            # Cancel monitoring with timeout
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await asyncio.wait_for(self._monitoring_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Try graceful shutdown first
            if self.process.returncode is None:
                logger.debug(f"Sending SIGTERM to service process {self.process.pid}")
                
                if os.name != 'nt':
                    # On Unix, send SIGTERM to process group
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        # Fallback to killing just the process
                        self.process.terminate()
                else:
                    # On Windows, use terminate
                    self.process.terminate()
                
                # Wait for graceful shutdown with shorter timeout for responsiveness
                try:
                    # Use even shorter timeout during global shutdown
                    if self._manager and getattr(self._manager, '_global_shutdown', False):
                        shutdown_timeout = 3.0  # Very short timeout during signal shutdown
                    else:
                        shutdown_timeout = min(self.config.shutdown_timeout, 8.0)
                    
                    await asyncio.wait_for(
                        self.process.wait(), 
                        timeout=shutdown_timeout
                    )
                    logger.debug(f"Service process {self.process.pid} terminated gracefully")
                    return True
                
                except asyncio.TimeoutError:
                    logger.warning(f"Service process {self.process.pid} did not terminate gracefully")
                    
                    # Force kill
                    if self.process.returncode is None:
                        logger.debug(f"Sending SIGKILL to service process {self.process.pid}")
                        
                        if os.name != 'nt':
                            try:
                                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                            except (OSError, ProcessLookupError):
                                self.process.kill()
                        else:
                            self.process.kill()
                        
                        # Wait a bit more
                        try:
                            await asyncio.wait_for(self.process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            logger.error(f"Failed to kill service process {self.process.pid}")
                            return False
            
            logger.info(f"Service process {self.process.pid} stopped")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping service process: {e}")
            return False


# Global service manager instance
_global_service_manager: Optional[HarmonyServiceManager] = None


def get_service_manager() -> HarmonyServiceManager:
    """
    Get the global harmony service manager instance.
    
    Returns:
        Global HarmonyServiceManager instance
    """
    global _global_service_manager
    if _global_service_manager is None:
        _global_service_manager = HarmonyServiceManager()
    return _global_service_manager


async def ensure_harmony_service(model: str) -> Tuple[bool, str]:
    """
    Convenience function to ensure harmony service is running for a model.
    
    Args:
        model: Model name that needs harmony service
        
    Returns:
        Tuple of (success: bool, base_url_or_error: str)
    """
    manager = get_service_manager()
    return await manager.ensure_service_running(model)


async def shutdown_harmony_services():
    """
    Convenience function to shutdown all harmony services.
    
    Returns:
        True if all services stopped successfully
    """
    global _global_service_manager
    if _global_service_manager is not None:
        result = await _global_service_manager.shutdown_all_services()
        _global_service_manager = None
        return result
    return True