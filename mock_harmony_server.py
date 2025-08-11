#!/usr/bin/env python3
"""
Mock Harmony Server for testing ASI-Arch pipeline

This is a minimal HTTP server that simulates harmony service responses
for testing the dual-mode Harmony+Standard architecture.
"""

import argparse
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import signal
import sys
from urllib.parse import urlparse, parse_qs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HarmonyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock harmony service"""
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr"""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def do_GET(self):
        """Handle GET requests (health checks)"""
        if self.path in ['/health', '/v1/health']:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "harmony-server",
                "model": getattr(self.server, 'model', 'gpt-oss-20b'),
                "version": "1.0.0-mock"
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests (completions)"""
        if self.path in ['/v1/completions', '/completions']:
            try:
                # Read request body
                content_length = int(self.headers.get('Content-Length', 0))
                request_body = self.rfile.read(content_length).decode('utf-8')
                request_data = json.loads(request_body) if request_body else {}
                
                # Extract request parameters
                model = request_data.get('model', 'gpt-oss-20b')
                prompt = request_data.get('prompt', '')
                max_tokens = request_data.get('max_tokens', 150)
                temperature = request_data.get('temperature', 0.7)
                
                # Determine response type based on prompt content
                # Handle both string and list prompts
                if isinstance(prompt, list):
                    prompt_text = ' '.join(str(p) for p in prompt)
                else:
                    prompt_text = str(prompt)
                
                is_planner_request = any(phrase in prompt_text.lower() for phrase in [
                    'neural architecture evolution', 'write_code_file', 'read_code_file',
                    'architecture designer', 'deliverable specifications'
                ])
                
                is_summary_request = any(phrase in prompt_text.lower() for phrase in [
                    'experience synthesis', 'experimental context', 'synthesis task'
                ])
                
                # Generate mock 3-channel harmony response
                if is_planner_request:
                    response_content = """
<analysis>
Processing architecture evolution request with harmony encoding.
Analyzing experimental context and architectural requirements.
Preparing comprehensive design response with tool usage directives.
</analysis>

<commentary>
Mock harmony service generating planner response.
System requires concrete code implementation using available tools.
Agent must use read_code_file and write_code_file for proper implementation.
</commentary>

<final>
{
    "name": "harmony_enhanced_deltanet",
    "motivation": "Based on experimental evidence, I will implement an enhanced DeltaNet architecture that integrates advanced attention mechanisms with improved computational efficiency. First, I must use read_code_file to examine the current implementation, then use write_code_file to implement specific improvements including: 1) Enhanced multi-head attention patterns, 2) Improved residual connections, 3) Optimized normalization strategies, and 4) Better parameter initialization. The implementation will maintain sub-quadratic complexity while improving performance on reasoning tasks."
}
</final>
"""
                elif is_summary_request:
                    response_content = """
<analysis>
Processing experience synthesis request with harmony encoding.
Analyzing experimental data and architectural performance patterns.
Generating comprehensive experience summary for planning guidance.
</analysis>

<commentary>
Mock harmony service generating summary response.
Extracting key insights from experimental context and performance data.
Providing actionable intelligence for architectural evolution.
</commentary>

<final>
{
    "experience": "The experimental evidence reveals critical insights about the current architecture: 1) Linear attention mechanisms show promise for O(n) complexity but require optimization for long sequences, 2) Hierarchical reasoning modules demonstrate improved performance on complex tasks but need better integration with attention layers, 3) Multi-timescale processing enables both tactical and strategic reasoning but convergence detection needs refinement, 4) Cross-modal fusion creates beneficial synergies but introduces computational overhead that should be optimized. Future architectural evolution should focus on: enhanced attention-reasoning integration, improved convergence mechanisms, and optimized cross-modal processing while maintaining linear computational complexity."
}
</final>
"""
                else:
                    # Default to planner format since most requests are planner requests
                    response_content = """
<analysis>
Processing architecture request with harmony encoding.
Mock harmony service providing simulated planner response.
</analysis>

<commentary>
General architecture request processed by mock harmony service.
Providing structured planner response with name and motivation fields.
</commentary>

<final>
{
    "name": "mock_harmony_architecture",
    "motivation": "This is a mock architecture generated by the harmony service simulation. In production, this would be replaced with actual model-generated architectural designs based on the harmony encoding system and experimental context provided."
}
</final>
"""
                
                # Send successful response in proper OpenAI completion format
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                # Wrap response in OpenAI completion format
                response_json = {
                    "choices": [
                        {
                            "text": response_content,
                            "index": 0,
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(str(request_data.get('prompt', ''))),
                        "completion_tokens": len(response_content),
                        "total_tokens": len(str(request_data.get('prompt', ''))) + len(response_content)
                    }
                }
                self.wfile.write(json.dumps(response_json).encode())
                
                logger.info(f"Served completion for model {model}, prompt length: {len(prompt)}")
                
            except Exception as e:
                logger.error(f"Error processing completion request: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {"error": f"Internal server error: {str(e)}"}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()


class MockHarmonyServer:
    """Mock Harmony Server for testing"""
    
    def __init__(self, host='localhost', port=8080, model='gpt-oss-20b'):
        self.host = host
        self.port = port
        self.model = model
        self.httpd = None
        self.running = False
    
    def start(self):
        """Start the mock harmony server"""
        try:
            self.httpd = HTTPServer((self.host, self.port), HarmonyHandler)
            self.httpd.model = self.model
            self.running = True
            
            logger.info(f"Mock Harmony Server starting on {self.host}:{self.port}")
            logger.info(f"Model: {self.model}")
            logger.info(f"Health endpoint: http://{self.host}:{self.port}/health")
            logger.info(f"Completions endpoint: http://{self.host}:{self.port}/v1/completions")
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start serving
            try:
                self.httpd.serve_forever()
            except KeyboardInterrupt:
                logger.info("Received KeyboardInterrupt")
                self.stop()
            finally:
                if self.running:
                    self.stop()
            
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.error(f"Port {self.port} is already in use")
            else:
                logger.error(f"Failed to start server: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Server error: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        
        # Mark as not running first
        self.running = False
        
        # Use a separate thread to shutdown since httpd.shutdown() can block
        def shutdown_thread():
            if self.httpd:
                try:
                    logger.info("Stopping Mock Harmony Server...")
                    self.httpd.shutdown()
                    self.httpd.server_close()
                    logger.info("Mock Harmony Server stopped")
                except Exception as e:
                    logger.error(f"Error during shutdown: {e}")
            
            # Force exit after a brief moment
            import time
            time.sleep(0.1)
            logger.info("Mock server exiting...")
            os._exit(0)
        
        # Start shutdown in background thread
        import threading
        threading.Thread(target=shutdown_thread, daemon=True).start()
        
        # Also set a timeout to force exit if shutdown hangs
        def timeout_exit():
            import time
            time.sleep(2.0)  # Wait max 2 seconds
            logger.warning("Shutdown timeout - force exiting")
            os._exit(0)
        
        threading.Thread(target=timeout_exit, daemon=True).start()
    
    def stop(self):
        """Stop the mock harmony server"""
        if self.httpd and self.running:
            logger.info("Stopping Mock Harmony Server...")
            try:
                # Set running to False first to stop serve_forever
                self.running = False
                self.httpd.shutdown()
                self.httpd.server_close()
                logger.info("Mock Harmony Server stopped")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
                # Force exit on error
                os._exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Mock Harmony Server for ASI-Arch testing')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--model', default='gpt-oss-20b', help='Model name to serve')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout (unused, for compatibility)')
    
    args = parser.parse_args()
    
    server = MockHarmonyServer(host=args.host, port=args.port, model=args.model)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        server.stop()


if __name__ == '__main__':
    main()