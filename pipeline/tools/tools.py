import subprocess
from typing import Any, Dict

from agents import function_tool
from config import Config


@function_tool
def read_code_file() -> Dict[str, Any]:
    """Read a code file and return its contents."""
    source_file = Config.SOURCE_FILE
    try:
        with open(source_file, 'r') as f:
            content = f.read()
        return {
            'success': True,
            'content': content
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@function_tool
def read_csv_file(file_path: str) -> Dict[str, Any]:
    """Read a CSV file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return {
            'success': True,
            'content': content
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@function_tool
def write_code_file(content: str) -> Dict[str, Any]:
    """Write content to a code file with comprehensive corruption detection and validation."""
    source_file = Config.SOURCE_FILE
    try:
        # Comprehensive corruption detection and fixing
        lines = content.split('\n')
        original_line_count = len(lines)
        
        # Remove standalone markdown language markers at start
        removed_lines = []
        while lines and lines[0].strip().lower() in ['python', 'py', '```python', '```py', '```', 'python:', 'py:']:
            removed_line = lines[0].strip()
            removed_lines.append(removed_line)
            print(f"⚠️  Detected corrupted line: '{removed_line}' - removing...")
            lines = lines[1:]
        
        # Remove trailing markdown markers
        while lines and lines[-1].strip() in ['```', '```python', '```py']:
            removed_line = lines[-1].strip()
            removed_lines.append(removed_line)
            print(f"⚠️  Detected trailing markdown marker: '{removed_line}' - removing...")
            lines = lines[:-1]
        
        # Reconstruct content
        content = '\n'.join(lines)
        
        # Enhanced validation
        if not content.strip():
            error_msg = f"Empty content after removing corrupted lines: {removed_lines}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
        
        # Validate Python syntax basics
        if not (content.lstrip().startswith(('#', 'from ', 'import ', 'class ', 'def ', '@')) or 
               'class DeltaNet' in content):
            error_msg = f"Content doesn't look like valid Python code. Starts with: {content[:50]}..."
            print(f"⚠️  {error_msg}")
            # Don't fail here, but log the warning
        
        # Check for required DeltaNet class
        if 'class DeltaNet' not in content:
            error_msg = "Missing required DeltaNet class in architecture code"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
        
        # Success case with detailed logging
        if removed_lines:
            print(f"✅ Fixed architecture file corruption - removed {len(removed_lines)} corrupted lines: {removed_lines}")
        
        with open(source_file, 'w') as f:
            f.write(content)
            
        print(f"✅ Successfully wrote {len(lines)} lines to {source_file}")
        
        return {
            'success': True,
            'message': f'Successfully wrote code to {source_file}',
            'lines_written': len(lines),
            'lines_removed': len(removed_lines),
            'removed_content': removed_lines
        }
        
    except Exception as e:
        error_msg = f"Failed to write code file: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }


@function_tool
def run_training_script(name: str) -> Dict[str, Any]:
    """Run the training script and return its output."""
    try:
        # Use the bash script configured in Config.BASH_SCRIPT with name formatting
        bash_command = Config.BASH_SCRIPT.format(name=name)
        print(f"Executing command: {bash_command}")  # Debug output
        
        result = subprocess.run(bash_command, 
                              shell=True,  # Enable shell to handle the full command
                              capture_output=True, 
                              text=True,
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr or '',
                'error': 'Training script executed successfully',
                'return_code': result.returncode
            }
        else:
            # Training failed - provide detailed error info
            error_details = []
            if result.stderr:
                error_details.append(f"STDERR: {result.stderr}")
            if result.stdout:
                error_details.append(f"STDOUT: {result.stdout}")
            
            error_summary = '\n'.join(error_details) if error_details else 'No error details available'
            
            return {
                'success': False,
                'stdout': result.stdout or '',
                'stderr': result.stderr or '',
                'error': f'Training failed (exit code {result.returncode}). Details:\n{error_summary}',
                'return_code': result.returncode
            }
            
    except subprocess.CalledProcessError as e:
        error_details = []
        if e.stderr:
            error_details.append(f"STDERR: {e.stderr}")
        if e.stdout:
            error_details.append(f"STDOUT: {e.stdout}")
        
        error_summary = '\n'.join(error_details) if error_details else 'No error details available'
        
        return {
            'success': False,
            'stdout': e.stdout or '',
            'stderr': e.stderr or '',
            'error': f'Training script failed with exit code {e.returncode}. Details:\n{error_summary}',
            'return_code': e.returncode
        }
    except subprocess.TimeoutExpired as e:
        return {
            'success': False,
            'stdout': e.stdout or '',
            'stderr': e.stderr or '',
            'error': f'Training script timed out after 5 minutes. Details:\nSTDOUT: {e.stdout or "None"}\nSTDERR: {e.stderr or "None"}',
            'return_code': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': '',
            'error': f'Unexpected error running training script: {str(e)}',
            'return_code': -2
        }


@function_tool
def run_plot_script(script_path: str) -> Dict[str, Any]:
    """Run the plotting script."""
    try:
        result = subprocess.run(['python', script_path],
                              capture_output=True,
                              text=True,
                              check=True)
        return {
            'success': True,
            'output': result.stdout,
            'error': result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'output': e.stdout,
            'error': e.stderr
        }


def run_rag(query: str) -> Dict[str, Any]:
    """Run RAG and return the results."""
    try:
        import requests
        
        response = requests.post(
            f'{Config.RAG}/search',
            headers={'Content-Type': 'application/json'},
            json={
                'query': query,
                'k': 3, 
                'similarity_threshold': 0.5
            }
        )
        
        response.raise_for_status()
        results = response.json()
        
        return {
            'success': True,
            'results': results
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }