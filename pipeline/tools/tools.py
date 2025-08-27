import subprocess
from typing import Any, Dict
import os
import shutil

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
    """Write content to a code file with H1-Titans protection."""
    source_file = Config.SOURCE_FILE
    try:
        # Create backup before overwriting
        backup_file = source_file + ".backup"
        if os.path.exists(source_file):
            shutil.copy2(source_file, backup_file)
        
        # Verify content has required H1-Titans components
        if ('class H1TitansModel' not in content or 
            'def build_model' not in content):
            return {
                'success': False,
                'error': 'Content missing required H1-Titans components (H1TitansModel, build_model). Write blocked to prevent corruption.'
            }
            
        with open(source_file, 'w') as f:
            f.write(content)
        return {
            'success': True,
            'message': f'Successfully wrote H1-Titans architecture to {source_file}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@function_tool
def run_training_script(experiment_name: str) -> Dict[str, Any]:
    """Run the training script with the given experiment name."""
    import os
    
    # Change to the project root directory
    original_dir = os.getcwd()
    project_root = "/home/nir/ml-tools/ASI-Arch"
    
    try:
        os.chdir(project_root)
        
        # Clear previous debug file
        debug_file = "./files/debug/training_error.txt"
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)
        
        # Run the training script with GPU-optimized arguments for RTX 5090
        result = subprocess.run(
            ["./venv-asi-arch/bin/python", "./train_architecture.py", 
             "--batch_size", "16", "--seq_len", "1024", "--max_steps", "500",
             "--eval_every", "50", "--bf16", "--compile"], 
            capture_output=True, 
            text=True,
            timeout=900  # 15 minute timeout for longer training
        )
        
        if result.returncode == 0:
            return {
                'success': True,
                'output': result.stdout
            }
        else:
            # Write error to debug file
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"Experiment: {experiment_name}\n")
                f.write(f"Error: {result.stderr}\n")
                if result.stdout:
                    f.write(f"Output: {result.stdout}\n")
            
            return {
                'success': False,
                'error': f"Training failed with return code {result.returncode}. Debug info written to {debug_file}."
            }
            
    except subprocess.TimeoutExpired:
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Error: Training timed out after 5 minutes\n")
        
        return {
            'success': False,
            'error': f"Training timed out after 5 minutes. Debug info written to {debug_file}."
        }
        
    except Exception as e:
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Error: Unexpected error: {str(e)}\n")
        
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}. Debug info written to {debug_file}."
        }
    
    finally:
        os.chdir(original_dir)


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