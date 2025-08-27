#!/usr/bin/env python
"""Test the run_training_script tool directly."""

import sys
import os
sys.path.append('./pipeline')

import subprocess
import os

def run_training_script_direct(experiment_name: str):
    """Run the training script with the given experiment name."""
    # Change to the project root directory
    original_dir = os.getcwd()
    project_root = "/home/nir/ml-tools/ASI-Arch"
    
    try:
        os.chdir(project_root)
        
        # Clear previous debug file
        debug_file = "./files/debug/training_error.txt"
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)
        
        # Run the training script with experiment name
        result = subprocess.run(
            ["./venv-asi-arch/bin/python", "./train_architecture.py", 
             "--experiment_name", experiment_name,
             "--epochs", "1", "--batch_size", "2", "--max_length", "64",
             "--dataset_size", "10", "--no_wandb", "--max_steps", "5"], 
            capture_output=True, 
            text=True,
            timeout=120  # 2 minute timeout
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
            f.write(f"Error: Training timed out after 2 minutes\n")
        
        return {
            'success': False,
            'error': f"Training timed out after 2 minutes. Debug info written to {debug_file}."
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

def test_training():
    print("Testing run_training_script tool...")
    
    # Test with a simple experiment
    result = run_training_script_direct("test_tool_integration")
    
    print(f"Result: {result}")
    
    # Check if debug file exists
    debug_file = "/home/nir/ml-tools/ASI-Arch/files/debug/training_error.txt"
    if os.path.exists(debug_file):
        print(f"\nDebug file exists: {debug_file}")
        with open(debug_file, 'r') as f:
            content = f.read()
        print(f"Debug file content:\n{content}")
    else:
        print(f"Debug file does not exist: {debug_file}")

if __name__ == "__main__":
    test_training()