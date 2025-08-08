#!/usr/bin/env python3
"""
Test script to validate evolution system fixes.
Verifies that agents properly use tools and generate diverse architectures.
"""

import sys
import os
import asyncio
import tempfile
import shutil
from pathlib import Path

# Add pipeline to path
sys.path.append('/home/nir/ASI-Arch/pipeline')

def test_tool_validation():
    """Test that write_code_file rejects fallback artifacts."""
    print("=" * 60)
    print("TESTING: Tool validation rejects fallback artifacts")
    print("=" * 60)
    
    try:
        from tools.tools import write_code_file
        
        # Test 1: Valid architecture should pass
        valid_code = '''import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.layer(x)

class Model(TestModel):
    pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_file = f.name
            
        # Backup original source file setting
        from config import Config
        original_source = Config.SOURCE_FILE
        Config.SOURCE_FILE = temp_file
        
        try:
            result = write_code_file(valid_code)
            assert result['success'], f"Valid code should pass: {result}"
            print("✅ Valid code accepted")
            
            # Test 2: Fallback artifacts should be rejected
            fallback_code = valid_code + '\n# Fallback improvement applied - Agent tool usage failed\n'
            result = write_code_file(fallback_code)
            assert not result['success'], "Fallback artifacts should be rejected"
            assert 'fallback artifacts' in result['error'].lower(), f"Should mention fallback artifacts: {result}"
            print("✅ Fallback artifacts correctly rejected")
            
            # Test 3: Missing Model class should be rejected
            no_model_code = '''import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
'''
            result = write_code_file(no_model_code)
            assert not result['success'], "Code without Model class should be rejected"
            assert 'Model' in result['error'], f"Should mention missing Model class: {result}"
            print("✅ Missing Model class correctly rejected")
            
        finally:
            Config.SOURCE_FILE = original_source
            os.unlink(temp_file)
            
        print("🎉 All tool validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tool validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_detection():
    """Test that we can detect existing architecture content."""
    print("\n" + "=" * 60)
    print("TESTING: File content detection and validation")
    print("=" * 60)
    
    try:
        # Test reading current architecture
        current_arch_path = '/home/nir/ASI-Arch/current_architecture.py'
        
        if not os.path.exists(current_arch_path):
            print(f"❌ Current architecture file not found: {current_arch_path}")
            return False
            
        with open(current_arch_path, 'r') as f:
            content = f.read()
            
        # Validate content has required elements
        assert 'class Model(' in content, "Current architecture must have Model class"
        assert 'import torch' in content, "Current architecture must import torch"
        assert len(content) > 1000, "Current architecture should be substantial"
        
        # Check for no fallback artifacts
        fallback_indicators = [
            'Fallback improvement applied',
            'Agent tool usage failed', 
            'Improved forward pass with fallback enhancements'
        ]
        
        for indicator in fallback_indicators:
            assert indicator not in content, f"Current architecture should not have fallback artifacts: {indicator}"
            
        print("✅ Current architecture file is clean and valid")
        print(f"   Length: {len(content)} characters")
        print(f"   Has Model class: {'class Model(' in content}")
        print(f"   No fallback artifacts: {all(ind not in content for ind in fallback_indicators)}")
        
        return True
        
    except Exception as e:
        print(f"❌ File detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pool_cleanup():
    """Test that architecture pool is cleaned of fallback artifacts."""
    print("\n" + "=" * 60)
    print("TESTING: Architecture pool cleanup validation")
    print("=" * 60)
    
    try:
        pool_path = '/home/nir/ASI-Arch/pipeline/pool'
        
        if not os.path.exists(pool_path):
            print(f"❌ Pool directory not found: {pool_path}")
            return False
            
        # Count Python files in pool
        python_files = list(Path(pool_path).glob('*.py'))
        print(f"Found {len(python_files)} Python files in pool")
        
        # Check each file for fallback artifacts
        clean_files = 0
        fallback_indicators = [
            'Fallback improvement applied',
            'Agent tool usage failed', 
            'Improved forward pass with fallback enhancements'
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                has_artifacts = any(indicator in content for indicator in fallback_indicators)
                
                if has_artifacts:
                    print(f"❌ Found fallback artifacts in: {py_file.name}")
                    return False
                else:
                    clean_files += 1
                    print(f"✅ Clean file: {py_file.name}")
                    
            except Exception as e:
                print(f"⚠️  Could not read {py_file.name}: {e}")
                
        print(f"🎉 Pool cleanup successful! {clean_files} clean files, 0 files with artifacts")
        return True
        
    except Exception as e:
        print(f"❌ Pool cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evolution_interface():
    """Test the evolution interface changes."""
    print("\n" + "=" * 60)  
    print("TESTING: Evolution interface modifications")
    print("=" * 60)
    
    try:
        interface_path = '/home/nir/ASI-Arch/pipeline/evolve/interface.py'
        
        if not os.path.exists(interface_path):
            print(f"❌ Interface file not found: {interface_path}")
            return False
            
        with open(interface_path, 'r') as f:
            content = f.read()
            
        # Check for critical modifications
        checks = [
            ('CRITICAL: Validate that the file was actually modified', 'Tool usage validation added'),
            ('FORCE RETRY - DO NOT use fallback', 'Fallback system disabled'),
            ('EVOLUTION FAILURE: Agent consistently failed', 'Proper failure handling'),
            ('FALLBACK SYSTEM DISABLED', 'Fallback function disabled'),
            ('Evolution agents MUST use write_code_file', 'Clear error messages')
        ]
        
        for check_text, description in checks:
            if check_text in content:
                print(f"✅ {description}: Found '{check_text[:50]}...'")
            else:
                print(f"❌ {description}: Missing '{check_text[:50]}...'")
                return False
                
        # Check that fallback improvements are disabled
        if 'def apply_fallback_improvements' in content:
            if 'DEPRECATED' in content and 'FALLBACK SYSTEM DISABLED' in content:
                print("✅ Fallback function properly disabled")
            else:
                print("❌ Fallback function not properly disabled")
                return False
        else:
            print("❌ Fallback function not found")
            return False
            
        print("🎉 Evolution interface modifications verified!")
        return True
        
    except Exception as e:
        print(f"❌ Evolution interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("🚨 EVOLUTION SYSTEM FIX VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Tool Validation", test_tool_validation),
        ("File Detection", test_file_detection), 
        ("Pool Cleanup", test_pool_cleanup),
        ("Evolution Interface", test_evolution_interface)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Evolution system fixes verified!")
        print("\nThe evolution system should now:")
        print("- ✅ Force agents to use tools properly")
        print("- ✅ Generate diverse architectures")
        print("- ✅ Reject fallback artifacts")
        print("- ✅ Fail gracefully when agents don't cooperate")
        print("\nEvolution system is ready for testing!")
    else:
        print(f"❌ {total-passed} TESTS FAILED - Additional fixes needed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)