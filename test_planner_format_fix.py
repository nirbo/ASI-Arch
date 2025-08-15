#!/usr/bin/env python3
"""Test planner JSON format conversion."""

import json
import sys
sys.path.insert(0, '/home/nir/ml-tools/ASI-Arch/pipeline')

from agents_config import HarmonyAwareAsyncOpenAI

def test_planner_format_conversion():
    """Test that planner content gets formatted correctly."""
    
    client = HarmonyAwareAsyncOpenAI(api_key="test")
    
    # Test content that matches the error case
    wrong_json = '{"experience": "analysisWe need to produce code for DeltaNet architecture implementing the hybrid linear attention + hierarchical reasoning modules. Use write_code_file tool to write a file (maybe delta_net.py). The user wants the code to be implemented using einops.rearrange() for reshaping. Must be O(N log N) or better. Support any batch size. End with \\"Model = DeltaNet\\"."}'
    
    # Test the formatting function for planner
    result = client._format_content_for_agent(wrong_json, "planner")
    
    print(f"Input JSON (wrong format): {wrong_json[:100]}...")
    print(f"Converted result: {result}")
    
    if result:
        try:
            parsed_result = json.loads(result)
            if isinstance(parsed_result, dict) and "name" in parsed_result and "motivation" in parsed_result:
                print("✅ SUCCESS: Wrong JSON was converted to planner format!")
                print(f"Name: {parsed_result.get('name', 'N/A')}")
                print(f"Motivation: {parsed_result.get('motivation', 'N/A')[:100]}...")
                return True
            else:
                print(f"❌ FAILED: Result is not in planner format: {list(parsed_result.keys())}")
                return False
        except json.JSONDecodeError as e:
            print(f"❌ FAILED: Result is not valid JSON: {e}")
            return False
    else:
        print("❌ FAILED: No result returned from formatting")
        return False

if __name__ == "__main__":
    success = test_planner_format_conversion()
    sys.exit(0 if success else 1)