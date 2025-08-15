#!/usr/bin/env python3
"""Test harmony extraction path for planner JSON fix."""

import json
import sys
sys.path.insert(0, '/home/nir/ml-tools/ASI-Arch/pipeline')

from agents_config import HarmonyAwareAsyncOpenAI

def test_harmony_planner_extraction():
    """Test that planner JSON gets processed through the harmony extraction path."""
    
    client = HarmonyAwareAsyncOpenAI(api_key="test")
    
    # Simulate the wrong JSON content that would come from harmony extraction
    wrong_json_content = '{"experience": "analysisWe need to produce code for DeltaNet architecture implementing the hybrid linear attention + hierarchical reasoning modules. Use write_code_file tool to write a file (maybe delta_net.py). The user wants the code to be implemented using einops.rearrange() for reshaping. Must be O(N log N) or better. Support any batch size. End with \\"Model = DeltaNet\\"."}'
    
    # Simulate the harmony extraction path - this is what would happen in _extract_final_content_from_harmony
    # The key insight is that the content starts with '{' so it goes through the JSON validation path
    agent_type = "planner"
    extracted_content = wrong_json_content
    
    # This simulates the exact logic from the harmony extraction
    if agent_type and extracted_content:
        if not extracted_content.strip().startswith('{'):
            # This path wouldn't be taken since our content starts with '{'
            result = client._format_content_for_agent(extracted_content, agent_type)
        else:
            # This is the path that should now call _format_content_for_agent for planner
            if agent_type == "planner":
                print(f"🔧 PLANNER: Processing JSON-like content (simulated)")
                result = client._format_content_for_agent(extracted_content, agent_type)
            else:
                # This is the old path that would return wrong content as-is
                result = extracted_content
    else:
        result = extracted_content or ""
    
    print(f"Input JSON (wrong format): {wrong_json_content[:100]}...")
    print(f"Converted result: {result}")
    
    if result:
        try:
            parsed_result = json.loads(result)
            if isinstance(parsed_result, dict) and "name" in parsed_result and "motivation" in parsed_result:
                print("✅ SUCCESS: Harmony extraction path converted wrong JSON to planner format!")
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
        print("❌ FAILED: No result returned from extraction")
        return False

if __name__ == "__main__":
    success = test_harmony_planner_extraction()
    sys.exit(0 if success else 1)