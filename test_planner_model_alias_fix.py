#!/usr/bin/env python3
"""Test that planner properly handles wrong JSON format and includes Model alias."""

import json
import logging
from pipeline.agents_config import HarmonyAwareAsyncOpenAI

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_planner_wrong_format_conversion():
    """Test that planner converts wrong JSON format correctly."""
    client = HarmonyAwareAsyncOpenAI(api_key="test", base_url="http://test")
    
    # Test case 1: Planner receives summarizer format (experience field)
    wrong_format = json.dumps({
        "experience": "analysisWe improved the model architecture by adding attention mechanisms and optimizing the transformer layers."
    })
    
    result = client._format_content_for_agent(wrong_format, "planner")
    parsed = json.loads(result)
    
    print(f"Test 1 - Wrong format conversion:")
    print(f"  Input: {wrong_format[:100]}...")
    print(f"  Output keys: {list(parsed.keys())}")
    print(f"  Has 'name': {'name' in parsed}")
    print(f"  Has 'motivation': {'motivation' in parsed}")
    print(f"  Has 'code': {'code' in parsed}")
    
    # Check for Model alias in code
    if 'code' in parsed:
        has_model_alias = "Model = DeltaNet" in parsed['code']
        has_class_def = "class DeltaNet" in parsed['code']
        print(f"  Has 'class DeltaNet': {has_class_def}")
        print(f"  Has 'Model = DeltaNet': {has_model_alias}")
        
        if not has_model_alias:
            print("  ❌ ERROR: Missing Model alias!")
        else:
            print("  ✅ SUCCESS: Model alias present!")
    
    print()
    
    # Test case 2: Planner receives unexpected JSON format
    unexpected_format = json.dumps({
        "random_field": "Some content about the architecture design",
        "another_field": 123
    })
    
    result2 = client._format_content_for_agent(unexpected_format, "planner")
    parsed2 = json.loads(result2)
    
    print(f"Test 2 - Unexpected format conversion:")
    print(f"  Input: {unexpected_format}")
    print(f"  Output keys: {list(parsed2.keys())}")
    print(f"  Has 'name': {'name' in parsed2}")
    print(f"  Has 'motivation': {'motivation' in parsed2}")
    print(f"  Has 'code': {'code' in parsed2}")
    
    if 'code' in parsed2:
        has_model_alias = "Model = DeltaNet" in parsed2['code'] or "Model = " in parsed2['code']
        print(f"  Has Model alias: {has_model_alias}")
        
        if not has_model_alias:
            print("  ❌ ERROR: Missing Model alias!")
        else:
            print("  ✅ SUCCESS: Model alias present!")
    
    print()
    
    # Test case 3: Planner receives plain text (non-JSON)
    plain_text = """class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.layer(x)"""
    
    result3 = client._format_content_for_agent(plain_text, "planner")
    parsed3 = json.loads(result3)
    
    print(f"Test 3 - Plain text with class definition:")
    print(f"  Input: Plain Python code with class MyNet")
    print(f"  Output keys: {list(parsed3.keys())}")
    
    if 'code' in parsed3:
        has_model_alias = "Model = " in parsed3['code']
        print(f"  Has Model alias: {has_model_alias}")
        if "Model = MyNet" in parsed3['code']:
            print("  ✅ SUCCESS: Correctly added 'Model = MyNet' alias!")
        elif has_model_alias:
            print(f"  ✅ SUCCESS: Added Model alias!")
        else:
            print("  ❌ ERROR: Missing Model alias!")

if __name__ == "__main__":
    test_planner_wrong_format_conversion()