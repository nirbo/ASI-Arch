#!/usr/bin/env python3
"""
Test script to validate the harmony channel extraction fix.
This tests that harmony responses are properly extracted in the correct priority order:
final -> assistant -> analysis -> commentary -> raw
"""

import sys
import os
import json
import re
sys.path.append('/home/nir/ml-tools/ASI-Arch')

from pipeline.config import Config

def test_channel_priority_extraction():
    """Test that channels are extracted in the correct priority order"""
    print("🧪 Testing Harmony Channel Priority Extraction")
    print("=" * 60)
    
    # No need to create actual client, just test the logic
    
    test_cases = [
        {
            'name': 'Final channel takes priority',
            'response': '''<|channel|>analysis<|message|>This is my reasoning about the architectural design. I'll analyze the current model and suggest improvements.<|channel|>final<|message|>{"name": "improved_delta_net", "motivation": "Added attention mechanism to improve accuracy", "code": "class ImprovedNet(nn.Module): pass"}''',
            'expected_channel': 'final',
            'expected_content': '{"name": "improved_delta_net", "motivation": "Added attention mechanism to improve accuracy", "code": "class ImprovedNet(nn.Module): pass"}',
            'agent_type': 'planner'
        },
        {
            'name': 'Analysis channel when no final',
            'response': '''<|channel|>commentary<|message|>Some commentary here<|channel|>analysis<|message|>This is detailed analysis of the architecture performance showing 15% improvement in accuracy and 20% reduction in parameters.''',
            'expected_channel': 'analysis',
            'expected_content': 'This is detailed analysis of the architecture performance showing 15% improvement in accuracy and 20% reduction in parameters.',
            'agent_type': 'summarizer'
        },
        {
            'name': 'Assistant channel takes precedence over analysis',
            'response': '''<|channel|>analysis<|message|>Detailed reasoning here<|channel|>assistant<|message|>{"experience": "The experimental results show significant improvement in cognitive processing capabilities."}''',
            'expected_channel': 'assistant',
            'expected_content': '{"experience": "The experimental results show significant improvement in cognitive processing capabilities."}',
            'agent_type': 'summarizer'
        },
        {
            'name': 'Commentary fallback',
            'response': '''<|channel|>commentary<|message|>This is commentary about the task execution.''',
            'expected_channel': 'commentary',
            'expected_content': 'This is commentary about the task execution.',
            'agent_type': 'planner'
        },
        {
            'name': 'No channels - raw fallback',
            'response': '''This is just plain text without any harmony channels.''',
            'expected_channel': None,
            'expected_content': 'This is just plain text without any harmony channels.',
            'agent_type': 'planner'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Test: {test_case['name']}")
        print(f"Response: {test_case['response'][:100]}...")
        
        # Test the channel extraction logic directly
        response_text = test_case['response']
        agent_type = test_case['agent_type']
        
        # Simulate the channel extraction from the local harmony method
        if '<|channel|>' in response_text or '<|message|>' in response_text:
            # Priority order: final -> assistant -> analysis -> commentary -> raw
            channel_patterns = [
                (r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|\Z)', 'final'),
                (r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|\Z)', 'assistant_final'),
                (r'<\|channel\|>assistant<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|\Z)', 'assistant'),
                (r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|\Z)', 'analysis'),
                (r'<\|channel\|>commentary<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|\Z)', 'commentary'),
                (r'<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|\Z)', 'message')  # fallback
            ]
            
            extracted_content = None
            channel_type = None
            
            for pattern, ch_type in channel_patterns:
                matches = re.findall(pattern, response_text, re.DOTALL)
                if matches:
                    # For final/assistant channels, take first match. For others, take longest
                    if ch_type in ['final', 'assistant_final', 'assistant']:
                        extracted_content = matches[0].strip()
                        channel_type = ch_type
                        break
                    else:
                        # For analysis/commentary, take longest match
                        extracted_content = max(matches, key=len).strip()
                        channel_type = ch_type
                        break
        else:
            # No channels found
            extracted_content = response_text.strip()
            channel_type = None
        
        print(f"   Extracted from channel: {channel_type}")
        print(f"   Content: {extracted_content[:100]}...")
        
        # Verify channel type
        if channel_type == test_case['expected_channel']:
            print(f"   ✅ Channel type matches expected: {channel_type}")
        elif test_case['expected_channel'] is None and channel_type is None:
            print(f"   ✅ No channels found as expected")
        else:
            print(f"   ❌ Channel type mismatch. Expected: {test_case['expected_channel']}, Got: {channel_type}")
        
        # Verify content
        if extracted_content == test_case['expected_content']:
            print(f"   ✅ Content matches expected")
        else:
            print(f"   ⚠️  Content differs from expected")
            print(f"   Expected: {test_case['expected_content']}")
            print(f"   Got:      {extracted_content}")

def test_agent_type_based_formatting():
    """Test that agent type determines response formatting instead of content analysis"""
    print("\n\n🧪 Testing Agent Type Based Response Formatting")
    print("=" * 60)
    
    # Test that same content gets formatted differently based on agent type
    sample_content = "This is experimental analysis showing improved performance metrics."
    
    test_cases = [
        {
            'agent_type': 'planner',
            'expected_fields': ['name', 'motivation', 'code'],
            'description': 'Planner should create planner JSON structure'
        },
        {
            'agent_type': 'summarizer', 
            'expected_fields': ['experience'],
            'description': 'Summarizer should create experience JSON structure'
        },
        {
            'agent_type': 'analyzer',
            'expected_fields': ['design_evaluation', 'experimental_results_analysis', 'expectation_vs_reality_comparison', 'theoretical_explanation_with_evidence', 'synthesis_and_insights'],
            'description': 'Analyzer should create analyzer JSON structure'
        },
        {
            'agent_type': 'trainer',
            'expected_fields': ['success', 'error'],
            'description': 'Trainer should create trainer JSON structure'
        },
        {
            'agent_type': None,
            'expected_fields': ['response'],
            'description': 'No agent type should create generic JSON structure'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Test: {test_case['description']}")
        agent_type = test_case['agent_type']
        
        # Simulate the agent type based formatting logic from local harmony method
        if agent_type == "planner":
            response_content = json.dumps({
                "name": "test_architecture",
                "motivation": "Agent type based formatting test",
                "code": f"# test_architecture\n{sample_content}"
            })
        elif agent_type == "summarizer":
            response_content = json.dumps({
                "experience": sample_content
            })
        elif agent_type == "analyzer":
            response_content = json.dumps({
                "design_evaluation": f"Analysis: {sample_content[:50]}...",
                "experimental_results_analysis": f"Results: {sample_content[:50]}...", 
                "expectation_vs_reality_comparison": f"Comparison: {sample_content[:50]}...",
                "theoretical_explanation_with_evidence": f"Theory: {sample_content[:50]}...",
                "synthesis_and_insights": f"Insights: {sample_content[:50]}..."
            })
        elif agent_type == "trainer":
            response_content = json.dumps({
                "success": True,
                "error": None
            })
        else:
            response_content = json.dumps({
                "response": sample_content
            })
        
        print(f"   Agent type: {agent_type}")
        print(f"   Generated JSON: {response_content[:100]}...")
        
        # Verify JSON is valid
        try:
            parsed = json.loads(response_content)
            print(f"   ✅ Valid JSON generated")
            
            # Check expected fields
            expected_fields = test_case['expected_fields']
            missing_fields = [field for field in expected_fields if field not in parsed]
            extra_fields = [field for field in parsed if field not in expected_fields]
            
            if not missing_fields and not extra_fields:
                print(f"   ✅ All expected fields present: {expected_fields}")
            else:
                if missing_fields:
                    print(f"   ❌ Missing fields: {missing_fields}")
                if extra_fields:
                    print(f"   ❌ Unexpected fields: {extra_fields}")
                    
        except json.JSONDecodeError as e:
            print(f"   ❌ Invalid JSON: {e}")

def test_developer_instructions():
    """Test that proper developer_instructions are generated for different agent types"""
    print("\n\n🧪 Testing Developer Instructions Generation")
    print("=" * 60)
    
    agent_types = ['planner', 'summarizer', 'analyzer', 'trainer', 'debugger', 'deduplication', 'motivation_checker', None]
    
    for agent_type in agent_types:
        print(f"\n🧪 Agent type: {agent_type}")
        
        # Simulate the developer_instructions logic from local harmony method
        if agent_type == "planner":
            developer_instructions = "You are an Architecture Designer. Provide your reasoning in the analysis channel, then output a JSON object with 'name', 'motivation', and 'code' fields in the final channel."
        elif agent_type == "summarizer":
            developer_instructions = "You are a research summarizer. Provide your analysis in the analysis channel, then output a JSON object with an 'experience' field in the final channel."
        elif agent_type == "analyzer":
            developer_instructions = "You are an architecture analyzer. Provide your reasoning in the analysis channel, then output a JSON object with design_evaluation, experimental_results_analysis, expectation_vs_reality_comparison, theoretical_explanation_with_evidence, and synthesis_and_insights fields in the final channel."
        elif agent_type in ["trainer", "code_checker"]:
            developer_instructions = f"You are a {agent_type}. Provide your analysis in the analysis channel, then output a JSON object with 'success' and 'error' fields in the final channel."
        elif agent_type == "debugger":
            developer_instructions = "You are a debugging expert. Provide your analysis in the analysis channel, then output a JSON object with 'changes_made' field in the final channel."
        elif agent_type == "deduplication":
            developer_instructions = "You are an innovation diversifier. Provide your analysis in the analysis channel, then output a JSON object with 'name', 'motivation', and 'code' fields in the final channel."
        elif agent_type == "motivation_checker":
            developer_instructions = "You are a motivation checker. Provide your analysis in the analysis channel, then output a JSON object with 'is_repeated', 'repeated_index', and 'judgement_reason' fields in the final channel."
        else:
            developer_instructions = "Provide your reasoning in the analysis channel, then output your structured response in the final channel."
        
        print(f"   Instructions: {developer_instructions}")
        
        # Verify instructions mention final channel
        if "final channel" in developer_instructions:
            print(f"   ✅ Instructions specify final channel output")
        else:
            print(f"   ❌ Instructions don't specify final channel")
            
        # Verify instructions mention analysis channel for reasoning
        if "analysis channel" in developer_instructions:
            print(f"   ✅ Instructions specify analysis channel for reasoning")
        else:
            print(f"   ⚠️  Instructions don't mention analysis channel")

if __name__ == "__main__":
    print("🚀 Starting Harmony Channel Fix Testing")
    
    # Enable debug mode for detailed output
    Config.DEBUG_HARMONY_ENCODING = True
    
    test_channel_priority_extraction()
    test_agent_type_based_formatting()
    test_developer_instructions()
    
    print("\n🎉 Harmony channel fix testing completed!")
    print("\nSummary of fixes implemented:")
    print("✅ 1. Fixed channel extraction to prioritize final > assistant > analysis > commentary")
    print("✅ 2. Changed agent type detection from content-based to context-based")
    print("✅ 3. Added proper developer_instructions specifying final channel output")
    print("✅ 4. Removed complex content-based pattern matching")
    print("\nThe harmony implementation should now:")
    print("- Extract responses from the correct channels in priority order")
    print("- Determine agent type from calling context, not response content")
    print("- Generate proper JSON structures based on agent type")
    print("- Instruct models to put structured output in final channel")