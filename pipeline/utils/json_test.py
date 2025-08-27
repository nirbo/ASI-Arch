#!/usr/bin/env python3
"""
Test script for JSON sanitization utilities.

This script tests the Unicode character sanitization functionality
to ensure it properly handles the specific error case from the ASI-Arch pipeline.
"""

import json
from json_sanitizer import (
    sanitize_json_string, 
    validate_json_ascii_only, 
    safe_json_loads,
    sanitized_json_loads
)


def test_specific_error_case():
    """Test the specific error case that was failing in the pipeline."""
    print("Testing specific error case from pipeline...")
    
    # This is the actual problematic JSON from the logs (with the malformed ending)
    problematic_json_malformed = '{"name":"H1-Titans-MAC-GATED","motivation":"By shifting the Titans memory from a simple gating mechanism (MAG) to a memory‑as‑context (MAC) paradigm, we allow the persistent key‑value slots to inform both the attention and Mamba2 branches during their forward passes. This contextual injection supplies long‑range, causally‑consistent signals that the linear‑time branches can exploit, improving long‑range dependency modeling while keeping the overall complexity linear. Coupling MAC with a gated fusion mixer gives the model learnable control over how much context each branch should use, enabling dynamic adaptation to diverse sequence patterns and further reducing redundancy between branches. The design preserves batch‑size independence, causal masking, and sub‑quadratic performance, and it respects the eval‑only write policy to maintain training stability.")'
    
    # The correctly formatted version (what it should be)  
    problematic_json = '{"name":"H1-Titans-MAC-GATED","motivation":"By shifting the Titans memory from a simple gating mechanism (MAG) to a memory‑as‑context (MAC) paradigm, we allow the persistent key‑value slots to inform both the attention and Mamba2 branches during their forward passes. This contextual injection supplies long‑range, causally‑consistent signals that the linear‑time branches can exploit, improving long‑range dependency modeling while keeping the overall complexity linear. Coupling MAC with a gated fusion mixer gives the model learnable control over how much context each branch should use, enabling dynamic adaptation to diverse sequence patterns and further reducing redundancy between branches. The design preserves batch‑size independence, causal masking, and sub‑quadratic performance, and it respects the eval‑only write policy to maintain training stability."}"
    
    print("=== Testing the MALFORMED JSON (actual error case) ===")
    print(f"Malformed JSON length: {len(problematic_json_malformed)}")
    print(f"Last 20 chars: '{problematic_json_malformed[-20:]}'")
    
    # Test standard JSON parsing (should fail due to malformed structure)
    try:
        result = json.loads(problematic_json_malformed)
        print("ERROR: Malformed JSON parsing succeeded (it shouldn't have)")
    except json.JSONDecodeError as e:
        print(f"Malformed JSON parsing failed as expected: {e}")
        print(f"Error position: {e.pos}")
    
    print("\n=== Testing the CORRECTLY FORMATTED JSON with Unicode ===")
    print(f"Original JSON length: {len(problematic_json)}")
    print(f"Last 20 chars: '{problematic_json[-20:]}'")
    
    # Check if it has non-ASCII characters
    is_ascii, error_msg = validate_json_ascii_only(problematic_json)
    print(f"Is ASCII only: {is_ascii}")
    if not is_ascii:
        print(f"ASCII validation error: {error_msg}")
    
    # Test standard JSON parsing (should fail due to Unicode)
    try:
        result = json.loads(problematic_json)
        print("ERROR: Standard JSON parsing succeeded (it shouldn't have with Unicode)")
    except json.JSONDecodeError as e:
        print(f"Standard JSON parsing failed due to Unicode: {e}")
        print(f"Error position: {e.pos}")
    
    # Debug: Let's find all Unicode characters in the string
    print("\nFinding all Unicode characters...")
    unicode_count = 0
    for i, char in enumerate(problematic_json):
        if ord(char) > 127:
            unicode_count += 1
            if unicode_count <= 5:  # Show first 5 only
                print(f"Unicode char at position {i}: '{char}' (U+{ord(char):04X})")
    if unicode_count > 5:
        print(f"... and {unicode_count - 5} more Unicode characters")
    
    # Test sanitization
    print("\nTesting sanitization...")
    sanitized = sanitize_json_string(problematic_json)
    print(f"Sanitized JSON length: {len(sanitized)}")
    print(f"Sanitization changed string: {sanitized != problematic_json}")
    
    # Test sanitized JSON parsing
    try:
        result = json.loads(sanitized)
        print("SUCCESS: Sanitized JSON parsing succeeded!")
        print(f"Parsed name: {result.get('name')}")
        print(f"Motivation length: {len(result.get('motivation', ''))}")
    except json.JSONDecodeError as e:
        print(f"ERROR: Even sanitized JSON parsing failed: {e}")
        print(f"Error at position {e.pos} in sanitized string")
    
    # Test our safe JSON loads
    print("\nTesting safe_json_loads...")
    try:
        result = safe_json_loads(problematic_json)
        print("SUCCESS: safe_json_loads succeeded!")
        print(f"Parsed name: {result.get('name')}")
        print(f"Motivation starts with: {result.get('motivation', '')[:50]}...")
    except Exception as e:
        print(f"ERROR: safe_json_loads failed: {e}")


def test_various_unicode_chars():
    """Test various Unicode characters that might cause issues."""
    print("\nTesting various Unicode characters...")
    
    test_cases = [
        ('{"text": "en‑dash"}', 'en-dash (U+2011)'),
        ('{"text": "em—dash"}', 'em-dash (U+2014)'), 
        ('{"text": ""smart quotes""}', 'smart quotes'),
        ('{"text": "\u2018smart apostrophe\u2019"}', 'smart apostrophe'),
        ('{"text": "ellipsis…"}', 'ellipsis'),
        ('{"text": "regular-hyphen"}', 'regular hyphen (should not change)'),
    ]
    
    for test_json, description in test_cases:
        print(f"\nTesting {description}:")
        print(f"Original: {test_json}")
        
        sanitized = sanitize_json_string(test_json)
        print(f"Sanitized: {sanitized}")
        print(f"Changed: {test_json != sanitized}")
        
        try:
            result = json.loads(sanitized)
            print(f"SUCCESS: Parsed as {result}")
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse: {e}")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting edge cases...")
    
    test_cases = [
        ('', 'empty string'),
        ('null', 'null value'),
        ('{}', 'empty object'),
        ('[]', 'empty array'),
        ('{"valid": "json"}', 'valid JSON'),
        ('invalid json', 'invalid JSON'),
        ('{"mixed": "en‑dash and—em‑dash"}', 'mixed Unicode dashes'),
    ]
    
    for test_input, description in test_cases:
        print(f"\nTesting {description}:")
        try:
            result = sanitized_json_loads(test_input)
            print(f"SUCCESS: {result}")
        except Exception as e:
            print(f"Expected error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("JSON Sanitization Test Suite")
    print("=" * 60)
    
    test_specific_error_case()
    test_various_unicode_chars()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)