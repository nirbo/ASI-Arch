#!/usr/bin/env python3
"""
Simple test for JSON sanitization utilities.
"""

import json
from json_sanitizer import (
    sanitize_json_string, 
    validate_json_ascii_only, 
    safe_json_loads
)


def test_unicode_replacement():
    """Test that Unicode dash replacement works."""
    print("Testing Unicode dash replacement...")
    
    # Create JSON with Unicode en-dashes (U+2011)
    test_json = '{"name":"test","text":"memory' + '\u2011' + 'as' + '\u2011' + 'context"}'
    print(f"Original with Unicode: {repr(test_json)}")
    
    # Check if it has non-ASCII characters
    is_ascii, error_msg = validate_json_ascii_only(test_json)
    print(f"Is ASCII only: {is_ascii}")
    if not is_ascii:
        print(f"ASCII validation error: {error_msg}")
    
    # Test standard JSON parsing (should fail)
    try:
        result = json.loads(test_json)
        print("ERROR: Standard JSON parsing with Unicode succeeded (shouldn't in some cases)")
        print(f"Result: {result}")
    except json.JSONDecodeError as e:
        print(f"Standard JSON parsing failed: {e}")
    
    # Test sanitization
    sanitized = sanitize_json_string(test_json)
    print(f"Sanitized: {repr(sanitized)}")
    print(f"Changed: {sanitized != test_json}")
    
    # Test sanitized parsing
    try:
        result = json.loads(sanitized)
        print(f"SUCCESS: Sanitized parsing worked! Result: {result}")
    except json.JSONDecodeError as e:
        print(f"ERROR: Sanitized parsing failed: {e}")
    
    # Test safe_json_loads
    try:
        result = safe_json_loads(test_json)
        print(f"SUCCESS: safe_json_loads worked! Result: {result}")
    except Exception as e:
        print(f"ERROR: safe_json_loads failed: {e}")


def test_malformed_structure():
    """Test handling of malformed JSON structure."""
    print("\nTesting malformed JSON structure...")
    
    # JSON with extra closing parenthesis (like the actual error)
    malformed_json = '{"name":"test","message":"some text")'
    print(f"Malformed JSON: {malformed_json}")
    
    try:
        result = json.loads(malformed_json)
        print("ERROR: Malformed JSON parsing succeeded (it shouldn't)")
    except json.JSONDecodeError as e:
        print(f"Malformed JSON parsing failed as expected: {e}")
    
    try:
        result = safe_json_loads(malformed_json)
        print("ERROR: safe_json_loads with malformed JSON succeeded (it shouldn't)")
    except Exception as e:
        print(f"safe_json_loads with malformed JSON failed as expected: {e}")


def test_comprehensive():
    """Test a comprehensive case with both Unicode and various structures."""
    print("\nTesting comprehensive cases...")
    
    test_cases = [
        ('{"valid": "json"}', "Valid JSON"),
        ('{"unicode": "test' + '\u2011' + 'dash"}', "Unicode dash"),
        ('{"unicode": "test' + '\u2014' + 'dash"}', "Unicode em-dash"), 
        ('{"malformed": "json")', "Malformed JSON"),
        ('{"mixed": "unicode' + '\u2011' + 'and' + '\u2014' + 'dashes"}', "Mixed Unicode dashes"),
    ]
    
    for test_json, description in test_cases:
        print(f"\nTesting {description}: {repr(test_json[:30])}...")
        
        # Check ASCII
        is_ascii, _ = validate_json_ascii_only(test_json)
        print(f"  ASCII-only: {is_ascii}")
        
        # Test sanitization + parsing
        try:
            result = safe_json_loads(test_json, sanitize=True)
            print(f"  SUCCESS: {result}")
        except Exception as e:
            print(f"  Expected error: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("JSON Sanitization Simple Test")
    print("=" * 50)
    
    test_unicode_replacement()
    test_malformed_structure()
    test_comprehensive()
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)