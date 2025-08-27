"""
JSON Sanitization Utilities

This module provides utilities to sanitize JSON strings by converting
Unicode characters to ASCII equivalents, preventing JSON parsing errors
caused by agents generating Unicode characters like en-dashes, em-dashes,
and smart quotes.
"""

import re
import json
from typing import Any, Dict


def sanitize_json_string(json_str: str) -> str:
    """
    Sanitize a JSON string by converting common Unicode characters to ASCII equivalents.
    
    This function addresses the specific issue where AI agents generate Unicode characters
    that break JSON parsing, such as:
    - En-dashes (‑) and em-dashes (—) -> regular hyphens (-)
    - Smart quotes (" ") -> regular quotes (" ")
    - Other Unicode punctuation -> ASCII equivalents
    
    Args:
        json_str: The JSON string to sanitize
        
    Returns:
        The sanitized JSON string with Unicode characters converted to ASCII
    """
    if not isinstance(json_str, str):
        return json_str
        
    # Convert Unicode dashes to regular hyphens
    json_str = json_str.replace('\u2011', '-')  # Non-breaking hyphen
    json_str = json_str.replace('\u2012', '-')  # Figure dash
    json_str = json_str.replace('\u2013', '-')  # En dash
    json_str = json_str.replace('\u2014', '-')  # Em dash
    json_str = json_str.replace('\u2015', '-')  # Horizontal bar
    
    # Convert smart quotes to regular quotes
    json_str = json_str.replace('\u201c', '"')  # Left double quotation mark
    json_str = json_str.replace('\u201d', '"')  # Right double quotation mark
    json_str = json_str.replace('\u2018', "'")  # Left single quotation mark
    json_str = json_str.replace('\u2019', "'")  # Right single quotation mark
    
    # Convert other common Unicode punctuation
    json_str = json_str.replace('\u2026', '...')  # Horizontal ellipsis
    json_str = json_str.replace('\u2010', '-')   # Hyphen
    
    return json_str


def validate_json_ascii_only(json_str: str) -> tuple[bool, str]:
    """
    Validate that a JSON string contains only ASCII characters.
    
    Args:
        json_str: The JSON string to validate
        
    Returns:
        A tuple of (is_valid, error_message). If is_valid is False,
        error_message contains details about the first non-ASCII character found.
    """
    try:
        # Check for non-ASCII characters
        json_str.encode('ascii')
        return True, ""
    except UnicodeEncodeError as e:
        char_pos = e.start
        problem_char = json_str[char_pos] if char_pos < len(json_str) else "unknown"
        unicode_name = f"U+{ord(problem_char):04X}"
        
        error_msg = (
            f"Non-ASCII character '{problem_char}' ({unicode_name}) found at position {char_pos}. "
            f"This will cause JSON parsing to fail."
        )
        return False, error_msg


def safe_json_loads(json_str: str, sanitize: bool = True) -> Any:
    """
    Safely load JSON with optional automatic sanitization.
    
    Args:
        json_str: The JSON string to parse
        sanitize: Whether to automatically sanitize Unicode characters
        
    Returns:
        The parsed JSON object
        
    Raises:
        json.JSONDecodeError: If JSON parsing fails even after sanitization
        ValueError: If non-ASCII characters are found and sanitize=False
    """
    original_str = json_str
    
    if sanitize:
        json_str = sanitize_json_string(json_str)
    else:
        # Validate ASCII-only if not sanitizing
        is_valid, error_msg = validate_json_ascii_only(json_str)
        if not is_valid:
            raise ValueError(f"JSON validation failed: {error_msg}")
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # If sanitization was applied and it still failed, provide more context
        if sanitize and original_str != json_str:
            raise json.JSONDecodeError(
                f"{str(e)} (Note: String was sanitized from Unicode to ASCII, "
                f"original length: {len(original_str)}, sanitized length: {len(json_str)})",
                json_str,
                e.pos
            ) from e
        else:
            raise


def create_sanitized_json_parser(original_parser_func):
    """
    Decorator to wrap existing JSON parsing functions with sanitization.
    
    Args:
        original_parser_func: The original JSON parsing function to wrap
        
    Returns:
        A wrapped function that sanitizes input before parsing
    """
    def sanitized_parser(json_str: str, *args, **kwargs):
        sanitized_str = sanitize_json_string(json_str)
        return original_parser_func(sanitized_str, *args, **kwargs)
    
    return sanitized_parser


# Pre-configured sanitized JSON loader
sanitized_json_loads = create_sanitized_json_parser(json.loads)