# Unicode JSON Parsing Error Fix Summary

## Problem Description

The ASI-Arch pipeline was experiencing JSON parsing failures due to AI agents generating Unicode characters instead of ASCII equivalents. Specifically:

- **Error**: `Invalid JSON: expected ',' or '}' at line 1 column 887`
- **Root Cause**: Unicode en-dashes (‑) were being generated instead of regular hyphens (-)
- **Impact**: Pipeline failures in the planner agent and potentially other agents

## Solution Overview

We implemented a two-pronged approach:

1. **Preventive**: Enhanced agent prompts with explicit ASCII-only instructions
2. **Reactive**: JSON sanitization system that converts Unicode to ASCII before parsing

## Implementation Details

### 1. Enhanced Agent Prompts

**File**: `/home/nir/ml-tools/ASI-Arch/pipeline/evolve/model/planner.py`

**Changes**:
- Added explicit "CRITICAL: USE ONLY ASCII CHARACTERS" instructions
- Specified "NO UNICODE, NO EM-DASHES, NO EN-DASHES, NO SMART QUOTES"
- Added detailed encoding rules for JSON output

### 2. JSON Sanitization System

**New Files Created**:

#### `/home/nir/ml-tools/ASI-Arch/pipeline/utils/json_sanitizer.py`
- `sanitize_json_string()`: Converts Unicode characters to ASCII equivalents
- `validate_json_ascii_only()`: Checks for non-ASCII characters with detailed error messages
- `safe_json_loads()`: JSON parser with automatic sanitization
- `sanitized_json_loads`: Pre-configured sanitized JSON loader

**Unicode Character Mappings**:
- En-dashes (‑, -, etc.) → Regular hyphens (-)
- Em-dashes (—) → Regular hyphens (-)
- Smart quotes (", ") → Regular quotes (")
- Smart apostrophes (', ') → Regular apostrophes (')
- Ellipsis (…) → Three dots (...)

#### `/home/nir/ml-tools/ASI-Arch/pipeline/utils/agent_wrapper.py`
- `sanitized_validate_json()`: Replacement for agents library's JSON validation
- `apply_global_json_sanitization()`: Monkey-patches the agents library
- `SanitizedAgentOutputSchema`: Extended schema with sanitization

### 3. Global Application

**File**: `/home/nir/ml-tools/ASI-Arch/pipeline/pipeline.py`

**Changes**:
- Added import for `apply_global_json_sanitization`
- Applied global sanitization at pipeline startup

## Validation and Testing

### Test Suite: `/home/nir/ml-tools/ASI-Arch/pipeline/utils/json_test_simple.py`

**Test Results**:
✅ Unicode en-dash replacement: `memory‑as‑context` → `memory-as-context`
✅ Unicode em-dash replacement: `test—dash` → `test-dash`  
✅ Mixed Unicode characters: `unicode‑and—dashes` → `unicode-and-dashes`
✅ Valid JSON passes through unchanged
✅ Malformed JSON still fails appropriately (as expected)

## Error Analysis: Before vs After

### Before Fix
```
Error: Invalid JSON: expected ',' or '}' at line 1 column 887
Input: "memory‑as‑context (MAC)"
Character at position 887: Unicode en-dash (‑)
```

### After Fix
```
Input: "memory‑as‑context (MAC)" 
↓ (Automatic sanitization)
Output: "memory-as-context (MAC)"
✅ JSON parsing succeeds
```

## Benefits

1. **Immediate Problem Resolution**: Fixes the specific Unicode JSON parsing error
2. **Comprehensive Coverage**: Handles multiple types of Unicode characters
3. **Backward Compatibility**: Existing valid JSON continues to work
4. **Clear Error Messages**: Provides detailed feedback on Unicode issues
5. **Global Application**: Fixes the issue for all agents in the pipeline
6. **Performance**: Minimal overhead - only processes when Unicode is detected

## Technical Approach

### Preventive Measures (Prompt Engineering)
- Explicit ASCII-only instructions in agent prompts
- Clear examples of prohibited characters
- Emphasis on using regular punctuation

### Reactive Measures (JSON Sanitization)
- Automatic Unicode-to-ASCII conversion
- Integration at the agents library level via monkey-patching
- Comprehensive logging and error reporting
- Fallback handling for malformed JSON

## Files Modified/Created

### Modified Files:
1. `/home/nir/ml-tools/ASI-Arch/pipeline/evolve/model/planner.py` - Enhanced prompts
2. `/home/nir/ml-tools/ASI-Arch/pipeline/pipeline.py` - Applied global sanitization

### New Files:
1. `/home/nir/ml-tools/ASI-Arch/pipeline/utils/json_sanitizer.py` - Core sanitization utilities
2. `/home/nir/ml-tools/ASI-Arch/pipeline/utils/agent_wrapper.py` - Agent integration wrapper
3. `/home/nir/ml-tools/ASI-Arch/pipeline/utils/json_test_simple.py` - Test suite

## Future Considerations

1. **Monitoring**: Add metrics to track Unicode character frequency
2. **Model Training**: Consider fine-tuning models to avoid Unicode generation
3. **Extended Coverage**: Add more Unicode character mappings as needed
4. **Performance Optimization**: Cache sanitization results if performance becomes an issue

## Conclusion

The implemented solution provides both immediate relief from the Unicode JSON parsing issue and a robust foundation for handling similar character encoding problems in the future. The two-pronged approach ensures that agents are less likely to generate problematic characters while providing automatic remediation when they do.