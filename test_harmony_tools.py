#!/usr/bin/env python3
"""
Test script to verify harmony tool calling works properly
"""

import logging
import asyncio
from pipeline.agents_config import HarmonyAwareAsyncOpenAI
from pipeline.config import Config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_harmony_tool_calling():
    """Test that harmony encoding properly handles tools"""
    
    # Create the harmony-aware client
    client = HarmonyAwareAsyncOpenAI(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_BASE_URL
    )
    
    # Define test tools (similar to the actual tools)
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "read_code_file",
                "description": "Read a code file and return its contents",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "write_code_file",
                "description": "Write content to a code file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The code content to write"
                        }
                    },
                    "required": ["content"]
                }
            }
        }
    ]
    
    # Test message that should trigger planner behavior
    messages = [
        {
            "role": "system",
            "content": "You are an Architecture Designer who must use tools before providing final output."
        },
        {
            "role": "user", 
            "content": "Create a new delta_net architecture. You must read the current code first, then implement improvements, then provide a JSON response with name and motivation fields."
        }
    ]
    
    try:
        logger.info("🧪 Testing harmony tool calling...")
        
        # This should trigger harmony encoding with tools
        response = await client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            tools=test_tools,
            max_tokens=2000,
            temperature=0.7
        )
        
        logger.info("✅ Response received!")
        logger.info(f"Response content: {response.choices[0].message.content}")
        
        # Check if response mentions tools
        content = response.choices[0].message.content
        if 'read_code_file' in content or 'write_code_file' in content:
            logger.info("✅ Tool usage detected in response!")
        else:
            logger.warning("❌ No tool usage detected in response")
            
        return response
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.exception("Full error traceback")
        return None

if __name__ == "__main__":
    asyncio.run(test_harmony_tool_calling())