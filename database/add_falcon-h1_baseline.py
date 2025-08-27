#!/usr/bin/env python3
"""
Add transformer baseline candidate to the database.
"""
import json
import requests

def add_falcon_h1_titans_baseline():
    """Add the transformer baseline candidate to the database."""
    
    # Load the transformer candidate
    with open('falcon-h1-titans-memory-candidate.json', 'r') as f:
        candidate_data = json.load(f)
    
    # API endpoint
    api_url = "http://localhost:8001/elements"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(api_url, headers=headers, json=candidate_data)
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Successfully added Falcon-H1-Titans baseline: {result}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to add Falcon-H1-Titans baseline: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Response content: {e.response.text}")
        return False

if __name__ == "__main__":
    add_falcon_h1_titans_baseline()
