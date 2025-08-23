#!/usr/bin/env python3
"""
Simple test to verify Ollama is working with DreamLog
"""

import urllib.request
import json

OLLAMA_HOST = "192.168.0.225"
OLLAMA_PORT = 11434
MODEL = "phi4-mini-reasoning:latest"

def test_ollama_direct():
    """Test Ollama directly without DreamLog"""
    print("Testing direct Ollama connection...")
    
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    
    prompt = """Return a JSON array with a Prolog rule for grandparent.
Format: [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
JSON:"""
    
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 100
        }
    }).encode('utf-8')
    
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    
    try:
        print(f"Calling Ollama at {url}...")
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            llm_response = result.get('response', 'No response')
            print(f"\nLLM Response:\n{llm_response}")
            
            # Try to parse the response
            print("\nTrying to parse as JSON...")
            try:
                # Find JSON in response
                import re
                json_matches = re.findall(r'\[.*?\]', llm_response, re.DOTALL)
                if json_matches:
                    parsed = json.loads(json_matches[0])
                    print(f"Parsed successfully: {parsed}")
                else:
                    print("No JSON found in response")
            except Exception as e:
                print(f"Parse error: {e}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ollama_direct()