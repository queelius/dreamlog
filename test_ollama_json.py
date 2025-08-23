#!/usr/bin/env python3
"""
Simple test to see what Ollama is actually returning
"""

import json
import urllib.request

OLLAMA_URL = "http://192.168.0.225:11434"
MODEL = "gemma3n:latest"

def test_direct_ollama():
    """Test Ollama directly to see raw output"""
    
    # Simple prompt asking for JSON
    prompt = """Generate a logic rule for grandparent relation.

Output ONLY valid JSON in this exact format:
[["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]

No other text, just the JSON array."""
    
    url = f"{OLLAMA_URL}/api/generate"
    
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # Tell Ollama we want JSON
        "options": {
            "temperature": 0.3,
            "num_predict": 200
        }
    }).encode('utf-8')
    
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    
    try:
        print(f"Sending request to {url}")
        print(f"Model: {MODEL}")
        print(f"Prompt:\n{prompt}\n")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            raw_response = result.get('response', '')
            
            print("=" * 60)
            print("RAW RESPONSE:")
            print("=" * 60)
            print(raw_response)
            print("=" * 60)
            
            # Try to parse as JSON
            try:
                parsed = json.loads(raw_response)
                print("\n✓ Valid JSON!")
                print("Parsed:", json.dumps(parsed, indent=2))
            except json.JSONDecodeError as e:
                print(f"\n✗ Invalid JSON: {e}")
                
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
                if json_match:
                    print("\nFound JSON-like content:")
                    print(json_match.group(0))
                    try:
                        parsed = json.loads(json_match.group(0))
                        print("✓ Extracted valid JSON!")
                    except:
                        print("✗ Still invalid after extraction")
                        
    except Exception as e:
        print(f"Error: {e}")


def test_without_format():
    """Test without format=json to see difference"""
    
    prompt = """Generate a logic rule for grandparent relation.

Return JSON: [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]"""
    
    url = f"{OLLAMA_URL}/api/generate"
    
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        # No format parameter
        "options": {
            "temperature": 0.3,
            "num_predict": 200
        }
    }).encode('utf-8')
    
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    
    try:
        print("\n" + "=" * 60)
        print("Testing WITHOUT format=json")
        print("=" * 60)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            raw_response = result.get('response', '')
            
            print("RAW RESPONSE:")
            print(raw_response)
            
            # Try to extract and parse JSON
            import re
            json_patterns = [
                r'\[\[.*?\]\]',  # Nested arrays
                r'```json\s*(.*?)\s*```',  # Code blocks
                r'```\s*(.*?)\s*```',
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, raw_response, re.DOTALL)
                if match:
                    try:
                        json_str = match.group(1) if '```' in pattern else match.group(0)
                        parsed = json.loads(json_str)
                        print(f"\n✓ Found valid JSON with pattern: {pattern[:20]}...")
                        print("Parsed:", json.dumps(parsed, indent=2))
                        break
                    except:
                        continue
                        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Testing Ollama JSON Generation")
    print("=" * 60)
    
    # Test 1: With format=json
    test_direct_ollama()
    
    # Test 2: Without format=json
    test_without_format()