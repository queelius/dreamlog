#!/usr/bin/env python3
"""
Test and experiment with prompt generation for LLMs.
This helps us understand what prompts work best for getting
proper S-expression/prefix notation responses.
"""

import json
import urllib.request
from typing import List, Dict, Any

OLLAMA_HOST = "192.168.0.225"
OLLAMA_PORT = 11434

def test_prompt(prompt: str, model: str = "phi4-mini-reasoning:latest") -> str:
    """Send a prompt to Ollama and return raw response."""
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 200
        }
    }).encode('utf-8')
    
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', 'No response')
    except Exception as e:
        return f"Error: {e}"

def parse_response(response: str) -> Dict[str, Any]:
    """Try to parse the response and extract facts/rules."""
    import re
    
    result = {
        "raw": response,
        "json_found": False,
        "facts": [],
        "rules": [],
        "s_expressions": []
    }
    
    # Look for JSON
    json_pattern = r'\[[\s\S]*?\]'
    json_matches = re.findall(json_pattern, response)
    
    for match in json_matches:
        try:
            data = json.loads(match)
            result["json_found"] = True
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, list) and len(item) > 0:
                        if item[0] == "fact":
                            result["facts"].append(item[1:])
                        elif item[0] == "rule":
                            result["rules"].append(item[1:])
            break
        except:
            pass
    
    # Look for S-expressions
    sexp_pattern = r'\([^()]*(?:\([^()]*\)[^()]*)*\)'
    sexps = re.findall(sexp_pattern, response)
    result["s_expressions"] = sexps
    
    return result

def experiment_1_basic_formats():
    """Test different prompt formats for basic rule generation."""
    print("=" * 60)
    print("Experiment 1: Basic Format Testing")
    print("=" * 60)
    
    prompts = [
        # Format 1: Direct S-expression request
        """Generate a Prolog rule in S-expression notation.
Query: (grandparent john X)
Rule needed: grandparent defined as parent of parent
Return S-expression format like: (rule (grandparent X Z) ((parent X Y) (parent Y Z)))
Response:""",

        # Format 2: JSON with S-expressions
        """Generate a logic rule for (grandparent john X).
Return JSON with S-expression notation:
[["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
JSON only:""",

        # Format 3: Lisp-style emphasis
        """You are a Lisp/Prolog expert. Generate a rule in prefix notation.
Query: (grandparent john X)
Context: We have facts (parent john mary) and (parent mary alice)
Generate the rule in this exact format:
[["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
Response:""",

        # Format 4: Step by step
        """Step 1: Understand that (grandparent X Z) means X is grandparent of Z
Step 2: This happens when X is parent of Y and Y is parent of Z
Step 3: Write this as JSON array:
[["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
Now generate this rule:""",

        # Format 5: Example-driven
        """Example rule in our format:
[["rule", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"]]]]

Now generate a similar rule for (grandparent X Z):""",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i} ---")
        print(prompt[:100] + "..." if len(prompt) > 100 else prompt)
        
        response = test_prompt(prompt)
        parsed = parse_response(response)
        
        print(f"\nRaw response: {response[:200]}...")
        print(f"JSON found: {parsed['json_found']}")
        print(f"Rules extracted: {parsed['rules']}")
        print(f"S-expressions found: {parsed['s_expressions'][:3]}")
        print("-" * 40)

def experiment_2_context_inclusion():
    """Test how to best include context."""
    print("\n" + "=" * 60)
    print("Experiment 2: Context Inclusion")
    print("=" * 60)
    
    contexts = [
        # Context 1: S-expression facts
        """Facts in KB:
(parent john mary)
(parent mary alice)""",

        # Context 2: JSON prefix
        """Facts in KB:
["parent", "john", "mary"]
["parent", "mary", "alice"]""",

        # Context 3: Natural language
        """Known facts:
- john is parent of mary
- mary is parent of alice""",
    ]
    
    base_prompt = """Generate rule for query: (grandparent john X)

{context}

Return JSON: [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
Response:"""
    
    for i, context in enumerate(contexts, 1):
        print(f"\n--- Context Style {i} ---")
        prompt = base_prompt.format(context=context)
        
        response = test_prompt(prompt)
        parsed = parse_response(response)
        
        print(f"Context: {context[:50]}...")
        print(f"JSON found: {parsed['json_found']}")
        print(f"Rules: {parsed['rules']}")
        print("-" * 40)

def experiment_3_different_models():
    """Test same prompt with different models."""
    print("\n" + "=" * 60)
    print("Experiment 3: Model Comparison")
    print("=" * 60)
    
    models = [
        "phi4-mini-reasoning:latest",
        "qwen3:4b",
        "gemma3:4b",
        "llama3.2:latest",
    ]
    
    prompt = """Generate Prolog rule.
Query: (grandparent john X)
Output format: [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
JSON:"""
    
    for model in models:
        print(f"\n--- Model: {model} ---")
        response = test_prompt(prompt, model)
        parsed = parse_response(response)
        
        print(f"Response: {response[:100]}...")
        print(f"JSON found: {parsed['json_found']}")
        print(f"Rules: {parsed['rules']}")
        print("-" * 40)

def experiment_4_creative_predicates():
    """Test generation of non-standard predicates."""
    print("\n" + "=" * 60)
    print("Experiment 4: Creative Predicates")
    print("=" * 60)
    
    queries = [
        "(ancestor john X)",
        "(sibling alice X)",
        "(cousin john X)",
        "(related X Y)",
    ]
    
    for query in queries:
        print(f"\n--- Query: {query} ---")
        
        prompt = f"""Generate a logical rule for: {query}
Context: We have (parent ...) facts in the knowledge base.
Think about what this predicate means and create an appropriate rule.
Format: [["rule", [predicate, vars...], [[body_pred, vars...], ...]]]
JSON response:"""
        
        response = test_prompt(prompt)
        parsed = parse_response(response)
        
        print(f"Response: {response[:150]}...")
        print(f"Rules generated: {parsed['rules']}")
        print("-" * 40)

def experiment_5_compression_tasks():
    """Test prompts for knowledge compression/optimization."""
    print("\n" + "=" * 60)
    print("Experiment 5: Compression/Optimization")
    print("=" * 60)
    
    prompt = """Given these rules:
[["rule", ["father", "X", "Y"], [["parent", "X", "Y"], ["male", "X"]]]]
[["rule", ["mother", "X", "Y"], [["parent", "X", "Y"], ["female", "X"]]]]

Can you create a more general rule that captures both patterns?
Return in same JSON format:"""
    
    response = test_prompt(prompt)
    parsed = parse_response(response)
    
    print(f"Response: {response[:200]}...")
    print(f"Rules: {parsed['rules']}")

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║           Prompt Generation Testing Suite               ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    print("Select experiment:")
    print("1. Basic Format Testing - Try different prompt formats")
    print("2. Context Inclusion - Test context formatting")
    print("3. Model Comparison - Same prompt, different models")
    print("4. Creative Predicates - Test non-standard predicates")
    print("5. Compression Tasks - Test optimization prompts")
    print("6. Run All")
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == "1":
        experiment_1_basic_formats()
    elif choice == "2":
        experiment_2_context_inclusion()
    elif choice == "3":
        experiment_3_different_models()
    elif choice == "4":
        experiment_4_creative_predicates()
    elif choice == "5":
        experiment_5_compression_tasks()
    elif choice == "6":
        experiment_1_basic_formats()
        experiment_2_context_inclusion()
        experiment_3_different_models()
        experiment_4_creative_predicates()
        experiment_5_compression_tasks()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()