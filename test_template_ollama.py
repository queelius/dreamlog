#!/usr/bin/env python3
"""
Test the prompt template system with Ollama at 192.168.0.225
Shows how templates adapt and improve over time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dreamlog.pythonic import dreamlog
from dreamlog.prompt_template_system import PromptTemplateLibrary, QueryContext
import time

# Ollama configuration
OLLAMA_URL = "http://192.168.0.225:11434"

# Test different models to see how templates adapt
MODELS_TO_TEST = [
    "phi4-mini-reasoning",  # Good for reasoning
    "llama3.2:3b",          # Fast, general purpose
    "qwen2.5:7b",           # Good for structured output
]

def test_model_adaptation(model_name: str):
    """Test how templates work with a specific model"""
    print(f"\n{'='*60}")
    print(f"Testing with model: {model_name}")
    print(f"{'='*60}")
    
    # Create KB with LLM support
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=model_name,
        temperature=0.7
    )
    
    # Enable debug mode
    if hasattr(kb.engine, 'llm_hook'):
        kb.engine.llm_hook.debug = True
        print(f"Debug mode enabled for {model_name}")
    
    # Test 1: Simple facts
    print("\n--- Test 1: Parent-Child Relations ---")
    kb.fact("parent", "alice", "bob")
    kb.fact("parent", "bob", "charlie")
    kb.fact("parent", "alice", "diana")
    
    # Query for grandparent (should generate rule)
    print("\nQuerying: grandparent(alice, charlie)")
    start = time.time()
    results = list(kb.query("grandparent", "alice", "charlie"))
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    if results:
        print(f"✓ Success: {len(results)} result(s)")
        print(f"  Generated rules: {len(kb.engine.kb.rules)}")
    else:
        print("✗ Failed: No results")
    
    # Test 2: Sibling relation
    print("\n--- Test 2: Sibling Relations ---")
    print("Querying: sibling(bob, X)")
    start = time.time()
    results = list(kb.query("sibling", "bob", "X"))
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    if results:
        print(f"✓ Success: {len(results)} result(s)")
        for r in results:
            print(f"  - Bob's sibling: {r['X']}")
    else:
        print("✗ Failed: No results")
    
    # Test 3: More complex - ancestor
    print("\n--- Test 3: Ancestor Relations ---")
    kb.fact("parent", "charlie", "eve")
    kb.fact("parent", "eve", "frank")
    
    print("Querying: ancestor(alice, frank)")
    start = time.time()
    results = list(kb.query("ancestor", "alice", "frank"))
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    if results:
        print(f"✓ Success: {len(results)} result(s)")
    else:
        print("✗ Failed: No results")
    
    # Show template performance
    if hasattr(kb.engine.llm_hook, 'template_library'):
        stats = kb.engine.llm_hook.template_library.get_performance_stats()
        if stats:
            print(f"\n--- Template Performance for {model_name} ---")
            for name, perf in stats.items():
                if perf['count'] > 0:
                    print(f"  {name}:")
                    print(f"    Success rate: {perf['success_rate']:.0%}")
                    print(f"    Avg quality: {perf['avg_quality']:.2f}")
                    print(f"    Uses: {perf['count']}")
    
    # Show what was learned
    print(f"\n--- Knowledge Base Summary ---")
    print(f"  Facts: {len(kb.engine.kb.facts)}")
    print(f"  Rules: {len(kb.engine.kb.rules)}")
    if kb.engine.kb.rules:
        print("  Generated rules:")
        for rule in kb.engine.kb.rules:
            print(f"    - {rule.head} :- {', '.join(str(g) for g in rule.body)}")
    
    return kb


def test_template_learning():
    """Test how templates learn and adapt over multiple queries"""
    print("\n" + "="*60)
    print("Testing Template Learning and Adaptation")
    print("="*60)
    
    # Use the first available model for this test
    model = "gemma3n:latest"  # Or dynamically detect
    
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=model,
        temperature=0.7
    )
    
    # Disable debug for cleaner output
    if hasattr(kb.engine, 'llm_hook'):
        kb.engine.llm_hook.debug = False
    
    # Add a variety of facts
    test_facts = [
        ("parent", "john", "mary"),
        ("parent", "mary", "alice"),
        ("parent", "john", "bob"),
        ("parent", "bob", "charlie"),
        ("parent", "alice", "diana"),
        ("male", "john"),
        ("male", "bob"),
        ("male", "charlie"),
        ("female", "mary"),
        ("female", "alice"),
        ("female", "diana"),
    ]
    
    for fact in test_facts:
        kb.fact(*fact)
    
    # Test various queries to see template adaptation
    test_queries = [
        ("grandparent", "john", "X"),
        ("sibling", "mary", "X"),
        ("father", "john", "X"),  # Needs to combine parent + male
        ("mother", "mary", "X"),  # Needs to combine parent + female
        ("uncle", "bob", "X"),    # Complex: sibling of parent
        ("ancestor", "john", "diana"),  # Recursive relation
    ]
    
    print(f"\nTesting {len(test_queries)} different query types...")
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        functor, *args = query
        print(f"\n{i}. Query: {functor}({', '.join(args)})")
        
        start = time.time()
        results = list(kb.query(functor, *args))
        elapsed = time.time() - start
        
        if results:
            success_count += 1
            print(f"   ✓ Success in {elapsed:.2f}s - {len(results)} result(s)")
            # Show first few results
            for r in results[:2]:
                bindings = {k: v for k, v in r.bindings.items() if k.startswith('X')}
                if bindings:
                    print(f"     {bindings}")
        else:
            print(f"   ✗ Failed in {elapsed:.2f}s")
    
    print(f"\n--- Overall Success Rate: {success_count}/{len(test_queries)} ({success_count/len(test_queries):.0%}) ---")
    
    # Show template learning progress
    if hasattr(kb.engine.llm_hook, 'template_library'):
        stats = kb.engine.llm_hook.template_library.get_performance_stats()
        if stats:
            print("\n--- Template Learning Summary ---")
            best_template = max(stats.items(), key=lambda x: x[1]['success_rate'] if x[1]['count'] > 0 else 0)
            worst_template = min(stats.items(), key=lambda x: x[1]['success_rate'] if x[1]['count'] > 0 else 1)
            
            print(f"Best performing: {best_template[0]} ({best_template[1]['success_rate']:.0%} success)")
            print(f"Most used: {max(stats.items(), key=lambda x: x[1]['count'])[0]} ({max(stats.items(), key=lambda x: x[1]['count'])[1]['count']} uses)")
            
            # Show all templates
            print("\nAll templates:")
            for name, perf in sorted(stats.items(), key=lambda x: -x[1]['success_rate']):
                if perf['count'] > 0:
                    print(f"  {name:20} - Success: {perf['success_rate']:5.0%}, Uses: {perf['count']:3}")


def main():
    """Run all tests"""
    print("="*60)
    print("DreamLog Prompt Template System Test")
    print(f"Ollama URL: {OLLAMA_URL}")
    print("="*60)
    
    # Check which models are available
    import requests
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            available_models = [m['name'] for m in response.json()['models']]
            print(f"\nAvailable models: {', '.join(available_models)}")
            
            # Filter test models to only available ones
            models_to_test = [m for m in MODELS_TO_TEST if m in available_models]
            if not models_to_test:
                print(f"\nWarning: None of the test models {MODELS_TO_TEST} are available.")
                print("Using first available model instead.")
                models_to_test = [available_models[0]] if available_models else []
        else:
            print(f"\nCouldn't check available models. Using configured models.")
            models_to_test = MODELS_TO_TEST[:1]  # Just use first one
    except Exception as e:
        print(f"\nError checking Ollama: {e}")
        print("Proceeding with first configured model.")
        models_to_test = MODELS_TO_TEST[:1]
    
    if not models_to_test:
        print("\nNo models available to test!")
        return
    
    # Test 1: Single model with debug output
    print("\n" + "="*60)
    print("TEST 1: Single Model with Debug Output")
    print("="*60)
    test_model_adaptation(models_to_test[0])
    
    # Test 2: Template learning across queries
    print("\n" + "="*60)
    print("TEST 2: Template Learning Across Queries")
    print("="*60)
    test_template_learning()
    
    # Test 3: Compare different models (if multiple available)
    if len(models_to_test) > 1:
        print("\n" + "="*60)
        print("TEST 3: Comparing Models")
        print("="*60)
        
        model_stats = {}
        for model in models_to_test[:2]:  # Test at most 2 models
            kb = test_model_adaptation(model)
            if hasattr(kb.engine.llm_hook, 'template_library'):
                stats = kb.engine.llm_hook.template_library.get_performance_stats()
                model_stats[model] = stats
        
        # Compare results
        print("\n" + "="*60)
        print("Model Comparison Summary")
        print("="*60)
        for model, stats in model_stats.items():
            total_uses = sum(s['count'] for s in stats.values())
            avg_success = sum(s['success_rate'] * s['count'] for s in stats.values()) / total_uses if total_uses > 0 else 0
            print(f"\n{model}:")
            print(f"  Total queries: {total_uses}")
            print(f"  Avg success rate: {avg_success:.0%}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()