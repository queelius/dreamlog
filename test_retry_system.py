#!/usr/bin/env python3
"""
Test the retry system with JSON validation
Shows how the system retries and samples to get valid JSON
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dreamlog.pythonic import dreamlog
import time

# Ollama configuration
OLLAMA_URL = "http://192.168.0.225:11434"
MODEL = "gemma3n:latest"

def test_without_retry():
    """Test without retry system"""
    print("="*60)
    print("Test WITHOUT Retry System")
    print("="*60)
    
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=MODEL,
        temperature=0.7,
        use_retry=False  # Disable retry
    )
    
    # Add facts
    kb.fact("parent", "alice", "bob")
    kb.fact("parent", "bob", "charlie")
    
    # Test queries
    success_count = 0
    test_queries = [
        ("grandparent", "alice", "charlie"),
        ("sibling", "bob", "X"),
        ("ancestor", "alice", "charlie"),
    ]
    
    for query in test_queries:
        functor, *args = query
        print(f"\nQuery: {functor}({', '.join(args)})")
        
        start = time.time()
        results = list(kb.query(functor, *args))
        elapsed = time.time() - start
        
        if results:
            success_count += 1
            print(f"  ✓ Success in {elapsed:.2f}s")
        else:
            print(f"  ✗ Failed in {elapsed:.2f}s")
    
    print(f"\nSuccess rate: {success_count}/{len(test_queries)} ({success_count/len(test_queries):.0%})")
    return success_count


def test_with_retry():
    """Test with retry system enabled"""
    print("\n" + "="*60)
    print("Test WITH Retry System")
    print("="*60)
    
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=MODEL,
        temperature=0.7,
        use_retry=True,  # Enable retry
        max_retries=3,
        verbose_retry=True  # Show retry attempts
    )
    
    # Add facts
    kb.fact("parent", "alice", "bob")
    kb.fact("parent", "bob", "charlie")
    
    # Test queries
    success_count = 0
    test_queries = [
        ("grandparent", "alice", "charlie"),
        ("sibling", "bob", "X"),
        ("ancestor", "alice", "charlie"),
    ]
    
    for query in test_queries:
        functor, *args = query
        print(f"\nQuery: {functor}({', '.join(args)})")
        
        start = time.time()
        results = list(kb.query(functor, *args))
        elapsed = time.time() - start
        
        if results:
            success_count += 1
            print(f"  ✓ Success in {elapsed:.2f}s")
        else:
            print(f"  ✗ Failed in {elapsed:.2f}s")
    
    print(f"\nSuccess rate: {success_count}/{len(test_queries)} ({success_count/len(test_queries):.0%})")
    return success_count


def test_complex_queries():
    """Test more complex queries that benefit from retry"""
    print("\n" + "="*60)
    print("Test Complex Queries with Retry")
    print("="*60)
    
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=MODEL,
        temperature=0.7,
        use_retry=True,
        max_retries=5,  # More retries for complex queries
        verbose_retry=False  # Clean output
    )
    
    # Build a more complex knowledge base
    facts = [
        ("parent", "alice", "bob"),
        ("parent", "alice", "claire"),
        ("parent", "bob", "diana"),
        ("parent", "bob", "eric"),
        ("parent", "claire", "frank"),
        ("male", "bob"),
        ("male", "eric"),
        ("male", "frank"),
        ("female", "alice"),
        ("female", "claire"),
        ("female", "diana"),
    ]
    
    for fact_data in facts:
        kb.fact(*fact_data)
    
    # Complex queries that combine multiple relations
    complex_queries = [
        ("grandparent", "alice", "X"),  # Should find diana, eric, frank
        ("sibling", "diana", "X"),      # Should find eric
        ("uncle", "bob", "X"),          # Uncle relation (brother of parent)
        ("aunt", "claire", "X"),        # Aunt relation (sister of parent)
        ("nephew", "diana", "X"),       # Reverse of uncle/aunt
        ("cousin", "diana", "X"),       # Children of siblings
    ]
    
    print("\nTesting complex family relations...")
    success_count = 0
    
    for query in complex_queries:
        functor, *args = query
        print(f"\n{functor}({', '.join(args)}):")
        
        start = time.time()
        results = list(kb.query(functor, *args))
        elapsed = time.time() - start
        
        if results:
            success_count += 1
            print(f"  ✓ Found {len(results)} result(s) in {elapsed:.2f}s")
            for i, result in enumerate(results[:3], 1):
                bindings = {k: str(v) for k, v in result.bindings.items() if k.startswith('X')}
                if bindings:
                    print(f"    {i}. {bindings}")
        else:
            print(f"  ✗ No results in {elapsed:.2f}s")
    
    print(f"\n--- Summary ---")
    print(f"Success rate: {success_count}/{len(complex_queries)} ({success_count/len(complex_queries):.0%})")
    
    # Show generated rules
    if kb.engine.kb.rules:
        print(f"\nGenerated {len(kb.engine.kb.rules)} rules:")
        for rule in kb.engine.kb.rules[:5]:  # Show first 5
            print(f"  - {rule.head} :- {', '.join(str(g) for g in rule.body)}")


def main():
    print("="*60)
    print("DreamLog Retry System Test")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Model: {MODEL}")
    print("="*60)
    
    # Test 1: Without retry
    without_retry = test_without_retry()
    
    # Test 2: With retry
    with_retry = test_with_retry()
    
    # Test 3: Complex queries
    test_complex_queries()
    
    # Compare results
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(f"Without retry: {without_retry}/3 queries succeeded")
    print(f"With retry:    {with_retry}/3 queries succeeded")
    
    if with_retry > without_retry:
        improvement = ((with_retry - without_retry) / max(without_retry, 1)) * 100
        print(f"\n✓ Retry system improved success rate by {improvement:.0f}%")
    elif with_retry == without_retry:
        print(f"\n• Retry system maintained same success rate")
    else:
        print(f"\n✗ Unexpected: retry system had lower success rate")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == "__main__":
    main()