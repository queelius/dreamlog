#!/usr/bin/env python3
"""
Simple test of prompt template system with Ollama
Shows how different templates are selected and perform
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dreamlog.pythonic import dreamlog

# Ollama configuration
OLLAMA_URL = "http://192.168.0.225:11434"
MODEL = "gemma3n:latest"

def main():
    print("="*60)
    print("Simple Prompt Template Test")
    print(f"Model: {MODEL}")
    print("="*60)
    
    # Create KB with LLM support
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=MODEL,
        temperature=0.7
    )
    
    # Enable debug to see prompts
    if hasattr(kb.engine, 'llm_hook'):
        kb.engine.llm_hook.debug = True
    
    # Add some facts
    print("\n1. Adding basic facts...")
    kb.fact("parent", "alice", "bob")
    kb.fact("parent", "bob", "charlie")
    kb.fact("parent", "alice", "diana")
    kb.fact("parent", "diana", "eve")
    
    print("   Facts added:")
    for fact in kb.engine.kb.facts:
        print(f"     - {fact.term}")
    
    # Test 1: Simple rule generation
    print("\n2. Testing grandparent rule generation...")
    results = list(kb.query("grandparent", "alice", "charlie"))
    
    if results:
        print(f"   ✓ Query succeeded with {len(results)} result(s)")
        if kb.engine.kb.rules:
            print("   Generated rules:")
            for rule in kb.engine.kb.rules:
                print(f"     - {rule.head} :- {', '.join(str(g) for g in rule.body)}")
    else:
        print("   ✗ Query failed")
    
    # Test 2: Sibling relation (more complex)
    print("\n3. Testing sibling relation...")
    results = list(kb.query("sibling", "bob", "diana"))
    
    if results:
        print(f"   ✓ Query succeeded with {len(results)} result(s)")
    else:
        print("   ✗ Query failed")
    
    # Test 3: Great-grandparent (recursive)
    print("\n4. Testing great-grandparent relation...")
    results = list(kb.query("great_grandparent", "alice", "eve"))
    
    if results:
        print(f"   ✓ Query succeeded with {len(results)} result(s)")
    else:
        print("   ✗ Query failed")
    
    # Show template performance
    print("\n5. Template Performance Summary")
    print("-" * 40)
    
    if hasattr(kb.engine.llm_hook, 'template_library'):
        stats = kb.engine.llm_hook.template_library.get_performance_stats()
        
        if stats:
            # Sort by success rate
            sorted_stats = sorted(stats.items(), 
                                key=lambda x: (x[1]['success_rate'], x[1]['count']), 
                                reverse=True)
            
            for name, perf in sorted_stats:
                print(f"   {name}:")
                print(f"     Success rate: {perf['success_rate']:.0%}")
                print(f"     Avg quality:  {perf['avg_quality']:.2f}")
                print(f"     Times used:   {perf['count']}")
                print()
            
            # Overall stats
            total_queries = sum(s['count'] for s in stats.values())
            total_success = sum(s['success_rate'] * s['count'] for s in stats.values())
            avg_success = total_success / total_queries if total_queries > 0 else 0
            
            print(f"   Overall:")
            print(f"     Total queries: {total_queries}")
            print(f"     Avg success:   {avg_success:.0%}")
        else:
            print("   No performance data collected")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == "__main__":
    main()