#!/usr/bin/env python3
"""
Test improved prompts with multiple examples
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dreamlog.pythonic import dreamlog

# Ollama configuration
OLLAMA_URL = "http://192.168.0.225:11434"
MODEL = "gemma3n:latest"

def test_improved_prompts():
    """Test with improved prompt templates"""
    print("=" * 60)
    print("Testing Improved Prompts with Examples")
    print("=" * 60)
    
    # Create KB without retry wrapper to see raw results
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=MODEL,
        temperature=0.5,
        use_retry=False  # No retry, just improved prompts
    )
    
    # Enable debug to see prompts
    if hasattr(kb.engine, 'llm_hook'):
        kb.engine.llm_hook.debug = True
    
    # Add facts
    kb.fact("parent", "alice", "bob")
    kb.fact("parent", "bob", "charlie")
    kb.fact("parent", "alice", "diana")
    
    print("\nKnowledge base:")
    for fact in kb.engine.kb.facts:
        print(f"  - {fact.term}")
    
    # Test queries
    test_queries = [
        ("grandparent", ["alice", "charlie"], "Should generate grandparent rule"),
        ("sibling", ["bob", "diana"], "Should generate sibling rule"),
        ("ancestor", ["alice", "charlie"], "Should generate ancestor rule (possibly recursive)"),
    ]
    
    success_count = 0
    
    for functor, args, description in test_queries:
        print(f"\n{'='*50}")
        print(f"Test: {description}")
        print(f"Query: {functor}({', '.join(args)})")
        print(f"{'='*50}")
        
        results = list(kb.query(functor, *args))
        
        if results:
            success_count += 1
            print(f"\n✓ SUCCESS: Found {len(results)} result(s)")
            
            # Show what rules were generated
            new_rules = [r for r in kb.engine.kb.rules if str(r.head).startswith(functor)]
            if new_rules:
                print(f"\nGenerated rules:")
                for rule in new_rules:
                    print(f"  {rule.head} :- {', '.join(str(g) for g in rule.body)}")
        else:
            print(f"\n✗ FAILED: No results")
    
    print(f"\n{'='*60}")
    print(f"Results: {success_count}/{len(test_queries)} queries succeeded")
    print(f"Success rate: {success_count/len(test_queries)*100:.0f}%")
    
    # Show all generated rules
    if kb.engine.kb.rules:
        print(f"\nAll generated rules ({len(kb.engine.kb.rules)} total):")
        for rule in kb.engine.kb.rules:
            print(f"  - {rule.head} :- {', '.join(str(g) for g in rule.body)}")
    
    return success_count


def main():
    print("Improved Prompt Template Test")
    print("=" * 60)
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Model: {MODEL}")
    print("=" * 60)
    
    success = test_improved_prompts()
    
    if success >= 2:
        print("\n✓ Improved prompts are working well!")
    else:
        print("\n⚠ Prompts may need further improvement")


if __name__ == "__main__":
    main()