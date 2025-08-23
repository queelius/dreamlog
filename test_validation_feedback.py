#!/usr/bin/env python3
"""
Test validation feedback system with verbose output
"""

from dreamlog.pythonic import dreamlog

OLLAMA_URL = "http://192.168.0.225:11434"
MODEL = "gemma3n:latest"

def test_validation_feedback():
    """Test with verbose validation feedback"""
    print("=" * 60)
    print("Testing Validation Feedback System")
    print("=" * 60)
    
    # Create KB with retry and verbose feedback
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=MODEL,
        temperature=0.5,
        use_retry=True,
        verbose_retry=True  # Show validation feedback
    )
    
    # Add facts
    kb.fact("parent", "alice", "bob")
    kb.fact("parent", "bob", "charlie")
    
    print("\nTest 1: Grandparent (should work)")
    print("-" * 40)
    results = list(kb.query("grandparent", "alice", "charlie"))
    print(f"Results: {len(results)} found")
    
    print("\n" + "=" * 60)
    print("\nTest 2: Sibling (more complex)")
    print("-" * 40)
    kb.fact("parent", "alice", "diana")
    results = list(kb.query("sibling", "bob", "diana"))
    print(f"Results: {len(results)} found")
    
    print("\n" + "=" * 60)
    print("\nTest 3: Uncle (requires multiple rules)")
    print("-" * 40)
    kb.fact("male", "bob")
    results = list(kb.query("uncle", "bob", "charlie"))
    print(f"Results: {len(results)} found")
    
    # Show generated rules
    if kb.engine.kb.rules:
        print("\n" + "=" * 60)
        print("Generated Rules:")
        for rule in kb.engine.kb.rules:
            print(f"  - {rule.head} :- {', '.join(str(g) for g in rule.body)}")


if __name__ == "__main__":
    test_validation_feedback()