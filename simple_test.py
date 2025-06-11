#!/usr/bin/env python3
"""
Simple JLOG Test - Clean demonstration of zero-shot learning
"""

import sys
from pathlib import Path

# Add the jlog package to the path
sys.path.insert(0, str(Path(__file__).parent))

from jlog import (
    atom, var, compound,
    JLogEngine, MockLLMProvider, LLMHook, create_engine_with_llm
)


def test_zero_shot_learning():
    """Clean test of zero-shot learning capabilities"""
    print("JLOG -- Zero-Shot Learning Test")
    print("=" * 40)
    
    # Create engine with LLM integration but empty knowledge base
    mock_llm = MockLLMProvider(knowledge_domain="family")
    engine = create_engine_with_llm(mock_llm, "family")
    
    print(f"Initial state: {len(engine.facts)} facts, {len(engine.rules)} rules")
    print()
    
    # Test 1: Ask about parent relationships (should generate facts)
    print("Test 1: Who are John's children?")
    print("Query: parent(john, X)")
    
    solutions = engine.query([compound("parent", atom("john"), var("X"))])
    
    for i, solution in enumerate(solutions, 1):
        x_binding = solution.get_binding("X")
        print(f"  {i}. X = {x_binding}")
    
    print(f"Knowledge after query: {len(engine.facts)} facts, {len(engine.rules)} rules")
    print()
    
    # Test 2: Ask about ancestors (should generate rules)
    print("Test 2: Is John an ancestor of Alice?")
    print("Query: ancestor(john, alice)")
    
    is_ancestor = engine.ask(compound("ancestor", atom("john"), atom("alice")))
    print(f"Result: {'Yes' if is_ancestor else 'No'}")
    
    print(f"Knowledge after query: {len(engine.facts)} facts, {len(engine.rules)} rules")
    print()
    
    # Test 3: Complex reasoning
    print("Test 3: Find all of John's descendants")
    print("Query: ancestor(john, X)")
    
    descendants = engine.find_all(compound("ancestor", atom("john"), var("X")), "X")
    
    print("John's descendants:")
    for i, desc in enumerate(descendants, 1):
        print(f"  {i}. {desc}")
    
    print()
    
    # Show final knowledge base
    print("Final Knowledge Base:")
    print("Facts:")
    for fact in engine.facts:
        print(f"  {fact}")
    
    print("Rules:")
    for rule in engine.rules:
        print(f"  {rule}")


def test_basic_prolog():
    """Test basic Prolog functionality without LLM"""
    print("\nBasic Prolog Test (No LLM)")
    print("=" * 30)
    
    engine = JLogEngine()  # No LLM hook
    
    # Add some facts manually
    engine.add_fact_from_term(compound("likes", atom("mary"), atom("food")))
    engine.add_fact_from_term(compound("likes", atom("mary"), atom("wine")))
    engine.add_fact_from_term(compound("likes", atom("john"), atom("wine")))
    engine.add_fact_from_term(compound("likes", atom("john"), atom("mary")))
    
    # Add a rule
    engine.add_rule_from_terms(
        compound("happy", var("X")),
        [compound("likes", var("X"), atom("wine"))]
    )
    
    print("Knowledge base:")
    for fact in engine.facts:
        print(f"  {fact}")
    for rule in engine.rules:
        print(f"  {rule}")
    
    print()
    
    # Query who is happy
    print("Query: happy(X)")
    happy_people = engine.find_all(compound("happy", var("X")), "X")
    
    print("Happy people:")
    for person in happy_people:
        print(f"  {person}")


if __name__ == "__main__":
    test_zero_shot_learning()
    test_basic_prolog()
