#!/usr/bin/env python3
"""
JLOG Demo - Complete demonstration of the rewritten JSON Prolog system with LLM integration

This demo shows:
1. Basic Prolog evaluation without LLM
2. Zero-shot learning with LLM integration  
3. Automatic rule generation
4. JSON serialization and deserialization
5. Complex reasoning with LLM-generated knowledge
"""

import sys
from pathlib import Path

# Add the jlog package to the path
sys.path.insert(0, str(Path(__file__).parent))

from jlog import (
    Term, Atom, Variable, Compound, atom, var, compound,
    Fact, Rule, KnowledgeBase, 
    PrologEvaluator, JLogEngine,
    LLMHook, MockLLMProvider,
    create_engine_with_llm
)


def demo_basic_prolog():
    """Demonstrate basic Prolog functionality without LLM"""
    print("=== Basic Prolog Demo (No LLM) ===\n")
    
    engine = JLogEngine()
    
    # Add some facts
    print("Adding facts:")
    facts = [
        compound("parent", atom("john"), atom("mary")),
        compound("parent", atom("john"), atom("tom")), 
        compound("parent", atom("mary"), atom("alice")),
        compound("male", atom("john")),
        compound("male", atom("tom")),
        compound("female", atom("mary")),
        compound("female", atom("alice"))
    ]
    
    for fact_term in facts:
        engine.add_fact_from_term(fact_term)
        print(f"  {fact_term}")
    
    # Add some rules
    print("\nAdding rules:")
    rules = [
        (compound("grandparent", var("X"), var("Z")), 
         [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]),
        
        (compound("father", var("X"), var("Y")), 
         [compound("parent", var("X"), var("Y")), compound("male", var("X"))]),
         
        (compound("mother", var("X"), var("Y")), 
         [compound("parent", var("X"), var("Y")), compound("female", var("X"))])
    ]
    
    for head, body in rules:
        engine.add_rule_from_terms(head, body)
        body_str = ", ".join(str(term) for term in body)
        print(f"  {head} :- {body_str}")
    
    print(f"\nKnowledge base: {len(engine.facts)} facts, {len(engine.rules)} rules")
    
    # Run some queries
    queries = [
        [compound("parent", atom("john"), var("X"))],
        [compound("grandparent", var("X"), var("Y"))],
        [compound("father", var("X"), var("Y"))],
        [compound("mother", atom("mary"), var("Z"))]
    ]
    
    print("\nQueries:")
    for query_goals in queries:
        goals_str = ", ".join(str(goal) for goal in query_goals)
        print(f"\n?- {goals_str}")
        
        solutions = engine.query(query_goals)
        if solutions:
            for i, solution in enumerate(solutions, 1):
                bindings = solution.get_ground_bindings()
                if bindings:
                    binding_strs = [f"{var}={val}" for var, val in bindings.items()]
                    print(f"  {i}. {', '.join(binding_strs)}")
                else:
                    print(f"  {i}. Yes")
        else:
            print("  No solutions found")


def demo_zero_shot_learning():
    """Demonstrate zero-shot learning with LLM integration"""
    print("\n=== Zero-Shot Learning Demo ===\n")
    
    # Start with completely empty knowledge base but with LLM integration
    mock_llm = MockLLMProvider(knowledge_domain="family")
    engine = create_engine_with_llm(mock_llm, "family")
    
    print("Starting with empty knowledge base...")
    print(f"Knowledge: {len(engine.facts)} facts, {len(engine.rules)} rules\n")
    
    # Query for parent relationships - should trigger LLM
    print("Query 1: ?- parent(john, X)")
    print("This should trigger the LLM to generate parent facts...")
    
    solutions = engine.query([compound("parent", atom("john"), var("X"))])
    
    print(f"Found {len(solutions)} solutions:")
    for i, solution in enumerate(solutions, 1):
        bindings = solution.get_ground_bindings()
        x_val = bindings.get("X")
        if x_val:
            print(f"  {i}. X = {x_val}")
    
    print(f"\nKnowledge after LLM generation: {len(engine.facts)} facts, {len(engine.rules)} rules")
    
    # Query for ancestor relationships - should trigger rule generation
    print("\nQuery 2: ?- ancestor(john, alice)")
    print("This should trigger the LLM to generate ancestor rules...")
    
    solutions = engine.query([compound("ancestor", atom("john"), atom("alice"))])
    
    if solutions:
        print("✓ Found ancestor relationship!")
    else:
        print("✗ No ancestor relationship found")
    
    print(f"\nFinal knowledge: {len(engine.facts)} facts, {len(engine.rules)} rules")
    
    # Show generated knowledge
    print("\nGenerated facts:")
    for fact in engine.facts:
        print(f"  {fact}")
    
    print("\nGenerated rules:")
    for rule in engine.rules:
        print(f"  {rule}")


def demo_progressive_reasoning():
    """Demonstrate how the system builds knowledge progressively"""
    print("\n=== Progressive Reasoning Demo ===\n")
    
    mock_llm = MockLLMProvider(knowledge_domain="family")
    engine = create_engine_with_llm(mock_llm, "family")
    engine.set_trace(True)  # Enable tracing to see what happens
    
    print("Progressive knowledge building through queries...\n")
    
    queries = [
        ([compound("parent", var("X"), var("Y"))], "Who are the parents?"),
        ([compound("grandparent", var("GP"), var("GC"))], "Who are the grandparents?"),
        ([compound("ancestor", atom("john"), var("DESC"))], "Who are John's descendants?"),
    ]
    
    for query_goals, description in queries:
        print(f"\n{description}")
        goals_str = ", ".join(str(goal) for goal in query_goals)
        print(f"Query: ?- {goals_str}")
        
        solutions = engine.query(query_goals)
        
        print(f"Solutions: {len(solutions)}")
        for i, solution in enumerate(solutions[:3], 1):  # Show first 3
            bindings = solution.get_ground_bindings()
            if bindings:
                binding_strs = [f"{var}={val}" for var, val in bindings.items()]
                print(f"  {i}. {', '.join(binding_strs)}")
        
        print(f"Knowledge state: {len(engine.facts)} facts, {len(engine.rules)} rules")


def demo_json_serialization():
    """Demonstrate JSON serialization and knowledge transfer"""
    print("\n=== JSON Serialization Demo ===\n")
    
    # Create engine with LLM and generate some knowledge
    mock_llm = MockLLMProvider(knowledge_domain="family")
    engine1 = create_engine_with_llm(mock_llm, "family")
    
    print("Generating knowledge in Engine 1...")
    engine1.query([compound("parent", var("X"), var("Y"))])
    engine1.query([compound("ancestor", var("A"), var("B"))])
    
    print(f"Engine 1: {len(engine1.facts)} facts, {len(engine1.rules)} rules")
    
    # Export to JSON
    json_data = engine1.save_to_json()
    print(f"\nExported knowledge to JSON ({len(json_data)} characters)")
    
    # Create new engine and import knowledge
    engine2 = JLogEngine()  # No LLM integration
    engine2.load_from_json(json_data)
    
    print(f"Engine 2 (after import): {len(engine2.facts)} facts, {len(engine2.rules)} rules")
    
    # Test that the imported knowledge works
    print("\nTesting imported knowledge:")
    test_queries = [
        [compound("parent", atom("john"), var("X"))],
        [compound("ancestor", atom("mary"), var("Y"))]
    ]
    
    for query_goals in test_queries:
        goals_str = ", ".join(str(goal) for goal in query_goals)
        print(f"\n?- {goals_str}")
        
        solutions = engine2.query(query_goals)
        for i, solution in enumerate(solutions, 1):
            bindings = solution.get_ground_bindings()
            if bindings:
                binding_strs = [f"{var}={val}" for var, val in bindings.items()]
                print(f"  {i}. {', '.join(binding_strs)}")


def demo_complex_reasoning():
    """Demonstrate complex multi-step reasoning with LLM"""
    print("\n=== Complex Reasoning Demo ===\n")
    
    mock_llm = MockLLMProvider(knowledge_domain="family")
    engine = create_engine_with_llm(mock_llm, "family")
    
    print("Complex query requiring multi-step reasoning:")
    print("Find all people who are both ancestors and descendants of someone")
    
    # This requires the system to:
    # 1. Generate parent facts
    # 2. Generate ancestor rules  
    # 3. Apply recursive reasoning
    # 4. Find intersections
    
    query = [
        compound("ancestor", var("X"), var("Y")),
        compound("ancestor", var("Z"), var("X"))
    ]
    
    goals_str = ", ".join(str(goal) for goal in query)
    print(f"\nQuery: ?- {goals_str}")
    
    solutions = engine.query(query)
    
    print(f"\nFound {len(solutions)} solution(s):")
    for i, solution in enumerate(solutions, 1):
        bindings = solution.get_ground_bindings()
        if bindings:
            binding_strs = [f"{var}={val}" for var, val in bindings.items()]
            print(f"  {i}. {', '.join(binding_strs)}")
    
    print(f"\nTotal knowledge generated: {len(engine.facts)} facts, {len(engine.rules)} rules")


def main():
    """Run all demos"""
    print("JLOG - Complete Demo")
    print("JSON Prolog with LLM Integration")
    print("=" * 50)
    
    demo_basic_prolog()
    print("\n" + "=" * 50)
    
    demo_zero_shot_learning() 
    print("\n" + "=" * 50)
    
    demo_progressive_reasoning()
    print("\n" + "=" * 50)
    
    demo_json_serialization()
    print("\n" + "=" * 50)
    
    demo_complex_reasoning()
    print("\n" + "=" * 50)
    
    print("\nDemo completed! JLOG successfully demonstrated:")
    print("✓ Clean JSON-based term representation")
    print("✓ Proper Prolog evaluation with SLD resolution")
    print("✓ Zero-shot learning from empty knowledge base")
    print("✓ Automatic fact and rule generation via LLM")
    print("✓ Progressive knowledge building")
    print("✓ JSON serialization and knowledge transfer")
    print("✓ Complex multi-step reasoning")
    print("✓ Proper variable scoping and unification")


if __name__ == "__main__":
    main()
