#!/usr/bin/env python3
"""
Test the prompt template system with Ollama integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dreamlog.pythonic import dreamlog
from dreamlog.prompt_template_system import PromptTemplateLibrary, QueryContext

# Ollama configuration
OLLAMA_URL = "http://192.168.0.225:11434"
MODEL = "phi4-mini-reasoning"
TEMPERATURE = 0.7

def test_template_library():
    """Test the prompt template library directly"""
    print("=" * 60)
    print("Testing Prompt Template Library")
    print("=" * 60)
    
    # Create template library for our model
    library = PromptTemplateLibrary(model_name=MODEL)
    
    # Test with different query contexts
    test_cases = [
        # Simple query with no KB
        QueryContext(
            term="(parent john mary)",
            kb_facts=[],
            kb_rules=[],
            existing_functors=[]
        ),
        # Query with some existing facts
        QueryContext(
            term="(grandparent alice charlie)",
            kb_facts=[
                "(parent alice bob)",
                "(parent bob charlie)",
                "(parent dave eve)"
            ],
            kb_rules=[],
            existing_functors=["parent"]
        ),
        # Query with facts and rules
        QueryContext(
            term="(ancestor X Y)",
            kb_facts=[
                "(parent alice bob)",
                "(parent bob charlie)",
                "(parent charlie diana)"
            ],
            kb_rules=[
                ("(grandparent X Z)", ["(parent X Y)", "(parent Y Z)"]),
            ],
            existing_functors=["parent", "grandparent"]
        ),
    ]
    
    for i, context in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Query: {context.term}")
        print(f"KB has {len(context.kb_facts)} facts, {len(context.kb_rules)} rules")
        
        # Get prompt
        prompt, template_name = library.get_best_prompt(context)
        print(f"Selected template: {template_name}")
        print(f"Prompt length: {len(prompt)} characters")
        
        # Show a preview
        if len(prompt) < 500:
            print(f"Full prompt:\n{prompt}")
        else:
            print(f"Prompt preview (first 300 chars):\n{prompt[:300]}...")
    
    # Show adaptive selection in action
    print("\n" + "=" * 60)
    print("Testing Adaptive Template Selection")
    print("=" * 60)
    
    # Simulate some performance data
    library.record_performance("many_shot_examples", success=True, response_quality=0.9)
    library.record_performance("many_shot_examples", success=True, response_quality=0.8)
    library.record_performance("chain_of_thought", success=False, response_quality=0.2)
    library.record_performance("zero_shot", success=True, response_quality=0.6)
    
    # Get performance stats
    stats = library.get_performance_stats()
    print("\nTemplate Performance Stats:")
    for name, perf in stats.items():
        if perf['count'] > 0:
            print(f"  {name}: avg_quality={perf['avg_quality']:.2f}, success_rate={perf['success_rate']:.2f}, count={perf['count']}")
    
    # Check which template gets selected now
    test_context = QueryContext(
        term="(sibling X Y)",
        kb_facts=["(parent alice bob)", "(parent alice charlie)"],
        kb_rules=[],
        existing_functors=["parent"]
    )
    
    prompt, template_name = library.get_best_prompt(test_context)
    print(f"\nAfter learning, selected template: {template_name}")


def test_with_dreamlog():
    """Test prompt templates integrated with DreamLog"""
    print("\n" + "=" * 60)
    print("Testing Prompt Templates with DreamLog Integration")
    print("=" * 60)
    
    # Create KB with LLM support
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=MODEL,
        temperature=TEMPERATURE
    )
    
    # Enable debug mode
    if hasattr(kb.engine, 'llm_hook'):
        kb.engine.llm_hook.debug = True
    
    # Add some initial facts
    kb.fact("parent", "alice", "bob")
    kb.fact("parent", "bob", "charlie")
    
    print("\nInitial KB:")
    print(f"  Facts: {[str(f.term) for f in kb.engine.kb.facts]}")
    
    # Query for something that requires LLM generation
    print("\n--- Querying for (grandparent alice charlie) ---")
    results = list(kb.query("grandparent", "alice", "charlie"))
    
    if results:
        print(f"Query succeeded: {results}")
        # Check what was added to KB
        print("\nKB after LLM generation:")
        print(f"  Facts: {[str(f.term) for f in kb.engine.kb.facts]}")
        print(f"  Rules: {[(str(r.head), [str(g) for g in r.body]) for r in kb.engine.kb.rules]}")
    else:
        print("Query failed - no results")
    
    # Try another query to test template adaptation
    print("\n--- Querying for (sibling bob X) ---")
    results = list(kb.query("sibling", "bob", "X"))
    
    if results:
        print(f"Query succeeded: {results}")
    else:
        print("Query failed - no results")
    
    # Check template performance if available
    if hasattr(kb.engine.llm_hook, 'template_library'):
        stats = kb.engine.llm_hook.template_library.get_performance_stats()
        print("\nTemplate Performance After Queries:")
        for name, perf in stats.items():
            if perf['count'] > 0:
                print(f"  {name}: avg_quality={perf['avg_quality']:.2f}, success_rate={perf['success_rate']:.2f}, count={perf['count']}")


def main():
    """Run all tests"""
    try:
        # Test template library directly
        test_template_library()
        
        # Test with DreamLog integration
        test_with_dreamlog()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()