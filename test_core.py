#!/usr/bin/env python3
"""
Quick test of core DreamLog functionality.
This verifies the essential features work before testing with Ollama.
"""

from dreamlog.pythonic import dreamlog
from dreamlog.prefix_parser import parse_s_expression, term_to_sexp
from dreamlog.terms import atom, var, compound

def test_basic_facts_and_queries():
    """Test basic fact addition and querying."""
    print("Testing basic facts and queries...")
    
    kb = dreamlog()
    
    # Add facts using S-expressions
    kb.parse("(parent john mary)")
    kb.parse("(parent mary alice)")
    kb.parse("(parent bob charlie)")
    
    # Query
    results = list(kb.query("parent", "john", "X"))
    assert len(results) == 1
    assert str(results[0]['X']) == 'mary'
    print("✓ Basic facts and queries work")

def test_rules_and_inference():
    """Test rules and inference."""
    print("\nTesting rules and inference...")
    
    kb = dreamlog()
    
    # Add facts and rules
    kb.parse("(parent john mary)")
    kb.parse("(parent mary alice)")
    kb.parse("(grandparent X Z) :- (parent X Y), (parent Y Z)")
    
    # Query using rule
    results = list(kb.query("grandparent", "john", "X"))
    assert len(results) == 1
    assert str(results[0]['X']) == 'alice'
    print("✓ Rules and inference work")

def test_pythonic_api():
    """Test the fluent/pythonic API."""
    print("\nTesting pythonic API...")
    
    kb = dreamlog()
    
    # Fluent API
    kb.fact("parent", "john", "mary") \
      .fact("parent", "mary", "alice")
    
    kb.rule("grandparent", ["X", "Z"]) \
      .when("parent", ["X", "Y"]) \
      .and_("parent", ["Y", "Z"]) \
      .build()
    
    # Query
    results = list(kb.query("grandparent", "john", "X"))
    print(f"Results: {results}")  # Debug
    if len(results) == 0:
        # Maybe the rule wasn't added correctly, let's check
        print(f"Facts: {[str(f.term) for f in kb.engine.kb.facts]}")
        print(f"Rules: {[(str(r.head), [str(b) for b in r.body]) for r in kb.engine.kb.rules]}")
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert str(results[0]['X']) == 'alice'
    print("✓ Pythonic API works")

def test_term_parsing():
    """Test S-expression parsing."""
    print("\nTesting S-expression parsing...")
    
    # Parse atoms
    a = parse_s_expression("john")
    assert a.value == "john"
    
    # Parse variables
    v = parse_s_expression("X")
    assert v.name == "X"
    
    # Parse compounds
    c = parse_s_expression("(parent john mary)")
    assert c.functor == "parent"
    assert len(c.args) == 2
    assert c.args[0].value == "john"
    assert c.args[1].value == "mary"
    
    # Convert back to S-expression
    sexp = term_to_sexp(c)
    assert sexp == "(parent john mary)"
    
    print("✓ S-expression parsing works")

def test_unification():
    """Test basic unification."""
    print("\nTesting unification...")
    
    from dreamlog.unification import unify
    
    # Unify atoms
    t1 = atom("john")
    t2 = atom("john")
    result = unify(t1, t2)
    assert result is not None
    
    # Unify with variables
    t1 = compound("parent", var("X"), atom("mary"))
    t2 = compound("parent", atom("john"), atom("mary"))
    result = unify(t1, t2)
    assert result is not None
    # Find the variable in the result
    x_var = var("X")
    assert x_var in result or str(x_var) in str(result)  # Check it unified X to john
    
    print("✓ Unification works")

def test_mock_provider():
    """Test that mock provider loads correctly for testing."""
    print("\nTesting mock provider...")
    
    from tests.mock_provider import MockLLMProvider
    
    provider = MockLLMProvider(knowledge_domain="family")
    assert provider.knowledge_domain == "family"
    
    # Test it can generate responses
    response = provider.generate_knowledge("parent", context="test")
    assert response is not None
    
    print("✓ Mock provider works")

def main():
    print("=" * 60)
    print("DreamLog Core Functionality Test")
    print("=" * 60)
    
    try:
        test_basic_facts_and_queries()
        test_rules_and_inference()
        test_pythonic_api()
        test_term_parsing()
        test_unification()
        test_mock_provider()
        
        print("\n" + "=" * 60)
        print("✅ All core tests passed!")
        print("=" * 60)
        print("\nDreamLog is ready to use with Ollama.")
        print("Update dreamlog_config.yaml with your Ollama server details.")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()