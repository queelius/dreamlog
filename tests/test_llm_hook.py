#!/usr/bin/env python3
"""
Test script for validating LLM hook functionality

This script tests that the LLM hook:
1. Triggers on undefined predicates
2. Generates reasonable rules
3. Uses those rules to answer queries

Note: These tests require a real LLM connection and are marked with @pytest.mark.llm
They are skipped by default unless explicitly run with: pytest -m llm
"""

import sys
import os
import pytest
from dreamlog import DreamLogEngine
from dreamlog.llm_providers import create_provider
from dreamlog.llm_hook import LLMHook
from dreamlog.prefix_parser import parse_s_expression
from dreamlog.tfidf_embedding_provider import TfIdfEmbeddingProvider
from dreamlog.prompt_template_system import RULE_EXAMPLES


# Skip these tests by default since they require external LLM service
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS", "").lower() not in ("1", "true", "yes"),
    reason="LLM tests require RUN_LLM_TESTS=1 environment variable"
)


def _run_family_relations_test():
    """
    Core test logic for family relations - returns results for assertions.

    Returns:
        tuple: (results, engine) for verification
    """
    print("=" * 70)
    print("TEST: Family Relations - LLM Hook")
    print("=" * 70)

    # Create LLM provider
    provider = create_provider(
        provider_type="ollama",
        base_url="http://192.168.0.225:11434",
        model="phi4-mini:latest"
    )

    # Create embedding provider for RAG
    embedding_provider = TfIdfEmbeddingProvider(corpus=RULE_EXAMPLES)

    # Create hook with debug enabled
    hook = LLMHook(provider, embedding_provider, debug=True)

    # Create engine
    engine = DreamLogEngine(llm_hook=hook)

    # Add basic parent facts
    facts = [
        "(parent john mary)",
        "(parent john tom)",
        "(parent mary alice)",
        "(parent tom bob)",
        "(parent alice charlie)"
    ]

    print("\n1. Adding parent facts:")
    for fact_str in facts:
        fact = parse_s_expression(fact_str)
        engine.add_fact(fact)
        print(f"   + {fact_str}")

    # Query for grandparents (should trigger LLM)
    print("\n2. Querying for grandparents (will trigger LLM):")
    print("-" * 70)
    query = parse_s_expression("(grandparent X Y)")
    results = engine.query([query])

    print("\n3. Results:")
    if results:
        for i, result in enumerate(results, 1):
            bindings = result.get_ground_bindings()
            print(f"   {i}. grandparent({bindings.get('X', '?')}, {bindings.get('Y', '?')})")
        print(f"\n   Total: {len(results)} solutions")
    else:
        print("   No solutions found!")

    # Show generated rules
    print("\n4. Generated rules:")
    if 'grandparent' in engine.kb.rules:
        for rule in engine.kb.rules['grandparent']:
            print(f"   {rule}")
    else:
        print("   No rules generated!")

    return results, engine


@pytest.mark.llm
def test_family_relations():
    """Test LLM hook on family relations - should find grandparent relationships"""
    results, engine = _run_family_relations_test()

    assert len(results) > 0, "LLM should generate grandparent rules that produce results"
    assert 'grandparent' in engine.kb.rules, "grandparent rule should be generated"

def _run_multiple_inferences_test():
    """
    Core test logic for cascading inferences - returns results for assertions.

    Returns:
        tuple: (results, engine) for verification
    """
    print("\n" + "=" * 70)
    print("TEST: Multiple Cascading Inferences")
    print("=" * 70)

    provider = create_provider(
        provider_type="ollama",
        base_url="http://192.168.0.225:11434",
        model="phi4-mini:latest"
    )

    # Create embedding provider for RAG
    embedding_provider = TfIdfEmbeddingProvider(corpus=RULE_EXAMPLES)

    hook = LLMHook(provider, embedding_provider, debug=True)
    engine = DreamLogEngine(llm_hook=hook)

    # Add basic facts
    facts = [
        "(parent john mary)",
        "(parent mary alice)",
    ]

    print("\n1. Adding minimal facts:")
    for fact_str in facts:
        fact = parse_s_expression(fact_str)
        engine.add_fact(fact)
        print(f"   + {fact_str}")

    # Query for ancestor (may require multiple rule generations)
    print("\n2. Querying for ancestors:")
    print("-" * 70)
    query = parse_s_expression("(ancestor john alice)")
    results = engine.query([query])

    print("\n3. Results:")
    if results:
        print(f"   + Found {len(results)} solution(s)")
    else:
        print("   No solutions found!")

    print("\n4. All generated rules:")
    for rule in engine.kb.rules:
        print(f"   {rule}")

    return results, engine


@pytest.mark.llm
def test_multiple_inferences():
    """Test cascading LLM inferences - should find ancestor relationship"""
    results, engine = _run_multiple_inferences_test()

    assert len(results) > 0, "LLM should generate ancestor rules that produce results"

def _run_llm_quality_test():
    """
    Core test logic for LLM rule quality - returns results for assertions.

    Returns:
        tuple: (results, engine) for verification
    """
    print("\n" + "=" * 70)
    print("TEST: LLM Rule Quality")
    print("=" * 70)

    provider = create_provider(
        provider_type="ollama",
        base_url="http://192.168.0.225:11434",
        model="phi4-mini:latest"
    )

    # Create embedding provider for RAG
    embedding_provider = TfIdfEmbeddingProvider(corpus=RULE_EXAMPLES)

    hook = LLMHook(provider, embedding_provider, debug=True)
    engine = DreamLogEngine(llm_hook=hook)

    # Add facts
    facts = [
        "(parent john mary)",
        "(parent john tom)",
        "(parent mary alice)",
    ]

    for fact_str in facts:
        engine.add_fact(parse_s_expression(fact_str))

    # Query for sibling
    print("\n1. Testing 'sibling' inference:")
    query = parse_s_expression("(sibling mary tom)")
    results = engine.query([query])

    if results:
        print(f"   + Correctly inferred sibling relationship ({len(results)} solutions)")
    else:
        print("   Failed to infer sibling relationship")

    # Check if the rule makes sense
    if 'sibling' in engine.kb.rules:
        print("\n2. Generated sibling rule:")
        for rule in engine.kb.rules['sibling']:
            print(f"   {rule}")
            # A good sibling rule should involve a shared parent
            rule_str = str(rule)
            if 'parent' in rule_str.lower():
                print("   + Rule references parent relationship (good!)")
            else:
                print("   Warning: Rule doesn't reference parent (may be incorrect)")

    return results, engine


@pytest.mark.llm
def test_llm_quality():
    """Test the quality of LLM-generated rules - sibling should use parent"""
    results, engine = _run_llm_quality_test()

    assert len(results) > 0, "LLM should generate sibling rules that produce results"
    assert 'sibling' in engine.kb.rules, "sibling rule should be generated"

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DreamLog LLM Hook Validation Tests")
    print("=" * 70)

    # Run the helper functions and check if results exist
    test_results = {}

    try:
        results, _ = _run_family_relations_test()
        test_results["Family Relations"] = len(results) > 0
    except Exception as e:
        print(f"Family Relations test failed with: {e}")
        test_results["Family Relations"] = False

    try:
        results, _ = _run_multiple_inferences_test()
        test_results["Multiple Inferences"] = len(results) > 0
    except Exception as e:
        print(f"Multiple Inferences test failed with: {e}")
        test_results["Multiple Inferences"] = False

    try:
        results, _ = _run_llm_quality_test()
        test_results["LLM Quality"] = len(results) > 0
    except Exception as e:
        print(f"LLM Quality test failed with: {e}")
        test_results["LLM Quality"] = False

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, passed in test_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")

    total = len(test_results)
    passed = sum(test_results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)
