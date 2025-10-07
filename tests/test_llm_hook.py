#!/usr/bin/env python3
"""
Test script for validating LLM hook functionality

This script tests that the LLM hook:
1. Triggers on undefined predicates
2. Generates reasonable rules
3. Uses those rules to answer queries
"""

import sys
from dreamlog import DreamLogEngine
from dreamlog.llm_providers import create_provider
from dreamlog.llm_hook import LLMHook
from dreamlog.prefix_parser import parse_s_expression

def test_family_relations():
    """Test LLM hook on family relations"""
    print("=" * 70)
    print("TEST: Family Relations - LLM Hook")
    print("=" * 70)

    # Create LLM provider
    provider = create_provider(
        provider_type="ollama",
        base_url="http://192.168.0.225:11434",
        model="phi4-mini:latest"
    )

    # Create hook with debug enabled
    hook = LLMHook(provider, debug=True)

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
        print(f"   ✓ {fact_str}")

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
        print("   ❌ No solutions found!")

    # Show generated rules
    print("\n4. Generated rules:")
    if 'grandparent' in engine.kb.rules:
        for rule in engine.kb.rules['grandparent']:
            print(f"   {rule}")
    else:
        print("   No rules generated!")

    return len(results) > 0

def test_multiple_inferences():
    """Test cascading LLM inferences"""
    print("\n" + "=" * 70)
    print("TEST: Multiple Cascading Inferences")
    print("=" * 70)

    provider = create_provider(
        provider_type="ollama",
        base_url="http://192.168.0.225:11434",
        model="phi4-mini:latest"
    )

    hook = LLMHook(provider, debug=True)
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
        print(f"   ✓ {fact_str}")

    # Query for ancestor (may require multiple rule generations)
    print("\n2. Querying for ancestors:")
    print("-" * 70)
    query = parse_s_expression("(ancestor john alice)")
    results = engine.query([query])

    print("\n3. Results:")
    if results:
        print(f"   ✓ Found {len(results)} solution(s)")
    else:
        print("   ❌ No solutions found!")

    print("\n4. All generated rules:")
    for functor, rules in engine.kb.rules.items():
        print(f"\n   Rules for '{functor}':")
        for rule in rules:
            print(f"      {rule}")

    return len(results) > 0

def test_llm_quality():
    """Test the quality of LLM-generated rules"""
    print("\n" + "=" * 70)
    print("TEST: LLM Rule Quality")
    print("=" * 70)

    provider = create_provider(
        provider_type="ollama",
        base_url="http://192.168.0.225:11434",
        model="phi4-mini:latest"
    )

    hook = LLMHook(provider, debug=True)
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
        print(f"   ✓ Correctly inferred sibling relationship ({len(results)} solutions)")
    else:
        print("   ❌ Failed to infer sibling relationship")

    # Check if the rule makes sense
    if 'sibling' in engine.kb.rules:
        print("\n2. Generated sibling rule:")
        for rule in engine.kb.rules['sibling']:
            print(f"   {rule}")
            # A good sibling rule should involve a shared parent
            rule_str = str(rule)
            if 'parent' in rule_str.lower():
                print("   ✓ Rule references parent relationship (good!)")
            else:
                print("   ⚠ Rule doesn't reference parent (may be incorrect)")

    return len(results) > 0

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DreamLog LLM Hook Validation Tests")
    print("=" * 70)

    results = {
        "Family Relations": test_family_relations(),
        "Multiple Inferences": test_multiple_inferences(),
        "LLM Quality": test_llm_quality()
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)
