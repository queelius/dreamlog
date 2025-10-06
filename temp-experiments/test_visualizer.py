#!/usr/bin/env python3
"""
Test script for the DreamLog visualizer
"""

import time
import sys
from pathlib import Path

# Add dreamlog to path
sys.path.insert(0, str(Path(__file__).parent))

from integrations.visualizer.instrumented_engine import InstrumentedDreamLogEngine
from dreamlog.llm_hook import LLMHook
from dreamlog.llm_providers import create_provider
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Fact

def test_visualizer():
    """Test the visualizer with some queries"""
    print("🔬 Testing DreamLog Visualizer")
    print("=" * 50)
    
    # Create engine without LLM for now
    engine = InstrumentedDreamLogEngine()
    
    print("✅ Created instrumented engine")
    
    # Add some facts
    print("\n📝 Adding sample facts...")
    facts = [
        compound("parent", atom("john"), atom("mary")),
        compound("parent", atom("mary"), atom("alice")),
        compound("parent", atom("bob"), atom("charlie")),
    ]
    
    for fact_term in facts:
        fact = Fact(fact_term)
        engine.add_fact(fact)
        print(f"   Added: {fact_term}")
    
    # Query for parents
    print("\n🔍 Querying for parents...")
    query_term = compound("parent", var("X"), var("Y"))
    print(f"   Query: {query_term}")
    
    solutions = list(engine.query([query_term]))
    print(f"   Found {len(solutions)} solutions:")
    for i, sol in enumerate(solutions):
        print(f"     Solution {i+1}: {sol.get_ground_bindings()}")
    
    print("\n🎉 Test completed! Check the visualizer at http://127.0.0.1:8080")
    return True

if __name__ == "__main__":
    test_visualizer()