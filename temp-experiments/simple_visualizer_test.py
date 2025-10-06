#!/usr/bin/env python3
"""
Simple test to verify visualizer functionality without server
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from integrations.visualizer.events import EventEmitter, EventType, get_global_emitter
from integrations.visualizer.instrumented_engine import InstrumentedDreamLogEngine
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Fact

def test_events():
    """Test if events are being emitted"""
    print("🔬 Testing Event System")
    print("=" * 30)
    
    # Set up event listener
    emitter = get_global_emitter()
    events_received = []
    
    def event_listener(event):
        events_received.append(event)
        print(f"📡 Event: {event.type} - {event.data.get('description', 'No description')}")
    
    emitter.add_listener(EventType.QUERY_START, event_listener)
    emitter.add_listener(EventType.FACT_ADDED, event_listener)
    emitter.add_listener(EventType.SOLUTION_FOUND, event_listener)
    
    # Create instrumented engine
    engine = InstrumentedDreamLogEngine()
    print("✅ Created instrumented engine")
    
    # Add a fact
    fact_term = compound("parent", atom("john"), atom("mary"))
    fact = Fact(fact_term)
    engine.add_fact(fact)
    print(f"✅ Added fact: {fact_term}")
    
    # Query
    query_term = compound("parent", var("X"), var("Y"))
    print(f"✅ Querying: {query_term}")
    solutions = list(engine.query([query_term]))
    print(f"✅ Found {len(solutions)} solutions")
    
    print(f"\n📊 Total events received: {len(events_received)}")
    
    if len(events_received) > 0:
        print("✅ Event system working!")
        return True
    else:
        print("❌ No events received - event system not working")
        return False

if __name__ == "__main__":
    success = test_events()
    if success:
        print("\n🎉 Events are working! The visualizer should show activity.")
    else:
        print("\n❌ Event system broken - this explains why visualizer is blank.")