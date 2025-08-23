#!/usr/bin/env python3
"""
DreamLog Wake-Sleep Cycle Demonstration

This example shows how DreamLog alternates between:
1. Wake phase: Using knowledge to answer queries
2. Sleep phase: Dreaming to optimize and compress knowledge

Run with: python examples/wake_sleep_demo.py
"""

from dreamlog.pythonic import dreamlog
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.llm_providers import MockLLMProvider
import time


def print_phase(phase_name):
    """Pretty print phase transitions"""
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}\n")


def print_kb_stats(kb, label=""):
    """Print knowledge base statistics"""
    stats = kb.stats
    print(f"{label} Knowledge Base Statistics:")
    print(f"  - Facts: {stats.get('num_facts', 0)}")
    print(f"  - Rules: {stats.get('num_rules', 0)}")
    print(f"  - Total items: {stats.get('total_items', 0)}")
    print()


def wake_phase(kb, queries):
    """
    Wake Phase: Use existing knowledge to answer queries
    """
    print_phase("🌞 WAKE PHASE - Exploiting Knowledge")
    
    print("Answering queries with current knowledge...")
    results = {}
    
    for query in queries:
        print(f"\nQuery: {query}")
        start_time = time.time()
        
        # Parse and execute query
        answers = list(kb.query(*query))
        elapsed = time.time() - start_time
        
        print(f"  Found {len(answers)} answer(s) in {elapsed:.3f}s")
        for i, answer in enumerate(answers[:3]):  # Show first 3
            print(f"    {i+1}. {answer}")
        
        results[str(query)] = {
            'count': len(answers),
            'time': elapsed,
            'sample': answers[:3]
        }
    
    return results


def sleep_phase(kb, dreamer):
    """
    Sleep Phase: Dream to optimize knowledge
    """
    print_phase("🌙 SLEEP PHASE - Dreaming for Optimization")
    
    print("Entering dream state...")
    print("Exploring different reorganization strategies...\n")
    
    # Dream with multiple exploration paths
    session = dreamer.dream(
        kb,
        dream_cycles=2,
        exploration_samples=3,
        focus="all",
        verify=True
    )
    
    print(f"Dream Results:")
    print(f"  - Explored {session.exploration_paths} different optimization paths")
    print(f"  - Found {len(session.insights)} insights")
    print(f"  - Compression ratio: {session.compression_ratio:.1%}")
    print(f"  - Generalization score: {session.generalization_score:.2f}")
    
    if session.verification:
        print(f"\nBehavior Verification:")
        print(f"  - Preserved: {session.verification.preserved}")
        print(f"  - Similarity: {session.verification.similarity_score:.2%}")
        
        if session.verification.improvements:
            print(f"  - Improvements found:")
            for imp in session.verification.improvements[:3]:
                print(f"      • {imp}")
    
    # Show discovered insights
    print(f"\nKey Insights Discovered:")
    for i, insight in enumerate(session.insights[:5], 1):
        print(f"  {i}. {insight.type.upper()}: {insight.description}")
        print(f"     - Compression: {insight.compression_ratio:.1f}x")
        print(f"     - Coverage gain: {insight.coverage_gain:.1f}x")
        if insight.verified:
            print(f"     - ✓ Verified")
    
    return session


def compare_performance(wake_results_before, wake_results_after):
    """Compare performance before and after dreaming"""
    print_phase("📊 PERFORMANCE COMPARISON")
    
    print("Query Performance Changes:")
    for query in wake_results_before:
        before = wake_results_before[query]
        after = wake_results_after.get(query, before)
        
        time_change = (after['time'] - before['time']) / before['time'] * 100
        count_change = after['count'] - before['count']
        
        print(f"\n  Query: {query}")
        print(f"    Time: {before['time']:.3f}s → {after['time']:.3f}s ({time_change:+.1f}%)")
        print(f"    Results: {before['count']} → {after['count']} ({count_change:+d})")


def main():
    """Run wake-sleep cycle demonstration"""
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         DreamLog Wake-Sleep Cycle Demonstration           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Create initial knowledge base
    print("\n📚 Building Initial Knowledge Base...")
    kb = dreamlog()
    
    # Add facts - some redundant for compression opportunities
    kb.parse("""
    (parent john mary)
    (parent john bob)
    (parent mary alice)
    (parent mary charlie)
    (parent bob david)
    (parent bob emma)
    
    (male john)
    (male bob)
    (male charlie)
    (male david)
    
    (female mary)
    (female alice)
    (female emma)
    """)
    
    # Add rules - some could be generalized
    kb.parse("""
    (father X Y) :- (parent X Y), (male X)
    (mother X Y) :- (parent X Y), (female X)
    
    (son X Y) :- (parent Y X), (male X)
    (daughter X Y) :- (parent Y X), (female X)
    
    (grandfather X Z) :- (father X Y), (parent Y Z)
    (grandmother X Z) :- (mother X Y), (parent Y Z)
    """)
    
    print_kb_stats(kb, "Initial")
    
    # Test queries for wake phase
    test_queries = [
        ("parent", "john", "X"),
        ("grandfather", "john", "X"),
        ("father", "X", "alice"),
        ("daughter", "X", "mary")
    ]
    
    # --- CYCLE 1: WAKE ---
    wake_results_before = wake_phase(kb, test_queries)
    
    # --- CYCLE 1: SLEEP ---
    # Create dreamer with mock LLM
    dreamer = KnowledgeBaseDreamer(MockLLMProvider(knowledge_domain="family"))
    session = sleep_phase(kb, dreamer)
    
    # Apply optimizations if verification passed
    if session.verification and session.verification.similarity_score > 0.9:
        print("\n✅ Applying verified optimizations...")
        optimized_kb = dreamer._apply_insights(kb, session.insights)
        print_kb_stats(optimized_kb, "Optimized")
        
        # --- CYCLE 2: WAKE (with optimized KB) ---
        print_phase("🌞 WAKE PHASE - Using Optimized Knowledge")
        wake_results_after = wake_phase(optimized_kb, test_queries)
        
        # Compare performance
        compare_performance(wake_results_before, wake_results_after)
    else:
        print("\n⚠️ Optimizations not applied (verification failed)")
    
    # --- INSIGHTS ---
    print_phase("💡 KEY INSIGHTS")
    
    print("""
The Wake-Sleep cycle demonstrates how DreamLog:

1. EXPLOITATION (Wake): Efficiently uses current knowledge
2. EXPLORATION (Sleep): Discovers optimizations through dreaming
3. COMPRESSION: Reduces redundancy while preserving behavior
4. ABSTRACTION: Finds general patterns from specific rules
5. VERIFICATION: Ensures changes don't break existing behavior

This creates a self-improving system that gets better through use,
similar to how the brain consolidates memories during sleep.
    """)


if __name__ == "__main__":
    main()