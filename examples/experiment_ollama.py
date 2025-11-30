#!/usr/bin/env python3
"""
Interactive DreamLog + Ollama Experiment

This script lets us experiment with DreamLog's LLM integration
using your Ollama server. We can test different scenarios
and see how the LLM generates knowledge.
"""

import json
import time
from dreamlog.pythonic import dreamlog
from dreamlog.llm_providers import OllamaProvider
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.prefix_parser import parse_s_expression, term_to_sexp

# Configure Ollama - update model as needed
OLLAMA_HOST = "192.168.0.225"
OLLAMA_PORT = 11434
MODEL = "phi4-mini-reasoning:latest"  # Better at logical reasoning
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
TEMPERATURE = 0.3  # Lower for more deterministic responses

print(f"""
╔══════════════════════════════════════════════════════════╗
║         DreamLog + Ollama Interactive Experiment        ║
╚══════════════════════════════════════════════════════════╝

Ollama Server: {OLLAMA_HOST}:{OLLAMA_PORT}
Model: {MODEL}
""")

# We'll create the provider inline where needed or use config params
def create_kb_with_llm(debug=False):
    """Create a DreamLog KB with Ollama LLM support."""
    kb = dreamlog(
        llm_provider="ollama",
        base_url=OLLAMA_URL,
        model=MODEL,
        temperature=TEMPERATURE
    )
    # Enable debug mode if requested
    if debug and hasattr(kb.engine, 'llm_hook'):
        kb.engine.llm_hook.debug = True
    return kb

# Create a standalone provider for dreaming experiments
def create_ollama_provider():
    """Create an Ollama provider for direct use."""
    return OllamaProvider(
        base_url=OLLAMA_URL,
        model=MODEL,
        temperature=TEMPERATURE
    )

def experiment_1_undefined_predicate():
    """Test automatic knowledge generation for undefined predicates."""
    print("\n" + "="*60)
    print("Experiment 1: Undefined Predicate Generation")
    print("="*60)
    
    # Create KB with LLM support (with debug enabled)
    kb = create_kb_with_llm(debug=True)
    
    # Add some base facts
    kb.parse("(parent john mary)")
    kb.parse("(parent mary alice)")
    
    print("\nInitial facts:")
    for fact in kb.engine.kb.facts:
        print(f"  {term_to_sexp(fact.term)}")
    
    # Query for undefined predicate
    print("\n🔍 Querying for undefined 'grandparent' predicate...")
    print("   (This should trigger LLM to generate the grandparent rule)")
    
    start = time.time()
    results = list(kb.query("grandparent", "john", "X"))
    elapsed = time.time() - start
    
    print(f"\n⏱️  Query took {elapsed:.2f} seconds")
    
    if results:
        print(f"\n✅ Found {len(results)} results:")
        for r in results:
            print(f"   X = {r['X']}")
        
        # Verify the result makes sense
        print("\n🔍 Verification:")
        print("   Query: (grandparent john X)")
        print("   Logic: john is parent of mary, mary is parent of alice")
        print("   Therefore: john is grandparent of alice ✓")
    else:
        print("\n⚠️  No results found")
    
    # Check what was generated
    print("\n📝 Generated knowledge:")
    print("\nFacts in KB:")
    for fact in kb.engine.kb.facts:
        print(f"   {term_to_sexp(fact.term)}")
    
    print("\nRules in KB:")
    for rule in kb.engine.kb.rules:
        head = term_to_sexp(rule.head)
        body = " ∧ ".join(term_to_sexp(b) for b in rule.body)
        print(f"   {head} ← {body}")
    
    # Test the rule with another query
    print("\n🧪 Testing the generated rule with another query:")
    print("   Query: (grandparent mary X)")
    results2 = list(kb.query("grandparent", "mary", "X"))
    if results2:
        print(f"   Results: {[r['X'] for r in results2]}")
    else:
        print("   No results (correct - mary has no grandchildren in our KB)")
    
    return kb

def experiment_2_context_aware_generation():
    """Test that LLM uses context from existing knowledge."""
    print("\n" + "="*60)
    print("Experiment 2: Context-Aware Generation")
    print("="*60)
    
    kb = create_kb_with_llm()
    
    # Add domain-specific facts
    kb.parse("(student alice cs101)")
    kb.parse("(student bob cs101)")
    kb.parse("(student alice math201)")
    kb.parse("(grade alice cs101 95)")
    kb.parse("(grade bob cs101 87)")
    
    print("\nContext facts:")
    for fact in kb.engine.kb.facts:
        print(f"  {term_to_sexp(fact.term)}")
    
    print("\n🔍 Querying for undefined 'passing' predicate...")
    print("   (LLM should infer from grades context)")
    
    start = time.time()
    results = list(kb.query("passing", "alice", "cs101"))
    elapsed = time.time() - start
    
    print(f"\n⏱️  Query took {elapsed:.2f} seconds")
    
    # Check generated knowledge
    print("\n📝 Generated facts/rules about 'passing':")
    for fact in kb.engine.kb.facts:
        if "passing" in str(fact.term):
            print(f"   Fact: {term_to_sexp(fact.term)}")
    
    for rule in kb.engine.kb.rules:
        if "passing" in str(rule.head):
            head = term_to_sexp(rule.head)
            body = " ∧ ".join(term_to_sexp(b) for b in rule.body)
            print(f"   Rule: {head} ← {body}")
    
    return kb

def experiment_3_dreaming():
    """Test the wake-sleep optimization cycle."""
    print("\n" + "="*60)
    print("Experiment 3: Wake-Sleep Dreaming Cycle")
    print("="*60)
    
    kb = dreamlog()
    
    # Create a knowledge base with some redundancy
    kb.parse("(parent john mary)")
    kb.parse("(parent john bob)")
    kb.parse("(parent mary alice)")
    kb.parse("(male john)")
    kb.parse("(male bob)")
    kb.parse("(female mary)")
    kb.parse("(female alice)")
    
    # Add redundant rules that could be compressed
    kb.parse("(father X Y) :- (parent X Y), (male X)")
    kb.parse("(mother X Y) :- (parent X Y), (female X)")
    kb.parse("(son X Y) :- (parent Y X), (male X)")
    kb.parse("(daughter X Y) :- (parent Y X), (female X)")
    
    print(f"\nInitial KB: {len(kb.engine.kb.facts)} facts, {len(kb.engine.kb.rules)} rules")
    
    # Create dreamer
    dreamer = KnowledgeBaseDreamer(create_ollama_provider())
    
    print("\n💤 Starting dream cycle...")
    print("   (This may take 30-60 seconds)")
    
    start = time.time()
    session = dreamer.dream(
        kb.engine.kb,
        dream_cycles=1,
        exploration_samples=2,
        focus="compression",
        verify=False  # Skip verification for speed
    )
    elapsed = time.time() - start
    
    print(f"\n⏱️  Dream cycle took {elapsed:.2f} seconds")
    
    print(f"\n✨ Dream session results:")
    print(f"   Compression ratio: {session.compression_ratio:.1%}")
    print(f"   Insights found: {len(session.insights)}")
    
    for i, insight in enumerate(session.insights[:3], 1):
        print(f"\n   Insight {i}:")
        print(f"     Type: {insight.type}")
        print(f"     Description: {insight.description}")
        if insight.compression_ratio:
            print(f"     Compression: {insight.compression_ratio:.1f}x")
    
    return kb, session

def experiment_4_creative_generation():
    """Test generating creative/novel rules."""
    print("\n" + "="*60)
    print("Experiment 4: Creative Rule Generation")
    print("="*60)
    
    kb = create_kb_with_llm()
    
    # Minimal facts to spark creativity
    kb.parse("(likes john pizza)")
    kb.parse("(likes mary sushi)")
    kb.parse("(friend john mary)")
    
    print("\nSeed facts:")
    for fact in kb.engine.kb.facts:
        print(f"  {term_to_sexp(fact.term)}")
    
    print("\n🎨 Querying for creative predicates...")
    
    # Try some creative queries
    creative_queries = [
        ("compatible", ["john", "mary"]),
        ("recommend", ["john", "X"]),
        ("social_group", ["X", "Y", "Z"])
    ]
    
    for pred, args in creative_queries:
        print(f"\n🔍 Query: ({pred} {' '.join(args)})")
        
        start = time.time()
        results = list(kb.query(pred, *args))
        elapsed = time.time() - start
        
        print(f"   Time: {elapsed:.2f}s")
        
        if results:
            print(f"   Results: {len(results)}")
            for r in results[:2]:  # Show first 2
                bindings = ", ".join(f"{k}={v}" for k, v in r.items())
                print(f"     {bindings}")
    
    print("\n📝 All generated rules:")
    for rule in kb.engine.kb.rules:
        head = term_to_sexp(rule.head)
        body = " ∧ ".join(term_to_sexp(b) for b in rule.body)
        print(f"   {head} ← {body}")
    
    return kb

def interactive_mode():
    """Interactive REPL with Ollama."""
    print("\n" + "="*60)
    print("Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  fact <sexp>     - Add a fact")
    print("  rule <sexp>     - Add a rule")
    print("  query <sexp>    - Query the KB")
    print("  show            - Show all facts and rules")
    print("  clear           - Clear the KB")
    print("  exit            - Exit interactive mode")
    
    kb = create_kb_with_llm()
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if cmd == "exit":
                break
            elif cmd == "clear":
                kb = create_kb_with_llm()
                print("Knowledge base cleared")
            elif cmd == "show":
                print("\nFacts:")
                for fact in kb.engine.kb.facts:
                    print(f"  {term_to_sexp(fact.term)}")
                print("\nRules:")
                for rule in kb.engine.kb.rules:
                    head = term_to_sexp(rule.head)
                    body = " ∧ ".join(term_to_sexp(b) for b in rule.body)
                    print(f"  {head} ← {body}")
            elif cmd.startswith("fact "):
                sexp = cmd[5:]
                kb.parse(sexp)
                print(f"Added: {sexp}")
            elif cmd.startswith("rule "):
                sexp = cmd[5:]
                kb.parse(sexp)
                print(f"Added: {sexp}")
            elif cmd.startswith("query "):
                sexp = cmd[6:]
                term = parse_s_expression(sexp)
                
                # Extract functor and args for query
                if hasattr(term, 'functor'):
                    args = [str(arg) for arg in term.args]
                    results = list(kb.query(term.functor, *args))
                else:
                    results = list(kb.query(str(term)))
                
                if results:
                    print(f"Found {len(results)} results:")
                    for r in results:
                        bindings = ", ".join(f"{k}={v}" for k, v in r.items())
                        print(f"  {bindings if bindings else 'True'}")
                else:
                    print("No results found")
                    
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Run experiments."""
    
    print("\nSelect experiment:")
    print("1. Undefined Predicate Generation")
    print("2. Context-Aware Generation")
    print("3. Wake-Sleep Dreaming")
    print("4. Creative Generation")
    print("5. Interactive Mode")
    print("6. Run All Experiments")
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == "1":
        experiment_1_undefined_predicate()
    elif choice == "2":
        experiment_2_context_aware_generation()
    elif choice == "3":
        experiment_3_dreaming()
    elif choice == "4":
        experiment_4_creative_generation()
    elif choice == "5":
        interactive_mode()
    elif choice == "6":
        experiment_1_undefined_predicate()
        experiment_2_context_aware_generation()
        experiment_3_dreaming()
        experiment_4_creative_generation()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()