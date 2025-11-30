#!/usr/bin/env python3
"""
Example configuration for using Ollama with DreamLog.

Run Ollama on your network machine:
    ollama serve  # Default port 11434
    ollama pull llama2  # or phi3, mistral, etc.

Then use this script to test DreamLog with Ollama.
"""

from dreamlog import dreamlog
from dreamlog.llm_providers import OllamaProvider, create_provider
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.config import DreamLogConfig, get_config

# Load configuration from dreamlog_config.yaml
config = DreamLogConfig.load()

# Create Ollama provider from config
# You can either use the config or override manually
ollama_provider = create_provider(
    config.provider.provider,
    base_url=config.provider.base_url,
    model=config.provider.model,
    temperature=config.provider.temperature,
    max_tokens=config.provider.max_tokens
)

# Alternative: manually configure if you want to override config
# ollama_provider = OllamaProvider(
#     base_url="http://192.168.1.100:11434",  # Your network IP
#     model="llama2",
#     temperature=0.3
# )

def test_basic_llm_generation():
    """Test that Ollama can generate facts and rules."""
    print("Testing basic LLM generation...")
    
    # Create KB with Ollama support
    kb = dreamlog(llm_provider=ollama_provider)
    
    # Add some base facts
    kb.parse("""
    (parent john mary)
    (parent mary alice)
    """)
    
    # Query for undefined predicate - should trigger Ollama
    print("\nQuerying for undefined 'grandparent' predicate...")
    results = list(kb.query("grandparent", "john", "X"))
    
    if results:
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  {r}")
    else:
        print("No results found - checking if rule was generated...")
        # Check if Ollama generated the grandparent rule
        for rule in kb.engine.kb.rules:
            if rule.head.functor == "grandparent":
                print(f"Generated rule: {rule}")

def test_dreaming_with_ollama():
    """Test dream cycles with Ollama."""
    print("\n\nTesting dream cycles with Ollama...")
    
    # Create KB with some redundancy
    kb = dreamlog()
    kb.parse("""
    (parent john mary)
    (parent john bob)
    (parent mary alice)
    (parent bob charlie)
    
    (male john)
    (male bob)
    (male charlie)
    
    (female mary)
    (female alice)
    
    (father X Y) :- (parent X Y), (male X)
    (mother X Y) :- (parent X Y), (female X)
    (son X Y) :- (parent Y X), (male X)
    (daughter X Y) :- (parent Y X), (female X)
    """)
    
    print(f"Initial KB: {len(kb.engine.kb.facts)} facts, {len(kb.engine.kb.rules)} rules")
    
    # Create dreamer with Ollama
    dreamer = KnowledgeBaseDreamer(ollama_provider)
    
    # Run a dream cycle
    print("\nRunning dream cycle...")
    session = dreamer.dream(
        kb.engine.kb,
        dream_cycles=1,
        exploration_samples=2,
        focus="compression",
        verify=False  # Skip verification for testing
    )
    
    print(f"\nDream session results:")
    print(f"  - Compression ratio: {session.compression_ratio:.1%}")
    print(f"  - Insights found: {len(session.insights)}")
    print(f"  - Exploration paths: {session.exploration_paths}")
    
    for i, insight in enumerate(session.insights[:3], 1):
        print(f"\n  Insight {i}:")
        print(f"    Type: {insight.type}")
        print(f"    Description: {insight.description}")
        print(f"    Compression: {insight.compression_ratio:.1f}x")

def test_query_with_context():
    """Test that context is properly provided to Ollama."""
    print("\n\nTesting context extraction for Ollama...")
    
    kb = dreamlog(llm_provider=ollama_provider)
    
    # Add diverse facts to test context sampling
    kb.parse("""
    (student alice cs101)
    (student bob cs101)
    (student alice math201)
    
    (grade alice cs101 95)
    (grade bob cs101 87)
    
    (prerequisite cs101 cs201)
    (prerequisite math101 math201)
    """)
    
    # Query for undefined predicate that needs context
    print("\nQuerying for 'passing' predicate (undefined)...")
    results = list(kb.query("passing", "alice", "cs101"))
    
    # Check what was generated
    print("\nGenerated facts:")
    for fact in kb.engine.kb.facts:
        if "passing" in str(fact.term):
            print(f"  {fact.term}")
    
    print("\nGenerated rules:")
    for rule in kb.engine.kb.rules:
        if "passing" in str(rule.head):
            print(f"  {rule.head} :- {', '.join(str(b) for b in rule.body)}")

if __name__ == "__main__":
    print("DreamLog + Ollama Integration Test")
    print("===================================\n")
    
    print("Make sure Ollama is running on your network machine:")
    print("  ollama serve")
    print("  ollama pull llama2  # or your preferred model\n")
    
    try:
        # Test basic generation
        test_basic_llm_generation()
        
        # Test dreaming
        test_dreaming_with_ollama()
        
        # Test context extraction
        test_query_with_context()
        
        print("\n\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that Ollama is running: curl http://localhost:11434/api/generate -d '{\"model\":\"llama2\",\"prompt\":\"test\"}'")
        print("2. Update the base_url in this script to match your Ollama server")
        print("3. Ensure you have a model pulled: ollama list")