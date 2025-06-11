from jlog.llm_config import LLMConfig
from jlog.llm_hook import LLMHook
from jlog.evaluator import PrologEvaluator
from jlog.knowledge import KnowledgeBase
from jlog.terms import compound, atom

# Create LLM provider from your config file
provider = LLMConfig.from_file("llm_config.json")

# Create the hook
llm_hook = LLMHook(provider, knowledge_domain="family")

# Create a Prolog evaluator with the LLM hook
kb = KnowledgeBase()
evaluator = PrologEvaluator(kb, unknown_hook=llm_hook)

# Test with an unknown term
unknown_term = compound("grandfather", [atom("john"), atom("mary")])
print(f"Testing LLM generation for: {unknown_term}")

# This should trigger the LLM to generate knowledge about "grandfather"
llm_hook(unknown_term, evaluator)

# Check what was added to the knowledge base
print("\nKnowledge base after LLM generation:")
print(f"Facts: {len(kb.facts)}")
print(f"Rules: {len(kb.rules)}")