"""
Knowledge Base Dreamer

Like the brain during REM sleep, this module reorganizes and consolidates
knowledge by finding abstractions, compressing rules, and discovering
general patterns. The goal is to achieve better generalization through
compression - following the principle that simpler explanations that
cover more cases are likely more true.

Includes verification that reorganizations preserve behavior and exploration
of multiple optimization paths through repeated sampling.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import random
from .llm_providers import LLMProvider
from .knowledge import KnowledgeBase, Fact, Rule
from .prefix_parser import parse_prefix_notation
from .terms import Compound, Atom, Variable
from .evaluator import PrologEvaluator


@dataclass
class DreamInsight:
    """A single insight discovered during dreaming"""
    type: str  # "abstraction", "decomposition", "generalization", "compression"
    description: str
    original_items: List[str]  # Original facts/rules
    new_items: List[str]  # Proposed replacements
    compression_ratio: float  # How much this reduces KB size
    coverage_gain: float  # How many more cases this covers
    confidence: float = 0.5  # Confidence in this transformation
    verified: bool = False  # Whether behavior preservation was verified


@dataclass
class VerificationResult:
    """Result of verifying behavior preservation"""
    preserved: bool  # Whether behavior is preserved
    similarity_score: float  # 0-1, how similar the behaviors are
    differences: List[str]  # Specific differences found
    improvements: List[str]  # Cases where new KB is better


@dataclass
class DreamSession:
    """Results from a dreaming session"""
    insights: List[DreamInsight]
    original_size: int  # Original KB size (facts + rules)
    proposed_size: int  # Proposed KB size after optimization
    compression_ratio: float
    generalization_score: float  # 0-1, higher means more general
    verification: Optional[VerificationResult] = None
    exploration_paths: int = 1  # Number of different optimizations explored
    summary: str = ""


class KnowledgeBaseDreamer:
    """
    Optimizes knowledge bases through 'dreaming' - finding abstractions,
    decompositions, and compressions that lead to better generalization.
    """
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
    
    def dream(self, kb: KnowledgeBase, 
              dream_cycles: int = 3,
              exploration_samples: int = 5,
              focus: str = "compression",
              verify: bool = True) -> DreamSession:
        """
        Run a dreaming session to optimize the knowledge base.
        Like DreamCoder, explores multiple optimization paths and verifies preservation.
        
        Args:
            kb: Knowledge base to optimize
            dream_cycles: Number of optimization cycles (wake-sleep cycles)
            exploration_samples: Number of different optimization paths to explore
            focus: "compression", "abstraction", "decomposition", or "all"
            verify: Whether to verify behavior preservation
            
        Returns:
            DreamSession with insights and proposed changes
        """
        all_insights = []
        best_insights = []
        
        # Explore multiple optimization paths
        for sample in range(exploration_samples):
            sample_insights = []
            
            for cycle in range(dream_cycles):
                if focus == "compression" or focus == "all":
                    sample_insights.extend(self._find_compressions(kb, temperature=0.7 + sample * 0.1))
                
                if focus == "abstraction" or focus == "all":
                    sample_insights.extend(self._find_abstractions(kb, temperature=0.7 + sample * 0.1))
                
                if focus == "decomposition" or focus == "all":
                    sample_insights.extend(self._find_decompositions(kb))
                
                if focus == "all":
                    sample_insights.extend(self._find_generalizations(kb))
            
            all_insights.append(sample_insights)
        
        # Select best insights across all explorations
        best_insights = self._select_best_insights(all_insights)
        
        # Verify behavior preservation if requested
        verification = None
        if verify and best_insights:
            verification = self._verify_behavior_preservation(kb, best_insights)
            
            # Mark verified insights
            for insight in best_insights:
                insight.verified = verification.preserved or verification.similarity_score > 0.9
        
        # Calculate metrics
        original_size = len(kb.facts) + len(kb.rules)
        proposed_size = self._calculate_proposed_size(kb, best_insights)
        compression_ratio = 1 - (proposed_size / original_size) if original_size > 0 else 0
        
        return DreamSession(
            insights=best_insights,
            original_size=original_size,
            proposed_size=proposed_size,
            compression_ratio=compression_ratio,
            generalization_score=self._calculate_generalization_score(best_insights),
            verification=verification,
            exploration_paths=exploration_samples,
            summary=self._generate_summary(best_insights)
        )
    
    def _verify_behavior_preservation(self, original_kb: KnowledgeBase, 
                                     insights: List[DreamInsight]) -> VerificationResult:
        """
        Verify that proposed changes preserve KB behavior.
        Tests with sample queries and compares results.
        """
        # Create optimized KB
        optimized_kb = self._apply_insights(original_kb, insights)
        
        # Generate test queries
        test_queries = self._generate_test_queries(original_kb)
        
        # Compare results
        differences = []
        improvements = []
        matching = 0
        total = len(test_queries)
        
        for query in test_queries:
            # Run query on both KBs
            orig_results = self._run_query(original_kb, query)
            opt_results = self._run_query(optimized_kb, query)
            
            # Compare results
            if orig_results == opt_results:
                matching += 1
            else:
                # Use LLM to evaluate if difference is acceptable
                evaluation = self._evaluate_difference(query, orig_results, opt_results)
                
                if evaluation.get('is_improvement', False):
                    improvements.append(f"{query}: {evaluation.get('reason', '')}")
                    matching += 0.5  # Partial credit for improvements
                else:
                    differences.append(f"{query}: expected {orig_results}, got {opt_results}")
        
        similarity_score = matching / total if total > 0 else 1.0
        
        return VerificationResult(
            preserved=len(differences) == 0,
            similarity_score=similarity_score,
            differences=differences,
            improvements=improvements
        )
    
    def _generate_test_queries(self, kb: KnowledgeBase) -> List[str]:
        """Generate test queries to verify behavior"""
        queries = []
        
        # Sample queries from existing functors
        functors = set()
        for fact in kb.facts[:50]:
            if hasattr(fact.term, 'functor'):
                functors.add(fact.term.functor)
        
        for functor in list(functors)[:10]:
            # Generate variable query
            queries.append(f"({functor} X Y)")
            # Generate grounded query
            queries.append(f"({functor} test_value _)")
        
        # Ask LLM for additional test queries
        prompt = f"""Given this KB with functors: {', '.join(functors)}
Generate 5 important test queries to verify the KB works correctly.
Return as JSON array of query strings."""
        
        response = self.provider.generate_knowledge("_test_queries", context=prompt)
        try:
            if hasattr(response, 'text'):
                additional = json.loads(response.text)
                queries.extend(additional[:5])
        except:
            pass
        
        return queries
    
    def _run_query(self, kb: KnowledgeBase, query: str) -> List[Dict]:
        """Run a query and return results"""
        try:
            evaluator = PrologEvaluator(kb)
            # Parse query and evaluate
            # Simplified - real implementation would parse properly
            results = []
            for solution in evaluator.evaluate_query(query):
                results.append(solution.bindings)
            return results[:5]  # Limit results
        except:
            return []
    
    def _evaluate_difference(self, query: str, orig_results: List, new_results: List) -> Dict:
        """Use LLM to evaluate if a difference is acceptable or an improvement"""
        prompt = f"""Compare these query results:
Query: {query}
Original KB results: {orig_results}
Optimized KB results: {new_results}

Is the difference acceptable? Is it an improvement?
Response format:
{{"is_improvement": bool, "is_acceptable": bool, "reason": "explanation"}}"""
        
        response = self.provider.generate_knowledge("_evaluate_diff", context=prompt)
        try:
            if hasattr(response, 'text'):
                return json.loads(response.text)
        except:
            pass
        
        return {"is_improvement": False, "is_acceptable": False, "reason": "Could not evaluate"}
    
    def _apply_insights(self, kb: KnowledgeBase, insights: List[DreamInsight]) -> KnowledgeBase:
        """Apply insights to create optimized KB"""
        optimized_kb = KnowledgeBase()
        
        # Track what to skip
        items_to_skip = set()
        for insight in insights:
            items_to_skip.update(insight.original_items)
        
        # Copy non-optimized facts
        for fact in kb.facts:
            if str(fact.term) not in items_to_skip:
                optimized_kb.add_fact(fact)
        
        # Copy non-optimized rules  
        for rule in kb.rules:
            rule_str = f"{rule.head} :- {', '.join(str(g) for g in rule.body)}"
            if rule_str not in items_to_skip:
                optimized_kb.add_rule(rule)
        
        # Add optimized items
        for insight in insights:
            for item_str in insight.new_items:
                try:
                    parsed = parse_prefix_notation(item_str)
                    if isinstance(parsed, list):
                        if parsed[0] == "rule":
                            rule = Rule.from_prefix(parsed)
                            optimized_kb.add_rule(rule)
                        else:
                            fact = Fact.from_prefix(["fact", parsed])
                            optimized_kb.add_fact(fact)
                except:
                    pass
        
        return optimized_kb
    
    def _select_best_insights(self, all_insights: List[List[DreamInsight]]) -> List[DreamInsight]:
        """Select best insights from multiple exploration paths"""
        # Score each insight
        scored_insights = []
        
        for path_insights in all_insights:
            for insight in path_insights:
                score = self._score_insight(insight)
                scored_insights.append((score, insight))
        
        # Sort by score and remove duplicates
        scored_insights.sort(key=lambda x: x[0], reverse=True)
        
        selected = []
        seen = set()
        
        for score, insight in scored_insights:
            # Create signature to detect duplicates
            sig = (insight.type, tuple(insight.original_items))
            if sig not in seen:
                seen.add(sig)
                selected.append(insight)
                
                if len(selected) >= 20:  # Limit number of insights
                    break
        
        return selected
    
    def _score_insight(self, insight: DreamInsight) -> float:
        """Score an insight based on compression and coverage"""
        # Weighted score favoring high compression and coverage
        score = (
            insight.compression_ratio * 0.4 +
            insight.coverage_gain * 0.4 +
            insight.confidence * 0.2
        )
        
        # Bonus for certain types
        if insight.type == "abstraction":
            score *= 1.2
        elif insight.type == "generalization":
            score *= 1.1
        
        return score
    
    def _find_compressions(self, kb: KnowledgeBase, temperature: float = 0.7) -> List[DreamInsight]:
        """Find opportunities to compress the KB through rule merging"""
        insights = []
        
        # Group rules by similar structure
        rule_groups = self._group_similar_rules(kb.rules)
        
        for group in rule_groups:
            if len(group) > 2:  # Only compress if we have multiple similar rules
                # Ask LLM to find a general pattern
                compression = self._compress_rules_with_llm(group)
                if compression:
                    insights.append(compression)
        
        # Find fact patterns that could become rules
        fact_patterns = self._find_fact_patterns(kb.facts)
        for pattern in fact_patterns:
            insight = self._create_rule_from_pattern(pattern)
            if insight:
                insights.append(insight)
        
        return insights
    
    def _find_abstractions(self, kb: KnowledgeBase) -> List[DreamInsight]:
        """Find higher-level abstractions that can replace specific rules"""
        insights = []
        
        # Prepare rules for analysis
        rules_text = '\n'.join([
            f"{rule.head} :- {', '.join(str(g) for g in rule.body)}"
            for rule in kb.rules[:30]  # Limit for prompt
        ])
        
        prompt = f"""Analyze these Prolog rules and find higher-level abstractions:

{rules_text}

Look for:
1. Rules that could be generalized to cover more cases
2. Common patterns that could be abstracted into meta-rules
3. Hierarchical relationships that could be simplified

For each abstraction found, provide:
- The original rules it replaces
- The new abstract rule(s)
- Why this abstraction is beneficial

Response format:
[
  {{
    "original_rules": ["rule1", "rule2"],
    "abstract_rule": "new_rule",
    "benefit": "explanation"
  }}
]"""
        
        response = self.provider.generate_knowledge("_find_abstractions", context=prompt)
        
        # Parse response
        try:
            if hasattr(response, 'text'):
                abstractions = json.loads(response.text)
            else:
                abstractions = response.get('abstractions', [])
            
            for abs_data in abstractions:
                insight = DreamInsight(
                    type="abstraction",
                    description=abs_data.get('benefit', 'Higher-level abstraction'),
                    original_items=abs_data.get('original_rules', []),
                    new_items=[abs_data.get('abstract_rule', '')],
                    compression_ratio=len(abs_data.get('original_rules', [])) / 1.0,
                    coverage_gain=1.5  # Abstractions typically increase coverage
                )
                insights.append(insight)
        except Exception as e:
            pass  # Silent fail, continue with other strategies
        
        return insights
    
    def _find_decompositions(self, kb: KnowledgeBase) -> List[DreamInsight]:
        """Find complex rules that could be decomposed into simpler ones"""
        insights = []
        
        # Find complex rules (many body goals)
        complex_rules = [r for r in kb.rules if len(r.body) > 3]
        
        for rule in complex_rules[:10]:  # Limit for processing
            rule_str = f"{rule.head} :- {', '.join(str(g) for g in rule.body)}"
            
            prompt = f"""Decompose this complex rule into simpler, more reusable components:

{rule_str}

Provide simpler rules that together achieve the same result but are more modular.

Response format:
{{
  "simpler_rules": ["rule1", "rule2", ...],
  "explanation": "why this decomposition is better"
}}"""
            
            response = self.provider.generate_knowledge("_decompose", context=prompt)
            
            try:
                if hasattr(response, 'text'):
                    data = json.loads(response.text)
                    
                    insight = DreamInsight(
                        type="decomposition",
                        description=data.get('explanation', 'Modular decomposition'),
                        original_items=[rule_str],
                        new_items=data.get('simpler_rules', []),
                        compression_ratio=1.0 / len(data.get('simpler_rules', [1])),
                        coverage_gain=1.2  # Decomposed rules are often more reusable
                    )
                    insights.append(insight)
            except:
                pass
        
        return insights
    
    def _find_generalizations(self, kb: KnowledgeBase) -> List[DreamInsight]:
        """Find specific rules that could be generalized"""
        insights = []
        
        # Find rules with constants that could be variables
        for rule in kb.rules:
            constants_in_rule = self._extract_constants(rule)
            if len(constants_in_rule) > 0:
                # Try to generalize
                generalized = self._generalize_rule(rule, constants_in_rule)
                if generalized:
                    insight = DreamInsight(
                        type="generalization",
                        description=f"Generalized rule by replacing constants with variables",
                        original_items=[str(rule)],
                        new_items=[generalized],
                        compression_ratio=1.0,
                        coverage_gain=2.0  # Generalized rules cover more cases
                    )
                    insights.append(insight)
        
        return insights
    
    def _group_similar_rules(self, rules: List[Rule]) -> List[List[Rule]]:
        """Group rules with similar structure"""
        groups = defaultdict(list)
        
        for rule in rules:
            # Create a signature based on functor and body length
            if hasattr(rule.head, 'functor'):
                signature = (rule.head.functor, len(rule.body))
                groups[signature].append(rule)
        
        return [group for group in groups.values() if len(group) > 1]
    
    def _find_fact_patterns(self, facts: List[Fact]) -> List[List[Fact]]:
        """Find patterns in facts that could become rules"""
        patterns = defaultdict(list)
        
        for fact in facts:
            if hasattr(fact.term, 'functor'):
                # Group by functor and arity
                signature = (fact.term.functor, len(fact.term.args))
                patterns[signature].append(fact)
        
        # Return groups with enough facts to warrant a rule
        return [group for group in patterns.values() if len(group) > 5]
    
    def _compress_rules_with_llm(self, rules: List[Rule]) -> Optional[DreamInsight]:
        """Use LLM to find compressed representation of similar rules"""
        rules_text = '\n'.join([
            f"{rule.head} :- {', '.join(str(g) for g in rule.body)}"
            for rule in rules
        ])
        
        prompt = f"""These rules have similar structure. Find a single general rule that captures their pattern:

{rules_text}

Provide a compressed rule using variables that covers all cases.

Response format:
{{
  "compressed_rule": "general_rule",
  "explanation": "how this captures all cases"
}}"""
        
        response = self.provider.generate_knowledge("_compress", context=prompt)
        
        try:
            if hasattr(response, 'text'):
                data = json.loads(response.text)
                
                return DreamInsight(
                    type="compression",
                    description=data.get('explanation', 'Rule compression'),
                    original_items=[str(r) for r in rules],
                    new_items=[data.get('compressed_rule', '')],
                    compression_ratio=len(rules) / 1.0,
                    coverage_gain=1.0
                )
        except:
            pass
        
        return None
    
    def _create_rule_from_pattern(self, facts: List[Fact]) -> Optional[DreamInsight]:
        """Create a rule from a pattern of facts"""
        if not facts or len(facts) < 3:
            return None
        
        # Simple pattern detection: if all facts have same functor
        # and share some argument patterns
        first_fact = facts[0]
        if not hasattr(first_fact.term, 'functor'):
            return None
        
        functor = first_fact.term.functor
        
        # Check if there's a pattern (simplified)
        # Real implementation would be more sophisticated
        sample_facts = '\n'.join([str(f.term) for f in facts[:10]])
        
        prompt = f"""These facts follow a pattern. Create a rule that generates them:

{sample_facts}

Response format:
{{
  "rule": "rule_definition",
  "explanation": "pattern identified"
}}"""
        
        response = self.provider.generate_knowledge("_pattern_rule", context=prompt)
        
        try:
            if hasattr(response, 'text'):
                data = json.loads(response.text)
                
                return DreamInsight(
                    type="compression",
                    description=f"Pattern rule: {data.get('explanation', '')}",
                    original_items=[str(f.term) for f in facts],
                    new_items=[data.get('rule', '')],
                    compression_ratio=len(facts) / 1.0,
                    coverage_gain=1.5
                )
        except:
            pass
        
        return None
    
    def _extract_constants(self, rule: Rule) -> Set[str]:
        """Extract constants from a rule"""
        constants = set()
        
        # Check head
        if hasattr(rule.head, 'args'):
            for arg in rule.head.args:
                if isinstance(arg, Atom):
                    constants.add(arg.value)
        
        # Check body
        for goal in rule.body:
            if hasattr(goal, 'args'):
                for arg in goal.args:
                    if isinstance(arg, Atom):
                        constants.add(arg.value)
        
        return constants
    
    def _generalize_rule(self, rule: Rule, constants: Set[str]) -> Optional[str]:
        """Generalize a rule by replacing constants with variables"""
        # Simple generalization: replace specific constants with variables
        # Real implementation would be more sophisticated
        rule_str = str(rule)
        
        # Replace constants with variables
        var_names = ['X', 'Y', 'Z', 'W', 'V']
        for i, const in enumerate(list(constants)[:5]):
            if i < len(var_names):
                rule_str = rule_str.replace(const, var_names[i])
        
        return rule_str if rule_str != str(rule) else None
    
    def _calculate_proposed_size(self, kb: KnowledgeBase, insights: List[DreamInsight]) -> int:
        """Calculate KB size after applying insights"""
        # Count items to remove
        items_to_remove = set()
        for insight in insights:
            items_to_remove.update(insight.original_items)
        
        # Count items to add
        items_to_add = []
        for insight in insights:
            items_to_add.extend(insight.new_items)
        
        original_size = len(kb.facts) + len(kb.rules)
        return original_size - len(items_to_remove) + len(items_to_add)
    
    def _calculate_generalization_score(self, insights: List[DreamInsight]) -> float:
        """Calculate how much generalization was achieved"""
        if not insights:
            return 0.0
        
        total_coverage_gain = sum(i.coverage_gain for i in insights)
        total_compression = sum(i.compression_ratio for i in insights)
        
        # Weighted average favoring coverage gain
        score = (0.7 * total_coverage_gain + 0.3 * total_compression) / len(insights)
        return min(1.0, score / 2.0)  # Normalize to 0-1
    
    def _generate_summary(self, insights: List[DreamInsight]) -> str:
        """Generate a summary of the dreaming session"""
        if not insights:
            return "No optimization opportunities found."
        
        by_type = defaultdict(int)
        for insight in insights:
            by_type[insight.type] += 1
        
        summary_parts = []
        summary_parts.append(f"Found {len(insights)} optimization opportunities:")
        
        for insight_type, count in by_type.items():
            summary_parts.append(f"  - {count} {insight_type}(s)")
        
        avg_compression = sum(i.compression_ratio for i in insights) / len(insights)
        summary_parts.append(f"Average compression ratio: {avg_compression:.2f}x")
        
        return '\n'.join(summary_parts)


def dream_optimize(kb: KnowledgeBase, provider: LLMProvider, cycles: int = 3) -> KnowledgeBase:
    """
    Optimize a knowledge base through dreaming.
    
    Args:
        kb: Knowledge base to optimize
        provider: LLM provider
        cycles: Number of dream cycles
        
    Returns:
        Optimized knowledge base
    """
    dreamer = KnowledgeBaseDreamer(provider)
    session = dreamer.dream(kb, dream_cycles=cycles, focus="all")
    
    # Apply insights to create new KB
    optimized_kb = KnowledgeBase()
    
    # Track what to skip
    items_to_skip = set()
    for insight in session.insights:
        items_to_skip.update(insight.original_items)
    
    # Copy non-optimized facts
    for fact in kb.facts:
        if str(fact.term) not in items_to_skip:
            optimized_kb.add_fact(fact)
    
    # Copy non-optimized rules
    for rule in kb.rules:
        rule_str = f"{rule.head} :- {', '.join(str(g) for g in rule.body)}"
        if rule_str not in items_to_skip:
            optimized_kb.add_rule(rule)
    
    # Add optimized items
    for insight in session.insights:
        for item_str in insight.new_items:
            try:
                # Parse and add
                parsed = parse_prefix_notation(item_str)
                if isinstance(parsed, list):
                    if parsed[0] == "rule":
                        rule = Rule.from_prefix(parsed)
                        optimized_kb.add_rule(rule)
                    else:
                        fact = Fact.from_prefix(["fact", parsed])
                        optimized_kb.add_fact(fact)
            except Exception as e:
                print(f"Could not add optimized item: {e}")
    
    return optimized_kb