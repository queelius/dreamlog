# RAG-Based Learning and Meta-Learning System for DreamLog

## Overview

DreamLog uses Retrieval-Augmented Generation (RAG) to create a self-improving system that learns from experience. The RAG system operates in two modes:

1. **Wake Mode (Query Time)**: Fast retrieval for immediate query answering
2. **Dream Mode (Learning Time)**: Deep evaluation and learning through experience replay

This document describes the architecture, insights, and experimental directions for our RAG-based learning system, with a focus on how it integrates with the dream cycle for continuous improvement.

## Core Insights

### 1. Multiple RAG Systems for Different Purposes

We use the same RAG framework and embedding providers for multiple purposes:

- **Example RAG**: Retrieves similar query/answer examples for many-shot learning
- **Prompt Template RAG**: Selects successful prompt templates based on query characteristics  
- **Knowledge Base RAG**: Could even store and retrieve entire knowledge bases for different domains
- **Meta-pattern RAG**: Stores patterns about what approaches work for what types of problems

All use the same `EmbeddingProvider` interface, promoting code reuse and consistency.

### 2. The Recursive Nature of Knowledge Generation

A key insight: When the LLM generates a rule like `(grandparent X,Z) :- (parent X Y), (parent Y Z)`, if `parent` is also undefined, the system will recursively call the LLM to define `parent`. This means:

- The system is **self-bootstrapping** - it builds up required knowledge on demand
- Complex queries naturally **decompose** into simpler ones
- Each LLM call can focus on a **single predicate** without worrying about dependencies
- Examples don't need to be complete - the system fills in gaps as needed

### 3. Context Sensitivity in Example Selection

We identified multiple approaches for example retrieval:

#### Simple Approach: Query-Only Embedding
```python
embedding = embed("(grandparent john X)")
```
- Pros: Simple, lets recursive calls handle dependencies
- Cons: Might generate redundant or conflicting knowledge

#### Context-Aware Approach: Query + KB State
```python
embedding = embed("(grandparent john X) | KB: parent facts exist, family domain")
```
- Pros: Can leverage existing knowledge, maintain consistency
- Cons: More complex, larger embedding space

#### Hybrid Approach: Multiple Embeddings
```python
query_embedding = embed("(grandparent john X)")
context_embedding = embed("facts: parent, child | rules: sibling")
combined = concatenate(query_embedding, context_embedding)
```
- Pros: Best of both worlds, can weight different signals
- Cons: Requires more computation and storage

### 4. Learning Through Success Tracking

The system tracks which examples and templates lead to successful inferences:

```python
@dataclass
class RAGItem:
    content: Any
    use_count: int = 0
    success_count: int = 0
    
    @property
    def confidence_score(self) -> float:
        if self.use_count < 3:
            return 0.0  # Not enough data
        success_rate = self.success_count / self.use_count
        usage_factor = min(1.0, self.use_count / 10.0)
        return success_rate * usage_factor
```

This creates a feedback loop where successful patterns are reinforced.

### 5. Probability-Weighted Sampling

Instead of always selecting the top-k most similar examples, we use probability-weighted sampling:

```python
def sample_by_similarity(similarities, temperature=1.0):
    # Convert similarities to probabilities
    scores = np.array([sim for _, sim in similarities])
    scores = scores / temperature
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)
    
    # Sample based on probabilities
    return np.random.choice(items, p=probabilities)
```

Benefits:
- **Exploration vs Exploitation**: Temperature controls randomness
- **Diversity**: Different examples on each attempt
- **Robustness**: Not stuck with potentially bad top matches

### 6. Meta-Learning Patterns

The system can learn meta-patterns about what works when:

```python
class MetaLearningTracker:
    def record_usage(query_type, selected_items, success):
        # Track what works for different query types
        
    def get_recommendations(query_type):
        # Return items that historically work for this type
```

Over time, the system learns:
- Recursive predicates need recursive examples
- Transitive relations need transitivity examples  
- Domain-specific patterns (family vs. graph vs. academic)

### 7. Experience Replay During Dream Cycles

The RAG learning system implements a form of **experience replay**, similar to reinforcement learning:

```python
class ExperienceReplay:
    """
    During dream cycles, replay past experiences to:
    - Re-evaluate with more computation time
    - Update confidence scores
    - Learn from patterns across multiple queries
    """
    
    def replay_experiences(self, activity_log, batch_size=32):
        # Sample batch of past experiences
        experiences = random.sample(activity_log, batch_size)
        
        for exp in experiences:
            # Re-evaluate with multiple methods
            success_score = self.deep_evaluate(exp)
            
            # Update RAG databases
            self.update_rag_items(exp, success_score)
            
            # Extract meta-patterns
            self.extract_patterns(exp)
        
        return self.consolidate_learning()
```

This experience replay happens during the **dream cycle**, not during wake time:

- **Wake Phase**: Log experiences quickly, focus on answering queries
- **Dream Phase**: Replay experiences, evaluate deeply, update knowledge

Benefits of dream-time experience replay:
- **No time pressure**: Can use expensive evaluation methods
- **Batch learning**: Learn patterns across multiple experiences
- **Consolidation**: Similar to how biological sleep consolidates memories
- **Continuous improvement**: System gets better even without explicit feedback

### 8. User Feedback as Ground Truth

While the system can self-evaluate during dream cycles, **user feedback remains the gold standard** for preventing drift:

```python
class GroundTruthManager:
    """
    Manages user feedback as the ultimate source of truth.
    This prevents the system from drifting into self-consistent but incorrect patterns.
    """
    
    def __init__(self):
        self.user_labels = []  # Permanent record of user feedback
        self.trust_scores = {
            "user": 1.0,      # Absolute trust
            "llm_judge": 0.6,  # Moderate trust
            "automatic": 0.3,  # Low trust
            "inferred": 0.1   # Very low trust
        }
    
    def record_user_feedback(self, query, result, feedback):
        """
        Record user feedback with high priority.
        Feedback can be: 'good', 'bad', or 'unsure'
        """
        self.user_labels.append({
            "query": query,
            "result": result,
            "label": feedback,
            "timestamp": time.now(),
            "trust_level": self.trust_scores["user"]
        })
        
        # Immediately update high-confidence examples
        if feedback in ['good', 'bad']:
            self.propagate_high_confidence_label(query, result, feedback)
    
    def prevent_drift(self, llm_evaluations):
        """
        Check if LLM evaluations are drifting from user labels.
        If drift detected, recalibrate the system.
        """
        disagreements = []
        
        for llm_eval in llm_evaluations:
            # Find corresponding user label if exists
            user_label = self.find_user_label(llm_eval["query"])
            
            if user_label and user_label != llm_eval["judgment"]:
                disagreements.append({
                    "query": llm_eval["query"],
                    "user_says": user_label,
                    "llm_says": llm_eval["judgment"]
                })
        
        drift_rate = len(disagreements) / len(llm_evaluations)
        
        if drift_rate > 0.2:  # 20% disagreement threshold
            print(f"⚠️ Drift detected: {drift_rate:.1%} disagreement with user labels")
            self.recalibrate_system(disagreements)
    
    def recalibrate_system(self, disagreements):
        """
        When drift is detected, trust user labels over LLM judgments.
        """
        for disagreement in disagreements:
            # Override LLM judgment with user label
            self.force_update_rag(
                query=disagreement["query"],
                label=disagreement["user_says"],
                source="user_override"
            )
```

#### Why User Feedback is Critical

1. **Prevents Hallucination Cascades**: LLMs can agree on plausible but wrong answers
2. **Domain Expertise**: Users know the correct logic for their specific domain
3. **Catches Edge Cases**: Users spot subtle incorrectness that LLMs miss
4. **Provides Grounding**: Acts as anchor points that prevent the system from drifting

#### Hybrid Evaluation Strategy

```python
def hybrid_evaluation(query, result, context):
    """
    Combine multiple evaluation sources with appropriate weighting.
    User feedback always overrides when available.
    """
    
    # Check for user feedback first (highest priority)
    user_feedback = get_user_feedback_if_exists(query)
    if user_feedback:
        return {
            "score": 1.0 if user_feedback == "good" else 0.0,
            "source": "user",
            "confidence": 1.0
        }
    
    # Use LLM judges during dream cycle
    llm_score = llm_consensus_evaluation(query, result)
    
    # Automatic checks
    auto_score = automatic_evaluation(query, result)
    
    # Weight based on trust levels
    if llm_score and auto_score:
        combined = 0.7 * llm_score + 0.3 * auto_score
    else:
        combined = llm_score or auto_score or 0.5
    
    return {
        "score": combined,
        "source": "hybrid",
        "confidence": 0.6  # Lower confidence without user feedback
    }
```

#### Feedback Collection Strategy

- **Optional but Encouraged**: Don't force feedback on every query
- **Selective Sampling**: Occasionally ask for feedback on important queries
- **Quick Interface**: Make it as easy as possible (single keystroke)
- **Batch Review**: Allow users to review and label multiple results at once

### 9. Active Learning Through Interactive Questioning

The system can proactively seek ground truth when it identifies areas of high uncertainty:

```python
class ActiveLearningMode:
    """
    Interactive mode where the system asks targeted questions to expand ground truth.
    Like a student asking a teacher for clarification.
    """
    
    def __init__(self, intrusiveness_level="low"):
        self.intrusiveness_level = intrusiveness_level
        self.questions_per_session = {
            "low": 1,      # Ask rarely
            "medium": 3,   # Ask occasionally  
            "high": 5,     # Educational mode
            "off": 0       # Never ask
        }
        self.uncertainty_threshold = 0.4  # When to consider asking
    
    def identify_learning_opportunities(self, recent_queries):
        """
        Find queries where user feedback would be most valuable.
        """
        opportunities = []
        
        for query in recent_queries:
            # High uncertainty cases
            if query.llm_confidence < self.uncertainty_threshold:
                opportunities.append({
                    "query": query,
                    "reason": "high_uncertainty",
                    "value": 1.0 - query.llm_confidence
                })
            
            # Disagreement between evaluators
            if query.evaluator_disagreement > 0.3:
                opportunities.append({
                    "query": query,
                    "reason": "evaluator_disagreement",
                    "value": query.evaluator_disagreement
                })
            
            # Novel pattern not in RAG
            if query.similarity_to_examples < 0.2:
                opportunities.append({
                    "query": query,
                    "reason": "novel_pattern",
                    "value": 1.0 - query.similarity_to_examples
                })
        
        # Sort by learning value
        opportunities.sort(key=lambda x: x["value"], reverse=True)
        return opportunities
    
    def generate_clarification_question(self, opportunity):
        """
        Create a natural question to ask the user.
        """
        query = opportunity["query"]
        reason = opportunity["reason"]
        
        if reason == "high_uncertainty":
            return f"""
I generated this rule but I'm not confident about it:
  Query: {query.original}
  Generated: {query.generated_knowledge}
  
Is this correct? (y)es/(n)o/(s)kip/(e)xplain: """
        
        elif reason == "evaluator_disagreement":
            return f"""
I'm getting mixed signals about this inference:
  Query: {query.original}
  Result: {query.result}
  
Does this look right to you? (y)es/(n)o/(s)kip: """
        
        elif reason == "novel_pattern":
            return f"""
This is a new type of query I haven't seen before:
  Query: {query.original}
  My attempt: {query.generated_knowledge}
  
Could you verify if this is the right approach? (y)es/(n)o/(e)xplain: """
    
    def interactive_learning_session(self, user_busy=False):
        """
        Conduct an interactive learning session if appropriate.
        """
        if user_busy or self.intrusiveness_level == "off":
            return
        
        max_questions = self.questions_per_session[self.intrusiveness_level]
        opportunities = self.identify_learning_opportunities(recent_queries)
        
        questions_asked = 0
        for opp in opportunities[:max_questions]:
            # Check if good time to ask
            if self.is_good_time_to_ask():
                question = self.generate_clarification_question(opp)
                response = prompt_user(question)
                
                if response == 'y' or response == 'yes':
                    self.record_ground_truth(opp["query"], "good")
                    print("✓ Thanks! I'll remember this pattern.")
                    
                elif response == 'n' or response == 'no':
                    self.record_ground_truth(opp["query"], "bad")
                    correct = input("Could you show me the correct answer? ")
                    if correct:
                        self.record_correction(opp["query"], correct)
                    
                elif response == 'e' or response == 'explain':
                    explanation = input("Please explain: ")
                    self.record_explanation(opp["query"], explanation)
                    
                questions_asked += 1
        
        if questions_asked > 0:
            print(f"Thanks for helping me learn! ({questions_asked} patterns clarified)")
```

#### Benefits of Active Learning Mode

1. **Targeted Learning**: Focuses on areas of maximum uncertainty
2. **Efficient Ground Truth Collection**: Gets labels where they matter most
3. **User Education**: Helps users understand what the system is learning
4. **Relationship Building**: Creates a teacher-student dynamic

#### When to Ask Questions

```python
def is_good_time_to_ask():
    """
    Heuristics for when to engage in active learning.
    """
    # After successful query completion
    if just_completed_query and query_was_successful:
        return True
    
    # During idle time
    if user_idle_for_seconds > 30:
        return True
    
    # Start of session (learning check-in)
    if session_just_started and not user_rushed:
        return True
    
    # Never during batch processing
    if in_batch_mode:
        return False
    
    return False
```

#### Intrusiveness Levels

- **Off**: Never ask questions
- **Low**: Ask 1 question per session, only for critical uncertainties
- **Medium**: Ask up to 3 questions, balanced approach
- **High**: Educational mode - actively engage in learning dialogue
- **Adaptive**: Adjust based on user engagement and response rate

This creates a system that can:
- Learn actively when the user is willing to teach
- Respect user time and attention
- Build high-quality ground truth incrementally
- Identify and fill knowledge gaps proactively

As you noted, it's like formal education - sometimes a bit intrusive, but ultimately valuable for building robust knowledge!

## Experimental Directions

### 1. Example Augmentation
- Generate variations of successful examples
- Use paraphrasing or rule transformations
- Expand the example database automatically

### 2. Negative Example Mining
- Track failed generations
- Include "don't do this" examples in prompts
- Learn from mistakes

### 3. Clustering for Diversity
- Cluster similar examples
- Ensure prompt examples come from different clusters
- Increase coverage of the problem space

### 4. Adaptive Temperature
- Start with high temperature (exploration)
- Decrease as confidence grows (exploitation)
- Balance based on success rates

### 5. Multi-Stage Retrieval
- First retrieval: Broad search (top 50)
- Re-ranking: More sophisticated scoring
- Final selection: From re-ranked results

### 6. Hybrid Retrieval Methods
- Combine embedding similarity with keyword matching
- Use BM25 for lexical similarity
- Weight multiple signals

## Implementation Architecture

```
┌─────────────────────────────────────────────────┐
│                 DreamLog Query                   │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Query Analyzer                      │
│  - Extract predicate type                        │
│  - Identify domain                               │
│  - Determine complexity                          │
└────────────────────┬─────────────────────────────┘
                     │
        ┌────────────┴────────────┬─────────────┐
        ▼                         ▼             ▼
┌──────────────┐        ┌──────────────┐  ┌──────────────┐
│ Example RAG  │        │ Template RAG  │  │   KB RAG     │
│              │        │               │  │              │
│ Retrieve     │        │ Retrieve      │  │ Retrieve     │
│ similar      │        │ successful    │  │ relevant     │
│ examples     │        │ templates     │  │ knowledge    │
└──────┬───────┘        └───────┬───────┘  └──────┬───────┘
       │                        │                  │
       └────────────┬───────────┴──────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│              Prompt Construction                 │
│  - Combine examples, templates, context          │
│  - Apply meta-learning insights                  │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│                  LLM Call                        │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│             Response Processing                  │
│  - Parse generated knowledge                     │
│  - Add to KB                                     │
│  - Track success/failure                         │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│            Feedback & Learning                   │
│  - Update RAG item statistics                    │
│  - Record meta-learning patterns                 │
│  - Add successful examples back to RAG           │
└──────────────────────────────────────────────────┘
```

## Configuration Example

```yaml
# dreamlog_config.yaml
rag:
  # Embedding provider configuration
  embedding:
    provider: ollama  # or openai, ngram
    base_url: http://192.168.0.225:11434
    model: nomic-embed-text
    cache_size: 1000
  
  # Example RAG configuration
  examples:
    db_path: ~/.dreamlog/example_rag.json
    retrieval_k: 8  # Number of examples to retrieve
    temperature: 1.0  # Sampling temperature
    use_confidence: true  # Weight by historical success
    
  # Template RAG configuration  
  templates:
    db_path: ~/.dreamlog/template_rag.json
    retrieval_k: 3
    temperature: 0.7
    
  # Meta-learning configuration
  meta_learning:
    tracking_path: ~/.dreamlog/meta_patterns.json
    min_uses_for_confidence: 3
    prune_failure_threshold: 0.8
    
  # Experimental flags
  experiments:
    use_context_embeddings: true  # Include KB context
    use_negative_examples: false  # Include failure examples
    use_clustering: false  # Cluster for diversity
    adaptive_temperature: true  # Adjust temperature over time
```

## Success Evaluation Mechanisms

The system uses multiple mechanisms to determine whether a query→answer was successful:

### 1. Automatic Success Detection
Basic heuristics that can run without external input:

```python
def evaluate_success_automatically(query, result, kb_before, kb_after):
    # Basic success criteria:
    # - Query completed without error
    # - No infinite loops detected
    # - Results were produced
    # - No contradictions introduced
    if result and not result.error:
        if not has_contradictions(kb_after):
            if not is_infinite_loop(query_trace):
                return True
    return False
```

### 2. User Feedback System
Allow users to rate results as good/bad/skip:

```python
class InteractiveQuerySession:
    def query(self, query_str: str, collect_feedback: bool = True):
        # Execute query with tracking
        used_examples = self.rag_system.retrieve(query_str)
        results = self.engine.query(query_str)
        
        if collect_feedback:
            rating = input("Was this answer correct? (g)ood/(b)ad/(s)kip: ")
            
            if rating in ['g', 'good']:
                # Reinforce successful patterns
                for example in used_examples:
                    self.rag_system.update_item_stats(example.id, success=True)
                # Add as new positive example
                self.rag_system.add_positive_example(query_str, results)
                
            elif rating in ['b', 'bad']:
                # Mark examples as failed
                for example in used_examples:
                    self.rag_system.update_item_stats(example.id, success=False)
                # Generate negative examples
                self.generate_negative_examples(query_str, results)
```

### 3. LLM-as-Judge Evaluation
Use LLMs to evaluate the quality of generated knowledge:

```python
class LLMEvaluator:
    def evaluate_result(self, query, generated_knowledge, result):
        prompt = f"""
        Query: {query}
        Generated Knowledge: {generated_knowledge}
        Result: {result}
        
        Is this a correct and reasonable answer?
        Respond with: GOOD, BAD, or UNSURE
        
        Consider:
        - Logical correctness
        - Consistency with common sense
        - Whether the rule makes semantic sense
        """
        
        response = llm.evaluate(prompt)
        return response  # GOOD, BAD, or UNSURE
    
    def consensus_evaluation(self, query, knowledge, result, num_judges=3):
        """Use multiple LLM calls for majority voting"""
        votes = []
        for _ in range(num_judges):
            vote = self.evaluate_result(query, knowledge, result)
            votes.append(vote)
        
        # Majority voting with UNSURE handling
        good_votes = votes.count("GOOD")
        bad_votes = votes.count("BAD")
        unsure_votes = votes.count("UNSURE")
        
        if good_votes > bad_votes and good_votes > unsure_votes:
            return "GOOD"
        elif bad_votes > good_votes and bad_votes > unsure_votes:
            return "BAD"
        else:
            return "UNSURE"  # No clear consensus
```

### 4. Test-Based Validation
When test cases are available:

```python
def validate_against_test_suite(generated_knowledge, test_suite):
    passed = 0
    failed = 0
    
    for test in test_suite:
        kb = KnowledgeBase()
        kb.add_facts(test["given_facts"])
        kb.add_knowledge(generated_knowledge)
        
        result = kb.query(test["query"])
        if result == test["expected"]:
            passed += 1
        else:
            failed += 1
    
    success_rate = passed / (passed + failed)
    return success_rate > 0.8  # 80% threshold
```

### 5. Coherence and Sanity Checks
Structural validation of generated knowledge:

```python
def validate_coherence(rule):
    checks = {
        "variables_connected": check_variable_connectivity(rule),
        "recursion_has_base": check_recursion_base_case(rule),
        "predicates_exist": check_predicate_validity(rule),
        "no_trivial_loops": check_for_trivial_loops(rule),
        "reasonable_arity": check_predicate_arity(rule)
    }
    return all(checks.values())
```

### 6. Cross-Validation with Multiple Providers
Use different LLMs to validate each other:

```python
def cross_validate_with_providers(query, providers):
    responses = []
    for provider in providers:
        knowledge = provider.generate_knowledge(query)
        responses.append(knowledge)
    
    # Check if majority agree on structure
    if has_consensus(responses, threshold=0.7):
        return True, majority_response(responses)
    return False, None
```

## Automatic Negative Example Generation

The system learns from failures by generating negative examples:

```python
class NegativeExampleGenerator:
    def generate_from_failure(self, query, bad_result, failure_reason=None):
        # Create primary negative example
        negative = {
            "query": query,
            "bad_output": bad_result,
            "reason": failure_reason or "Marked as incorrect",
            "type": "negative"
        }
        
        # Generate variations to prevent similar mistakes
        variations = []
        
        # Variation 1: Similar predicates
        similar_predicates = find_similar_predicates(query)
        for pred in similar_predicates:
            variations.append({
                "query": replace_predicate(query, pred),
                "avoid_pattern": bad_result,
                "type": "negative_variation"
            })
        
        # Variation 2: Different variable arrangements
        var_permutations = generate_variable_permutations(query)
        for perm in var_permutations:
            variations.append({
                "query": perm,
                "avoid_pattern": bad_result,
                "type": "negative_variation"
            })
        
        return [negative] + variations
    
    def include_in_prompt(self, negative_examples):
        """Format negative examples for inclusion in prompts"""
        return "\n".join([
            f"AVOID: {ex['query']} -> {ex['bad_output']} (Reason: {ex['reason']})"
            for ex in negative_examples[:3]  # Include top 3 most relevant
        ])
```

## Success Evaluation Pipeline

The complete pipeline combines multiple evaluation methods:

```python
class SuccessEvaluationPipeline:
    def __init__(self, config):
        self.use_automatic = config.get("use_automatic", True)
        self.use_llm_judge = config.get("use_llm_judge", True)
        self.use_user_feedback = config.get("use_user_feedback", False)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
    
    def evaluate(self, query, result, context):
        confidence_scores = {}
        
        # Level 1: Automatic checks (fast, always run)
        if self.use_automatic:
            auto_success = self.automatic_checks(query, result)
            confidence_scores["automatic"] = 1.0 if auto_success else 0.0
        
        # Level 2: LLM judgment (slower, optional)
        if self.use_llm_judge and confidence_scores.get("automatic", 0) > 0:
            llm_result = self.llm_consensus_judge(query, result)
            if llm_result == "GOOD":
                confidence_scores["llm"] = 1.0
            elif llm_result == "BAD":
                confidence_scores["llm"] = 0.0
            else:  # UNSURE
                confidence_scores["llm"] = 0.5
        
        # Level 3: User feedback (when available)
        if self.use_user_feedback:
            user_rating = self.request_user_feedback(query, result)
            if user_rating:
                confidence_scores["user"] = 1.0 if user_rating == "good" else 0.0
        
        # Combine scores with weights
        weights = {"automatic": 0.2, "llm": 0.5, "user": 1.0}
        total_weight = sum(weights[k] for k in confidence_scores.keys())
        weighted_score = sum(
            weights[k] * v for k, v in confidence_scores.items()
        ) / total_weight
        
        return weighted_score > self.confidence_threshold
```

## Configuration for Success Evaluation

```yaml
success_evaluation:
  # Automatic evaluation
  automatic:
    enabled: true
    check_contradictions: true
    check_infinite_loops: true
    timeout_seconds: 10
  
  # LLM-as-judge
  llm_judge:
    enabled: true
    providers: [ollama, openai]  # Multiple for consensus
    num_judges: 3  # Odd number for majority voting
    include_unsure: true  # Allow UNSURE responses
    
  # User feedback
  user_feedback:
    enabled: false  # Enable in interactive mode
    optional: true  # Don't force feedback
    save_feedback: true  # Store for future training
    
  # Negative examples
  negative_examples:
    generate_variations: true
    max_variations_per_failure: 5
    include_in_prompts: true
    max_in_prompt: 3
    
  # Confidence thresholds
  confidence:
    minimum_for_success: 0.7
    minimum_for_retention: 0.3  # Below this, prune
    high_confidence: 0.9  # Mark as "gold standard"
```

## Metrics and Evaluation

### Success Metrics
- **Query Resolution Rate**: Percentage of queries successfully answered
- **Knowledge Quality**: Do generated rules/facts lead to correct inferences?
- **Convergence Speed**: How quickly does the system improve?
- **Diversity**: Are we exploring different solution approaches?
- **Agreement Rate**: How often do different evaluation methods agree?

### Tracking Metrics
```python
{
    "total_queries": 1000,
    "successful_queries": 850,
    "avg_examples_per_query": 6.3,
    "avg_llm_calls_per_query": 2.1,  # Due to recursion
    "example_db_size": 523,
    "high_confidence_examples": 89,
    "pruned_examples": 34,
    "evaluation_methods": {
        "automatic_only": 450,
        "llm_judged": 380,
        "user_rated": 120,
        "consensus_achieved": 720
    },
    "negative_examples_generated": 156,
    "llm_judge_agreement_rate": 0.82
}
```

## Future Research Directions

### 1. Curriculum Learning
- Start with simple examples, gradually increase complexity
- Build foundational knowledge before complex rules

### 2. Adversarial Examples
- Generate challenging cases to improve robustness
- Test edge cases and unusual combinations

### 3. Transfer Learning
- Use knowledge from one domain to bootstrap another
- Identify universal patterns across domains

### 4. Ensemble Methods
- Use multiple LLMs with different characteristics
- Combine outputs for better reliability

### 5. Active Learning
- Identify areas where the system is uncertain
- Actively seek examples for those areas

## Conclusion

The RAG-based learning system transforms DreamLog from a static logic programming language into a dynamic, self-improving system. By combining retrieval, generation, and meta-learning, we create a system that:

1. **Learns from experience** - Successful patterns are reinforced
2. **Adapts to domains** - Different strategies for different problem types
3. **Self-bootstraps** - Builds up knowledge recursively as needed
4. **Improves over time** - Meta-learning identifies what works when

This architecture provides a foundation for exploring how logic programming and machine learning can be deeply integrated, creating systems that combine the rigor of formal logic with the adaptability of neural approaches.