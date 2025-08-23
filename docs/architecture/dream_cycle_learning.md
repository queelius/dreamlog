# Dream Cycle Learning System

## Overview

The dream cycle in DreamLog is inspired by the wake-sleep algorithm and serves as the system's learning and reorganization phase. During "sleep", the system has time to evaluate past performance, learn from successes and failures, and optimize its knowledge structures without the pressure of real-time query answering.

## Wake-Sleep Architecture

### Wake Phase (Query Time)
During the wake phase, the system:
- Answers queries using existing knowledge and RAG examples
- Generates new knowledge via LLMs when needed
- Logs all activities but doesn't evaluate deeply
- Focuses on speed and getting answers

```python
class WakePhase:
    def query(self, query_str):
        # Fast retrieval and generation
        examples = self.rag.retrieve_fast(query_str)
        result = self.engine.query_with_llm(query_str, examples)
        
        # Log for later evaluation
        self.activity_log.append({
            "query": query_str,
            "examples_used": examples,
            "knowledge_generated": result.new_knowledge,
            "result": result.answer,
            "timestamp": time.now()
        })
        
        return result.answer  # Return quickly
```

### Sleep Phase (Dream Cycle)
During the sleep phase, the system takes time to:
- Evaluate past queries using multiple methods
- Update RAG databases with confidence scores
- Generate negative examples from failures
- Reorganize and compress knowledge
- Learn meta-patterns

```python
class SleepPhase:
    def dream(self, activity_log, duration_minutes=10):
        """
        Run dream cycle for specified duration.
        Can be interrupted but will resume from checkpoint.
        """
        start_time = time.now()
        
        while time.now() - start_time < duration_minutes * 60:
            # Phase 1: Evaluate past queries
            self.evaluate_historical_queries(activity_log)
            
            # Phase 2: Update confidence scores
            self.update_rag_confidence()
            
            # Phase 3: Generate negative examples
            self.mine_negative_examples()
            
            # Phase 4: Compress and reorganize
            self.compress_knowledge()
            
            # Phase 5: Meta-learning analysis
            self.analyze_meta_patterns()
            
            # Checkpoint progress
            self.save_checkpoint()
```

## Dream Cycle Evaluation Process

### 1. Historical Query Evaluation

During sleep, the system can afford expensive evaluation methods:

```python
class DreamEvaluator:
    def evaluate_historical_queries(self, activity_log):
        for entry in activity_log:
            if entry.get("evaluated"):
                continue  # Skip already evaluated
            
            # Multi-method evaluation
            evaluations = {}
            
            # LLM-as-judge with multiple attempts
            llm_scores = []
            for _ in range(5):  # Multiple samples for confidence
                score = self.llm_judge.evaluate(
                    entry["query"],
                    entry["knowledge_generated"],
                    entry["result"]
                )
                llm_scores.append(score)
            
            # Consensus from different LLM providers
            provider_scores = []
            for provider in self.llm_providers:
                score = provider.judge(entry)
                provider_scores.append(score)
            
            # Coherence analysis (expensive checks)
            coherence = self.deep_coherence_check(entry)
            
            # Combine evaluations
            final_score = self.combine_evaluations(
                llm_scores, provider_scores, coherence
            )
            
            # Update entry with evaluation
            entry["evaluated"] = True
            entry["success_score"] = final_score
            entry["evaluation_details"] = evaluations
```

### 2. RAG Database Updates

Update example and template databases based on evaluations:

```python
def update_rag_confidence(self):
    for entry in self.evaluated_entries:
        success = entry["success_score"] > 0.7
        
        # Update examples that were used
        for example_id in entry["examples_used"]:
            self.example_rag.update_stats(example_id, success)
        
        # Update templates that were used
        for template_id in entry["templates_used"]:
            self.template_rag.update_stats(template_id, success)
        
        # Add successful patterns as new examples
        if success and entry["success_score"] > 0.9:
            self.example_rag.add_item(
                content={
                    "query": entry["query"],
                    "output": entry["knowledge_generated"]
                },
                text_for_embedding=entry["query"],
                metadata={"learned_from": "dream_cycle"},
                source="learned"
            )
```

### 3. Negative Example Mining

Generate and store patterns to avoid:

```python
def mine_negative_examples(self):
    for entry in self.evaluated_entries:
        if entry["success_score"] < 0.3:  # Failed query
            # Generate negative example
            negative = {
                "query": entry["query"],
                "bad_output": entry["knowledge_generated"],
                "failure_reason": self.analyze_failure(entry)
            }
            
            # Generate variations
            variations = self.generate_failure_variations(negative)
            
            # Store in negative example database
            for var in [negative] + variations:
                self.negative_rag.add_item(
                    content=var,
                    text_for_embedding=var["query"],
                    metadata={"failure_type": var["failure_reason"]},
                    source="mined"
                )
```

### 4. Knowledge Compression and Reorganization

Use the dream time to optimize knowledge structures:

```python
def compress_knowledge(self):
    # Identify redundant rules
    redundant = self.find_redundant_rules()
    
    # Find more general patterns
    generalizations = self.find_generalizations()
    
    # Compress using LLM
    for pattern_group in redundant:
        compressed = self.llm.compress_rules(pattern_group)
        if self.validate_compression(compressed, pattern_group):
            self.kb.replace_rules(pattern_group, compressed)
    
    # Abstract common patterns
    for concept in self.identify_concepts():
        abstraction = self.llm.create_abstraction(concept)
        self.kb.add_abstraction(abstraction)
```

### 5. Meta-Learning Analysis

Identify what works when:

```python
def analyze_meta_patterns(self):
    # Group queries by type
    query_types = self.classify_queries(self.evaluated_entries)
    
    for qtype, entries in query_types.items():
        # Find successful patterns for this type
        successful = [e for e in entries if e["success_score"] > 0.8]
        
        # Extract common features
        common_features = self.extract_common_features(successful)
        
        # Create meta-rule
        meta_rule = {
            "query_type": qtype,
            "successful_example_features": common_features,
            "recommended_templates": self.find_best_templates(successful),
            "recommended_approach": self.determine_approach(successful)
        }
        
        self.meta_knowledge.add_rule(meta_rule)
```

## Dream Cycle Configuration

```yaml
dream_cycle:
  # Schedule
  schedule:
    mode: "periodic"  # periodic, idle, manual
    period_minutes: 60  # Run every hour
    idle_threshold: 300  # Or when idle for 5 minutes
    max_duration: 600  # Max 10 minutes per cycle
  
  # Evaluation settings
  evaluation:
    use_llm_judge: true
    llm_judge_samples: 5  # Multiple samples for confidence
    use_multiple_providers: true
    providers: ["ollama", "openai"]
    consensus_threshold: 0.7
    
  # Learning settings
  learning:
    min_success_score: 0.8  # To add as positive example
    max_failure_score: 0.3  # To add as negative example
    prune_threshold: 0.2  # Remove very bad examples
    
  # Compression settings
  compression:
    enable_rule_compression: true
    enable_abstraction: true
    min_rules_to_compress: 3
    validation_required: true
    
  # Meta-learning
  meta_learning:
    enable: true
    min_examples_per_type: 10
    confidence_threshold: 0.85
```

## Advantages of Dream-Time Evaluation

### 1. No Time Pressure
- Can use expensive evaluation methods
- Multiple LLM calls for consensus
- Deep coherence checking
- Cross-validation with multiple providers

### 2. Batch Processing
- Evaluate many queries together
- Find patterns across queries
- Identify systematic issues

### 3. Learning Without Interference
- Update RAG databases without affecting ongoing queries
- Reorganize knowledge structures
- Experiment with compressions

### 4. Continuous Improvement
- Each dream cycle makes the system better
- Learns from both successes and failures
- Adapts to the types of queries it receives

## Dream Cycle Workflow

```
┌─────────────────────────────────────────┐
│         Activity Log (Wake Phase)        │
│  - Queries, results, knowledge generated │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ╔════════════════╗
        ║  DREAM CYCLE   ║
        ╚════════════════╝
                 │
    ┌────────────┴────────────┬──────────────┬─────────────┐
    ▼                         ▼              ▼             ▼
┌──────────────┐    ┌──────────────┐  ┌──────────┐  ┌──────────┐
│  Evaluation  │    │   Learning   │  │Compression│ │   Meta    │
│              │    │              │  │           │ │ Learning  │
│ LLM Judge    │    │Update RAG    │  │Find       │ │ Patterns  │
│ Consensus    │    │Mine Negative │  │Redundancy │ │ Rules     │
│ Coherence    │    │Generate New  │  │Abstract   │ │ Insights  │
└──────┬───────┘    └──────┬───────┘  └─────┬─────┘ └─────┬─────┘
       │                   │                 │             │
       └───────────────────┴─────────────────┴─────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Updated Knowledge   │
                │  - Better examples   │
                │  - Refined rules     │
                │  - Meta-patterns     │
                └──────────────────────┘
```

## Metrics and Monitoring

### Dream Cycle Metrics
```python
{
    "cycles_completed": 42,
    "total_queries_evaluated": 3580,
    "avg_evaluation_time_ms": 125,
    "positive_examples_added": 234,
    "negative_examples_mined": 156,
    "rules_compressed": 45,
    "abstractions_created": 12,
    "meta_patterns_identified": 28,
    "knowledge_quality_improvement": 0.34,  # 34% improvement
    "avg_success_score_trend": [0.65, 0.72, 0.78, 0.81],  # Over time
}
```

## Integration with Wake Phase

The wake phase can use dream cycle insights:

```python
class WakePhaseWithDreamInsights:
    def query(self, query_str):
        # Check meta-knowledge for query type
        query_type = self.classify_query(query_str)
        meta_rule = self.meta_knowledge.get_rule(query_type)
        
        if meta_rule:
            # Use recommended approach
            examples = self.rag.retrieve_with_features(
                query_str,
                preferred_features=meta_rule["successful_example_features"]
            )
            template = meta_rule["recommended_templates"][0]
        else:
            # Default approach
            examples = self.rag.retrieve(query_str)
            template = self.default_template
        
        # Also check negative examples to avoid
        negatives = self.negative_rag.retrieve(query_str, k=3)
        
        # Generate with insights
        result = self.engine.query_with_insights(
            query_str, 
            examples,
            template,
            avoid_patterns=negatives
        )
        
        return result
```

## Conclusion

By moving evaluation and learning to the dream cycle, DreamLog can:
1. **Maintain fast query response** during wake phase
2. **Learn deeply** during sleep phase
3. **Continuously improve** without user intervention
4. **Adapt to usage patterns** automatically

This creates a self-improving system that gets better with use, learning from its experiences and reorganizing its knowledge for optimal performance.