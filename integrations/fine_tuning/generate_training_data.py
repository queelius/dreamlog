"""
LLM Fine-tuning Data Generator for DreamLog

Generates training data in various formats for fine-tuning LLMs to work with DreamLog.
Supports multiple output formats:
- OpenAI fine-tuning format (JSONL)
- HuggingFace datasets format
- Anthropic Constitutional AI format
- Generic instruction tuning format

Usage:
    python generate_training_data.py --kb knowledge.json --output dataset.jsonl --format openai
"""

import json
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dreamlog import (
    DreamLogEngine, parse_prefix_notation, parse_s_expression,
    atom, var, compound, Fact, Rule
)
from dreamlog.prefix_parser import term_to_sexp, term_to_prefix_json


@dataclass
class TrainingExample:
    """Represents a single training example"""
    instruction: str
    input: str
    output: str
    metadata: Optional[Dict[str, Any]] = None


class TrainingDataGenerator:
    """
    Generates diverse training data for fine-tuning LLMs on DreamLog tasks.
    """
    
    def __init__(self, engine: DreamLogEngine, include_explanations: bool = True):
        self.engine = engine
        self.include_explanations = include_explanations
    
    def generate_query_examples(self, num_examples: int = 100) -> List[TrainingExample]:
        """Generate examples of querying the knowledge base"""
        examples = []
        facts = list(self.engine.kb.facts)
        rules = list(self.engine.kb.rules)
        
        for _ in range(min(num_examples, len(facts) * 10)):
            # Choose a random fact or rule head
            if random.random() < 0.7 and facts:
                fact = random.choice(facts)
                base_term = fact.term
            elif rules:
                rule = random.choice(rules)
                base_term = rule.head
            else:
                continue
            
            # Create variations of queries
            query_type = random.choice(["ground", "partial", "variable"])
            
            if query_type == "ground" and hasattr(base_term, 'functor'):
                # Ground query (all constants)
                query_term = base_term
                instruction = "Query the DreamLog knowledge base with the following query"
            elif query_type == "partial" and hasattr(base_term, 'args') and len(base_term.args) > 1:
                # Partial query (mix of constants and variables)
                import copy
                query_term = copy.deepcopy(base_term)
                num_vars = random.randint(1, len(query_term.args))
                var_positions = random.sample(range(len(query_term.args)), num_vars)
                for i, pos in enumerate(var_positions):
                    query_term.args = list(query_term.args)
                    query_term.args[pos] = var(f"X{i}")
                    query_term.args = tuple(query_term.args)
                instruction = "Find all solutions for the following query with variables"
            else:
                # All variables
                if hasattr(base_term, 'functor') and hasattr(base_term, 'args'):
                    args = [var(f"X{i}") for i in range(len(base_term.args))]
                    query_term = compound(base_term.functor, *args)
                    instruction = "Find all possible values for the variables in this query"
                else:
                    continue
            
            # Execute query
            solutions = list(self.engine.query([query_term]))[:5]  # Limit solutions
            
            # Format input and output
            input_text = f"Query: {term_to_sexp(query_term)}"
            
            if solutions:
                output_lines = []
                for i, sol in enumerate(solutions, 1):
                    if sol.ground_bindings:
                        bindings_str = ", ".join(f"{k}={v}" for k, v in sol.ground_bindings.items())
                        output_lines.append(f"Solution {i}: {bindings_str}")
                    else:
                        output_lines.append(f"Solution {i}: Yes")
                output_text = "\n".join(output_lines)
            else:
                output_text = "No solutions found."
            
            if self.include_explanations:
                output_text += f"\n\nExplanation: The query {term_to_sexp(query_term)} "
                if solutions:
                    output_text += f"matches {len(solutions)} fact(s)/rule(s) in the knowledge base."
                else:
                    output_text += "does not match any facts or rules in the knowledge base."
            
            examples.append(TrainingExample(
                instruction=instruction,
                input=input_text,
                output=output_text,
                metadata={"type": "query", "num_solutions": len(solutions)}
            ))
        
        return examples
    
    def generate_fact_addition_examples(self, num_examples: int = 50) -> List[TrainingExample]:
        """Generate examples of adding facts to the knowledge base"""
        examples = []
        
        # Common predicates for generating examples
        predicates = ["parent", "likes", "owns", "works_at", "studies", "teaches", "friend"]
        entities = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry"]
        
        for _ in range(num_examples):
            pred = random.choice(predicates)
            arity = random.randint(1, 3)
            args = random.sample(entities, arity)
            
            fact_term = compound(pred, *[atom(a) for a in args])
            
            instruction = "Add the following fact to the DreamLog knowledge base"
            input_text = f"Fact: {term_to_sexp(fact_term)}"
            output_text = f"Added fact: {term_to_sexp(fact_term)} to the knowledge base."
            
            if self.include_explanations:
                output_text += f"\n\nThis establishes that {pred}({', '.join(args)}) is true."
            
            examples.append(TrainingExample(
                instruction=instruction,
                input=input_text,
                output=output_text,
                metadata={"type": "add_fact", "predicate": pred}
            ))
        
        return examples
    
    def generate_rule_creation_examples(self, num_examples: int = 50) -> List[TrainingExample]:
        """Generate examples of creating rules"""
        examples = []
        
        rule_templates = [
            ("sibling", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"], ["different", "X", "Y"]]),
            ("grandparent", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]),
            ("ancestor", ["ancestor", "X", "Y"], [["parent", "X", "Y"]]),
            ("ancestor_recursive", ["ancestor", "X", "Z"], [["parent", "X", "Y"], ["ancestor", "Y", "Z"]]),
            ("friend_mutual", ["friends", "X", "Y"], [["likes", "X", "Y"], ["likes", "Y", "X"]]),
            ("colleague", ["colleagues", "X", "Y"], [["works_at", "X", "Z"], ["works_at", "Y", "Z"], ["different", "X", "Y"]]),
        ]
        
        for _ in range(min(num_examples, len(rule_templates) * 5)):
            name, head_pattern, body_patterns = random.choice(rule_templates)
            
            # Convert patterns to terms
            head = parse_prefix_notation(head_pattern)
            body = [parse_prefix_notation(p) for p in body_patterns]
            
            instruction = f"Create a rule for {name} relationships in DreamLog"
            input_text = f"Define a rule for: {name}"
            
            output_text = f"Rule: {term_to_sexp(head)} :- "
            output_text += ", ".join(term_to_sexp(b) for b in body)
            
            if self.include_explanations:
                output_text += f"\n\nThis rule states that {term_to_sexp(head)} is true when "
                output_text += " AND ".join(term_to_sexp(b) for b in body) + " are all true."
            
            examples.append(TrainingExample(
                instruction=instruction,
                input=input_text,
                output=output_text,
                metadata={"type": "create_rule", "rule_name": name}
            ))
        
        return examples
    
    def generate_explanation_examples(self, num_examples: int = 30) -> List[TrainingExample]:
        """Generate examples explaining DreamLog concepts"""
        examples = []
        
        concepts = [
            ("unification", "Explain how unification works in DreamLog", 
             "Unification in DreamLog is the process of making two terms identical by finding substitutions for variables. "
             "For example, parent(X, mary) unifies with parent(john, mary) by substituting X=john. "
             "Unification is the core mechanism for pattern matching and query resolution."),
            
            ("variables", "Explain variables in DreamLog",
             "Variables in DreamLog start with uppercase letters (X, Y, Person). They act as placeholders that can be "
             "substituted with any term during unification. Variables are scoped to a single rule or query."),
            
            ("facts_vs_rules", "What's the difference between facts and rules in DreamLog?",
             "Facts are assertions that are always true, like parent(john, mary). "
             "Rules define relationships using implications, like grandparent(X, Z) :- parent(X, Y), parent(Y, Z). "
             "Facts have no conditions, while rules have a head and body with conditions."),
            
            ("backtracking", "How does backtracking work in DreamLog?",
             "Backtracking occurs when DreamLog explores alternative solutions. If a query path fails, "
             "the system backtracks to the last choice point and tries a different path. "
             "This ensures all possible solutions are found."),
            
            ("prefix_notation", "Explain DreamLog's S-expression notation",
             "DreamLog uses S-expressions (prefix notation) like (parent john mary) instead of parent(john, mary). "
             "This makes parsing simpler and more consistent. Lists are [\"parent\", \"john\", \"mary\"] in JSON."),
        ]
        
        for _ in range(min(num_examples, len(concepts) * 3)):
            concept, question, answer = random.choice(concepts)
            
            examples.append(TrainingExample(
                instruction="Explain the following DreamLog concept",
                input=question,
                output=answer,
                metadata={"type": "explanation", "concept": concept}
            ))
        
        return examples
    
    def generate_conversion_examples(self, num_examples: int = 30) -> List[TrainingExample]:
        """Generate examples of converting between formats"""
        examples = []
        
        for _ in range(num_examples):
            # Generate a random term
            pred = random.choice(["parent", "likes", "knows", "teaches"])
            args = random.choice([
                ["john", "mary"],
                ["X", "alice"],
                ["bob", "Y"],
                ["X", "Y"]
            ])
            
            # Randomly choose conversion direction
            if random.random() < 0.5:
                # S-expression to JSON
                sexp = f"({pred} {' '.join(args)})"
                json_repr = json.dumps([pred] + args)
                
                instruction = "Convert this S-expression to JSON prefix notation"
                input_text = f"S-expression: {sexp}"
                output_text = f"JSON: {json_repr}"
            else:
                # JSON to S-expression
                json_repr = json.dumps([pred] + args)
                sexp = f"({pred} {' '.join(args)})"
                
                instruction = "Convert this JSON prefix notation to S-expression"
                input_text = f"JSON: {json_repr}"
                output_text = f"S-expression: {sexp}"
            
            examples.append(TrainingExample(
                instruction=instruction,
                input=input_text,
                output=output_text,
                metadata={"type": "conversion"}
            ))
        
        return examples
    
    def generate_all_examples(self) -> List[TrainingExample]:
        """Generate a comprehensive training dataset"""
        all_examples = []
        
        # Generate different types of examples
        all_examples.extend(self.generate_query_examples(100))
        all_examples.extend(self.generate_fact_addition_examples(50))
        all_examples.extend(self.generate_rule_creation_examples(50))
        all_examples.extend(self.generate_explanation_examples(30))
        all_examples.extend(self.generate_conversion_examples(30))
        
        # Shuffle for variety
        random.shuffle(all_examples)
        
        return all_examples


def format_for_openai(examples: List[TrainingExample]) -> List[Dict[str, Any]]:
    """Format examples for OpenAI fine-tuning (JSONL format)"""
    formatted = []
    
    for ex in examples:
        formatted.append({
            "messages": [
                {"role": "system", "content": "You are an expert DreamLog logic programming assistant."},
                {"role": "user", "content": f"{ex.instruction}\n\n{ex.input}"},
                {"role": "assistant", "content": ex.output}
            ]
        })
    
    return formatted


def format_for_anthropic(examples: List[TrainingExample]) -> List[Dict[str, Any]]:
    """Format examples for Anthropic-style training"""
    formatted = []
    
    for ex in examples:
        formatted.append({
            "prompt": f"Human: {ex.instruction}\n\n{ex.input}\n\nAssistant:",
            "completion": f" {ex.output}",
            "metadata": ex.metadata
        })
    
    return formatted


def format_for_huggingface(examples: List[TrainingExample]) -> Dict[str, List]:
    """Format examples for HuggingFace datasets"""
    dataset = {
        "instruction": [],
        "input": [],
        "output": [],
        "text": []  # Full formatted text
    }
    
    for ex in examples:
        dataset["instruction"].append(ex.instruction)
        dataset["input"].append(ex.input)
        dataset["output"].append(ex.output)
        
        # Alpaca-style format
        text = f"### Instruction:\n{ex.instruction}\n\n"
        if ex.input:
            text += f"### Input:\n{ex.input}\n\n"
        text += f"### Response:\n{ex.output}"
        dataset["text"].append(text)
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate DreamLog training data for LLM fine-tuning")
    parser.add_argument("--kb", help="Path to knowledge base file")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["openai", "anthropic", "huggingface", "jsonl"],
                       default="jsonl", help="Output format")
    parser.add_argument("--num-examples", type=int, default=200, help="Total number of examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-explanations", action="store_true", help="Omit explanations")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Create engine and load KB if provided
    engine = DreamLogEngine()
    if args.kb and os.path.exists(args.kb):
        with open(args.kb, 'r') as f:
            engine.load_from_prefix(f.read())
    else:
        # Load default family example
        engine.load_from_prefix('''[
            ["fact", ["parent", "john", "mary"]],
            ["fact", ["parent", "mary", "alice"]],
            ["fact", ["parent", "tom", "bob"]],
            ["fact", ["parent", "bob", "charlie"]],
            ["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]
        ]''')
    
    # Generate training data
    generator = TrainingDataGenerator(engine, include_explanations=not args.no_explanations)
    examples = generator.generate_all_examples()
    
    # Limit to requested number
    examples = examples[:args.num_examples]
    
    # Format and save
    output_path = Path(args.output)
    
    if args.format == "openai" or args.format == "jsonl":
        formatted = format_for_openai(examples)
        with open(output_path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(formatted)} examples in OpenAI JSONL format to {output_path}")
    
    elif args.format == "anthropic":
        formatted = format_for_anthropic(examples)
        with open(output_path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(formatted)} examples in Anthropic format to {output_path}")
    
    elif args.format == "huggingface":
        dataset = format_for_huggingface(examples)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset['instruction'])} examples in HuggingFace format to {output_path}")
    
    # Print statistics
    print("\nDataset statistics:")
    type_counts = {}
    for ex in examples:
        if ex.metadata:
            t = ex.metadata.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
    
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count} examples")


if __name__ == "__main__":
    main()