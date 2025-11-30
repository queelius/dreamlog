"""
DreamLog LLM Response Parser and Validator

This module provides a proper abstraction for parsing and validating LLM responses
using DreamLog's native parser and evaluator instead of ad-hoc JSON parsing.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from .prefix_parser import (
    parse_prefix_notation, 
    parse_rule,
    parse_s_expression
)
from .terms import Term, Atom, Variable, Compound
from .knowledge import Fact, Rule
from .evaluator import PrologEvaluator
from .llm_providers import LLMResponse


@dataclass
class ParsedKnowledge:
    """Structured representation of parsed knowledge from LLM"""
    facts: List[Fact]
    rules: List[Rule]
    raw_data: Any
    errors: List[str]
    
    @property
    def is_valid(self) -> bool:
        """Check if parsing was successful"""
        return len(self.errors) == 0 and (len(self.facts) > 0 or len(self.rules) > 0)
    
    def to_llm_response(self, text: str = "") -> LLMResponse:
        """Convert to LLMResponse format"""
        # Convert facts and rules to the format LLMResponse expects
        fact_data = [fact.term.to_prefix() for fact in self.facts]
        rule_data = [[rule.head.to_prefix(), [t.to_prefix() for t in rule.body]] for rule in self.rules]
        
        return LLMResponse(
            text=text,
            facts=fact_data,
            rules=rule_data,
            raw_response=json.dumps(self.raw_data) if self.raw_data else text
        )


class DreamLogResponseParser:
    """
    Parser for LLM responses that uses DreamLog's native parsing capabilities
    """
    
    def __init__(self, strict: bool = False, verbose: bool = False):
        """
        Initialize parser
        
        Args:
            strict: If True, fail on any parsing error. If False, collect what we can.
            verbose: Enable debug output
        """
        self.strict = strict
        self.verbose = verbose
    
    def parse(self, response: str) -> ParsedKnowledge:
        """
        Parse an LLM response into DreamLog knowledge
        
        Args:
            response: Raw LLM response text
            
        Returns:
            ParsedKnowledge with facts, rules, and any errors
        """
        facts = []
        rules = []
        errors = []
        raw_data = None
        
        # Try multiple parsing strategies - Prolog first since it's most natural for LLMs
        strategies = [
            self._parse_as_prolog,
            self._parse_as_json,
            self._parse_as_sexp,
            self._parse_with_extraction,
            self._parse_as_mixed
        ]
        
        for strategy in strategies:
            if self.verbose:
                print(f"Trying strategy: {strategy.__name__}")
            
            result = strategy(response)
            if result:
                facts, rules, raw_data, strategy_errors = result
                errors.extend(strategy_errors)
                
                # If we got something and not in strict mode, accept it
                if (facts or rules) and not self.strict:
                    break
                # In strict mode, only accept if no errors
                elif not strategy_errors:
                    break
        
        return ParsedKnowledge(facts, rules, raw_data, errors)

    def _fix_unquoted_json(self, json_str: str) -> str:
        """
        Fix common LLM JSON formatting errors by quoting unquoted identifiers.

        Handles cases like:
        - [["rule", ["ancestor", X, Y], ...]]  ->  [["rule", ["ancestor", "X", "Y"], ...]]
        - [[parent, X, Y]]  ->  [["parent", "X", "Y"]]
        """
        import re

        # Pattern to match unquoted identifiers (alphanumeric + underscore, starting with letter)
        # This matches things like: X, Y, parent, ancestor, etc.
        # But NOT things already quoted or numbers
        pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\b(?=\s*[,\]\)])'

        def quote_if_needed(match):
            word = match.group(1)
            # Don't quote JSON keywords
            if word in ['true', 'false', 'null']:
                return word
            # Quote everything else
            return f'"{word}"'

        # Apply the fix
        result = re.sub(pattern, quote_if_needed, json_str)
        return result

    def _parse_as_prolog(self, response: str) -> Optional[Tuple[List[Fact], List[Rule], Any, List[str]]]:
        """
        Parse Prolog-style syntax from LLM response.

        Handles:
        - Facts: predicate(arg1, arg2).
        - Rules: head(X, Y) :- body1(X), body2(Y).

        Also extracts from ```prolog code blocks.
        """
        facts = []
        rules = []
        errors = []

        # Extract from ```prolog code blocks first
        prolog_block_pattern = r'```(?:prolog)?\n?(.*?)```'
        blocks = re.findall(prolog_block_pattern, response, re.DOTALL | re.IGNORECASE)

        # If no code blocks, try to find Prolog statements in the raw text
        if blocks:
            text_to_parse = '\n'.join(blocks)
        else:
            text_to_parse = response

        # Pattern for Prolog clauses (facts and rules)
        # Matches: predicate(args) :- body.  OR  predicate(args).
        clause_pattern = r'([a-z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(?::-\s*(.+?))?\s*\.'

        for match in re.finditer(clause_pattern, text_to_parse, re.MULTILINE):
            functor = match.group(1)
            args_str = match.group(2)
            body_str = match.group(3)

            try:
                # Parse arguments
                args = self._parse_prolog_args(args_str)
                head_term = self._build_term(functor, args)

                if body_str:
                    # It's a rule
                    body_terms = self._parse_prolog_body(body_str)
                    if body_terms:
                        rules.append(Rule(head_term, body_terms))
                else:
                    # It's a fact
                    facts.append(Fact(head_term))

            except Exception as e:
                errors.append(f"Error parsing Prolog clause '{match.group(0)}': {e}")

        if facts or rules:
            return (facts, rules, text_to_parse, errors)
        return None

    def _parse_prolog_args(self, args_str: str) -> List[Term]:
        """Parse comma-separated Prolog arguments into terms."""
        from .factories import atom, var

        args = []
        if not args_str.strip():
            return args

        # Simple split by comma (doesn't handle nested terms yet)
        for arg in args_str.split(','):
            arg = arg.strip()
            if not arg:
                continue
            if arg[0].isupper() or arg.startswith('_'):
                # Variable
                args.append(var(arg))
            else:
                # Atom (constant)
                args.append(atom(arg))
        return args

    def _parse_prolog_body(self, body_str: str) -> List[Term]:
        """Parse Prolog rule body (comma-separated goals)."""
        body_terms = []

        # Pattern for body goals: predicate(args)
        goal_pattern = r'([a-z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)'

        for match in re.finditer(goal_pattern, body_str):
            functor = match.group(1)
            args_str = match.group(2)
            args = self._parse_prolog_args(args_str)
            body_terms.append(self._build_term(functor, args))

        return body_terms

    def _build_term(self, functor: str, args: List[Term]) -> Term:
        """Build a compound term or atom from functor and args."""
        from .factories import atom, compound

        if not args:
            return atom(functor)
        return compound(functor, *args)

    def _parse_as_json(self, response: str) -> Optional[Tuple[List[Fact], List[Rule], Any, List[str]]]:
        """
        Parse response as JSON array of facts/rules

        Expected formats:
        - [["fact", ["parent", "alice", "bob"]], ...]
        - [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
        """
        facts = []
        rules = []
        errors = []

        try:
            # Try to parse as JSON, with preprocessing to handle unquoted strings
            # Many LLMs generate invalid JSON with unquoted variable names like:
            # [["rule", ["ancestor", X, Y], [["parent", X, Y]]]]
            # We need to quote these
            response_fixed = self._fix_unquoted_json(response)
            data = json.loads(response_fixed)
            
            if not isinstance(data, list):
                return None
            
            for item in data:
                if not isinstance(item, list) or len(item) < 2:
                    errors.append(f"Invalid item format: {item}")
                    continue
                
                item_type = item[0]
                
                try:
                    if item_type == "fact":
                        # Parse as fact
                        term = parse_prefix_notation(item[1])
                        facts.append(Fact(term))
                        
                    elif item_type == "rule" and len(item) >= 3:
                        # Parse as rule using native parser
                        rule = parse_rule(item)
                        rules.append(rule)
                        
                    else:
                        errors.append(f"Unknown item type: {item_type}")
                        
                except Exception as e:
                    errors.append(f"Error parsing {item_type}: {e}")
                    if self.strict:
                        return None
            
            return (facts, rules, data, errors)
            
        except json.JSONDecodeError:
            return None
    
    def _parse_as_sexp(self, response: str) -> Optional[Tuple[List[Fact], List[Rule], Any, List[str]]]:
        """
        Parse response as S-expressions (handling multi-line)

        Expected formats:
        - (parent alice bob)
        - (rule (grandparent X Z) ((parent X Y) (parent Y Z)))
        """
        facts = []
        rules = []
        errors = []
        parsed_items = []

        # Pre-process: Remove commas that LLMs often add incorrectly
        # Convert ", " to " " within S-expressions
        response = response.replace('),', ')')

        # Extract all complete S-expressions (including multi-line ones)
        expressions = []
        current_expr = ""
        paren_count = 0
        in_expr = False

        for char in response:
            if char == '(':
                in_expr = True
                paren_count += 1
                current_expr += char
            elif char == ')':
                if in_expr:
                    paren_count -= 1
                    current_expr += char
                    if paren_count == 0:
                        # Complete expression found
                        expressions.append(current_expr.strip())
                        current_expr = ""
                        in_expr = False
            elif in_expr:
                current_expr += char
        
        # Process each complete expression
        for expr in expressions:
            if not expr.startswith('('):
                continue
                
            try:
                # Check if it's a rule
                if expr.startswith('(rule '):
                    # Parse as rule: (rule (head ...) ((body1 ...) (body2 ...)))
                    content = expr[6:-1]  # Remove "(rule " and final ")"
                    
                    # Find the head (first S-expression)
                    paren_count = 0
                    head_end = 0
                    for i, char in enumerate(content):
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                            if paren_count == 0:
                                head_end = i + 1
                                break
                    
                    head_str = content[:head_end]
                    body_str = content[head_end:].strip()
                    
                    # Parse head
                    head_term = parse_s_expression(head_str)
                    
                    # Parse body - it should be ((term1) (term2) ...)
                    body_terms = []
                    if body_str.startswith('(') and body_str.endswith(')'):
                        body_content = body_str[1:-1]  # Remove outer parens
                        
                        # Split body into individual terms
                        current_term = ""
                        paren_count = 0
                        for char in body_content:
                            current_term += char
                            if char == '(':
                                paren_count += 1
                            elif char == ')':
                                paren_count -= 1
                                if paren_count == 0 and current_term.strip():
                                    body_terms.append(parse_s_expression(current_term.strip()))
                                    current_term = ""
                    
                    if head_term and body_terms:
                        rules.append(Rule(head_term, body_terms))
                        parsed_items.append(expr)
                else:
                    # Parse as a term/fact
                    term = parse_s_expression(expr)
                    facts.append(Fact(term))
                    parsed_items.append(expr)
                    
            except Exception as e:
                errors.append(f"Error parsing S-expression '{expr[:50]}...': {e}")
                if self.strict:
                    return None
        
        if facts or rules:
            return (facts, rules, parsed_items, errors)
        return None
    
    def _parse_with_extraction(self, response: str) -> Optional[Tuple[List[Fact], List[Rule], Any, List[str]]]:
        """
        Extract S-expressions or JSON from response text that might have other content
        """
        # First priority: Extract JSON from ```json code blocks
        json_code_block_pattern = r'```json\n?(.*?)```'
        json_blocks = re.findall(json_code_block_pattern, response, re.DOTALL)

        if json_blocks:
            # Try to parse each JSON block
            for block in json_blocks:
                result = self._parse_as_json(block.strip())
                if result:
                    return result

        # Second priority: S-expressions from markdown code blocks
        sexp_code_block_pattern = r'```(?:prolog|lisp|scheme|sexp)?\n?(.*?)```'
        sexp_blocks = re.findall(sexp_code_block_pattern, response, re.DOTALL)

        if sexp_blocks:
            # Combine all code blocks
            combined = '\n'.join(sexp_blocks)
            result = self._parse_as_sexp(combined)
            if result:
                return result

        # Try to parse the whole response as S-expressions
        result = self._parse_as_sexp(response)
        if result:
            return result

        # Fallback to JSON extraction anywhere in the text
        # Find the largest JSON structure (greedy match)
        match = re.search(r'\[.*\]', response, re.DOTALL)

        if match:
            json_str = match.group(0)
            result = self._parse_as_json(json_str)
            if result:
                return result

        return None
    
    def _parse_as_mixed(self, response: str) -> Optional[Tuple[List[Fact], List[Rule], Any, List[str]]]:
        """
        Try to parse response that might have mixed formats
        """
        facts = []
        rules = []
        errors = []

        # Pre-process: Remove commas that LLMs often add incorrectly
        response = response.replace('),', ')')

        # Split by lines and try to parse each
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try JSON first
            if line.startswith('['):
                try:
                    data = json.loads(line)
                    if isinstance(data, list) and len(data) > 0:
                        if data[0] == "fact":
                            term = parse_prefix_notation(data[1])
                            facts.append(Fact(term))
                        elif data[0] == "rule":
                            rule = parse_rule(data)
                            rules.append(rule)
                except:
                    pass

            # Try S-expression
            elif line.startswith('('):
                try:
                    # Check if it's a rule or a fact
                    if line.startswith('(rule '):
                        # Parse as rule: (rule (head ...) ((body1 ...) (body2 ...)))
                        # Extract the parts using a simple approach
                        content = line[6:-1]  # Remove "(rule " and final ")"

                        # Find the head (first S-expression)
                        paren_count = 0
                        head_end = 0
                        for i, char in enumerate(content):
                            if char == '(':
                                paren_count += 1
                            elif char == ')':
                                paren_count -= 1
                                if paren_count == 0:
                                    head_end = i + 1
                                    break

                        head_str = content[:head_end]
                        body_str = content[head_end:].strip()

                        # Parse head
                        head_term = parse_s_expression(head_str)

                        # Parse body - it should be ((term1) (term2) ...)
                        body_terms = []
                        if body_str.startswith('(') and body_str.endswith(')'):
                            body_content = body_str[1:-1]  # Remove outer parens

                            # Split body into individual terms
                            current_term = ""
                            paren_count = 0
                            for char in body_content:
                                current_term += char
                                if char == '(':
                                    paren_count += 1
                                elif char == ')':
                                    paren_count -= 1
                                    if paren_count == 0 and current_term.strip():
                                        body_terms.append(parse_s_expression(current_term.strip()))
                                        current_term = ""

                        if head_term and body_terms:
                            rules.append(Rule(head_term, body_terms))
                    else:
                        # Parse as fact
                        term = parse_s_expression(line)
                        facts.append(Fact(term))
                except Exception as e:
                    errors.append(f"S-expression parse error: {e}")
                    pass

        if facts or rules:
            return (facts, rules, lines, errors)
        return None

    def extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON structure from text.

        Tries to find JSON in code blocks first, then looks for raw JSON.

        Args:
            text: Text that may contain JSON

        Returns:
            JSON string or None if not found
        """
        # Try JSON code blocks first
        json_code_block_pattern = r'```json\n?(.*?)```'
        json_blocks = re.findall(json_code_block_pattern, text, re.DOTALL)

        if json_blocks:
            return json_blocks[0].strip()

        # Try to find raw JSON structure
        # Look for { ... } or [ ... ]
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            return match.group(0)

        return None

    def parse_rule_from_response(self, response: str) -> Optional[Rule]:
        """
        Parse a single rule from LLM response.

        Args:
            response: LLM response text

        Returns:
            Parsed Rule or None if parsing failed
        """
        # Parse the response
        knowledge = self.parse(response)

        # Return the first rule found
        if knowledge.rules:
            return knowledge.rules[0]

        return None


class DreamLogKnowledgeValidator:
    """
    Validator for parsed knowledge using DreamLog's evaluator
    """
    
    def __init__(self, kb=None, evaluator: Optional[PrologEvaluator] = None):
        """
        Initialize validator
        
        Args:
            kb: Knowledge base to validate against
            evaluator: Optional evaluator to use for validation
        """
        from .knowledge import KnowledgeBase
        
        self.kb = kb or KnowledgeBase()
        self.evaluator = evaluator or PrologEvaluator(self.kb)
    
    def validate_fact(self, fact: Fact) -> Tuple[bool, List[str]]:
        """
        Validate a fact
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check structural validity
        term = fact.term
        
        # Check if it's a valid term structure
        if isinstance(term, Variable):
            issues.append("Fact cannot be a variable")
            return False, issues
        
        if isinstance(term, Compound):
            # Check functor
            if not term.functor:
                issues.append("Compound term has empty functor")
                return False, issues
            
            # Check for unbound variables (facts should be ground)
            variables = term.get_variables()
            if variables:
                issues.append(f"Fact contains unbound variables: {variables}")
                # This might be okay in some contexts
        
        return len(issues) == 0, issues
    
    def validate_rule(self, rule: Rule) -> Tuple[bool, List[str]]:
        """
        Validate a rule
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check head
        if isinstance(rule.head, Atom):
            issues.append("Rule head cannot be an atom")
            return False, issues
        
        # Check body
        if len(rule.body) == 0:
            issues.append("Rule body is empty (use fact instead)")
        
        # Check for singleton variables (appear only once)
        head_vars = rule.head.get_variables()
        body_vars = set()
        for term in rule.body:
            body_vars.update(term.get_variables())
        
        # Variables in head should appear in body (unless it's intentional)
        head_only = head_vars - body_vars
        if head_only:
            issues.append(f"Variables in head not used in body: {head_only}")
        
        # Check for variables that appear only once in body
        var_counts = {}
        for term in rule.body:
            for var in term.get_variables():
                var_counts[var] = var_counts.get(var, 0) + 1
        
        singletons = [var for var, count in var_counts.items() if count == 1 and var not in head_vars]
        if singletons:
            issues.append(f"Singleton variables in body: {singletons}")
        
        return len(issues) == 0, issues
    
    def validate_knowledge(self, knowledge: ParsedKnowledge) -> Dict[str, Any]:
        """
        Validate all parsed knowledge
        
        Returns:
            Validation report with details
        """
        report = {
            'valid': True,
            'fact_issues': [],
            'rule_issues': [],
            'facts_validated': 0,
            'rules_validated': 0,
            'facts_valid': 0,
            'rules_valid': 0
        }
        
        # Validate facts
        for fact in knowledge.facts:
            report['facts_validated'] += 1
            is_valid, issues = self.validate_fact(fact)
            if is_valid:
                report['facts_valid'] += 1
            else:
                report['fact_issues'].append({
                    'fact': str(fact),
                    'issues': issues
                })
                report['valid'] = False
        
        # Validate rules
        for rule in knowledge.rules:
            report['rules_validated'] += 1
            is_valid, issues = self.validate_rule(rule)
            if is_valid:
                report['rules_valid'] += 1
            else:
                report['rule_issues'].append({
                    'rule': str(rule),
                    'issues': issues
                })
                report['valid'] = False
        
        return report


def parse_llm_response(response: str, strict: bool = False, validate: bool = True) -> Tuple[ParsedKnowledge, Optional[Dict]]:
    """
    Convenience function to parse and optionally validate an LLM response

    Args:
        response: Raw LLM response
        strict: Whether to use strict parsing
        validate: Whether to validate the parsed knowledge

    Returns:
        (parsed_knowledge, validation_report)
    """
    parser = DreamLogResponseParser(strict=strict)
    knowledge = parser.parse(response)

    validation_report = None
    if validate:
        validator = DreamLogKnowledgeValidator()
        validation_report = validator.validate_knowledge(knowledge)

    return knowledge, validation_report