"""
Output validation and feedback generation for LLM responses.

Provides analysis of LLM output quality and generates feedback
for retry attempts.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from .llm_response_parser import parse_llm_response


@dataclass
class ValidationResult:
    """Result of validating LLM output"""
    valid: bool
    parsed: Optional[Dict[str, Any]]
    close_but_wrong: bool
    score: float
    issues: list


class OutputValidator:
    """
    Validates LLM output and generates feedback for retries.

    Analyzes whether output contains valid facts/rules and
    provides specific feedback to improve retry attempts.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def analyze_output(self, raw_response: str) -> Dict[str, Any]:
        """
        Analyze LLM output for validity and quality.

        Returns dict with:
        - valid: bool - whether output is fully valid
        - parsed: dict or None - parsed facts and rules if valid
        - close_but_wrong: bool - output is almost valid but has issues
        - score: float - quality score 0-1
        - issues: list - specific issues found
        """
        if not raw_response or not raw_response.strip():
            return {
                'valid': False,
                'parsed': None,
                'close_but_wrong': False,
                'score': 0.0,
                'issues': ['Empty response']
            }

        try:
            # Try to parse the response
            # parse_llm_response returns (ParsedKnowledge, validation_dict)
            parsed_knowledge, validation = parse_llm_response(raw_response)

            # Extract facts and rules from ParsedKnowledge
            facts = parsed_knowledge.facts if hasattr(parsed_knowledge, 'facts') else []
            rules = parsed_knowledge.rules if hasattr(parsed_knowledge, 'rules') else []

            if facts or rules:
                return {
                    'valid': True,
                    'parsed': {
                        'facts': [str(f) for f in facts],
                        'rules': [str(r) for r in rules]
                    },
                    'close_but_wrong': False,
                    'score': 1.0,
                    'issues': []
                }
            else:
                # Parsed but empty
                return {
                    'valid': False,
                    'parsed': None,
                    'close_but_wrong': True,
                    'score': 0.3,
                    'issues': ['Parsed successfully but no facts or rules found']
                }

        except Exception as e:
            # Check if it's close to valid
            score, issues = self._analyze_parse_failure(raw_response, str(e))

            return {
                'valid': False,
                'parsed': None,
                'close_but_wrong': score > 0.3,
                'score': score,
                'issues': issues
            }

    def _analyze_parse_failure(self, raw_response: str, error: str) -> tuple:
        """
        Analyze a parse failure to determine how close to valid it was.

        Returns (score, issues) tuple.
        """
        issues = []
        score = 0.0

        # Check for common patterns that indicate partial validity
        response_lower = raw_response.lower()

        # Contains S-expression-like content
        if '(' in raw_response and ')' in raw_response:
            score += 0.3
        else:
            issues.append('Missing S-expression syntax (parentheses)')

        # Contains rule indicator
        if ':-' in raw_response:
            score += 0.2

        # Contains JSON-like structure
        if '{' in raw_response and '}' in raw_response:
            score += 0.1

        # Contains common keywords
        if any(kw in response_lower for kw in ['fact', 'rule', 'true', 'false']):
            score += 0.1

        # Add parse error as issue
        issues.append(f'Parse error: {error}')

        return min(score, 1.0), issues

    def generate_feedback_prompt(self, analysis: Dict[str, Any]) -> str:
        """
        Generate a feedback prompt based on the analysis.

        Used to provide specific guidance for retry attempts.
        """
        issues = analysis.get('issues', [])
        score = analysis.get('score', 0)

        feedback_parts = [
            "Your previous response could not be parsed correctly.",
            "Please fix the following issues:"
        ]

        for issue in issues:
            feedback_parts.append(f"- {issue}")

        feedback_parts.append("")
        feedback_parts.append("Remember to:")
        feedback_parts.append("- Use S-expression syntax: (predicate arg1 arg2)")
        feedback_parts.append("- For rules, use: (head X Y) :- (body1 X), (body2 Y)")
        feedback_parts.append("- Return ONLY the facts and rules, no explanation")

        return "\n".join(feedback_parts)
