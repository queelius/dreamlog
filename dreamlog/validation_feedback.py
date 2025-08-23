"""
Validation and Feedback System for LLM Outputs

Provides detailed feedback to guide LLMs toward correct output format.
"""

import json
from typing import Any, Dict, List, Tuple, Optional


class OutputValidator:
    """Validates LLM output and provides constructive feedback"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def analyze_output(self, raw_response: str) -> Dict[str, Any]:
        """
        Analyze LLM output and provide detailed feedback
        
        Returns:
            {
                'valid': bool,
                'parsed': parsed_data or None,
                'score': float (0-1),
                'issues': List[str],
                'suggestions': List[str],
                'close_but_wrong': bool
            }
        """
        result = {
            'valid': False,
            'parsed': None,
            'score': 0.0,
            'issues': [],
            'suggestions': [],
            'close_but_wrong': False
        }
        
        # Try to extract JSON
        json_str = self._extract_json(raw_response)
        if not json_str:
            result['issues'].append("No JSON found in response")
            result['suggestions'].append("Response must contain a JSON array")
            return result
        
        # Try to parse JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            result['issues'].append(f"Invalid JSON: {e}")
            result['suggestions'].append("Ensure proper JSON syntax with matching brackets and quotes")
            
            # Check if it's close to valid JSON
            if '[' in json_str and ']' in json_str:
                result['close_but_wrong'] = True
                result['score'] = 0.3
            return result
        
        # Analyze the parsed structure
        return self._analyze_structure(parsed, result)
    
    def _analyze_structure(self, parsed: Any, result: Dict) -> Dict:
        """Analyze the structure of parsed JSON"""
        
        # Check if it's an array
        if not isinstance(parsed, list):
            # Check if it's the Ollama dict format
            if isinstance(parsed, dict):
                if 'rule' in parsed and 'conditions' in parsed:
                    result['close_but_wrong'] = True
                    result['score'] = 0.7
                    result['issues'].append("Output is a dictionary, not an array")
                    result['suggestions'].append(
                        f"Convert {json.dumps(parsed, separators=(',', ':'))} "
                        f"to [['rule', {json.dumps(parsed['rule'])}, {json.dumps(parsed['conditions'])}]]"
                    )
                    # Try to convert it
                    converted = [['rule', parsed['rule'], parsed['conditions']]]
                    result['parsed'] = self._validate_strict(converted)
                else:
                    result['issues'].append("Output must be an array, not a dictionary")
                    result['score'] = 0.2
            else:
                result['issues'].append(f"Output must be a JSON array, got {type(parsed).__name__}")
            return result
        
        # It's an array - check each item
        valid_items = []
        for i, item in enumerate(parsed):
            item_result = self._analyze_item(item, i)
            if item_result['valid']:
                valid_items.append(item_result['parsed'])
            else:
                result['issues'].extend(item_result['issues'])
                result['suggestions'].extend(item_result['suggestions'])
                if item_result['close']:
                    result['close_but_wrong'] = True
        
        # Calculate score
        if len(parsed) > 0:
            result['score'] = len(valid_items) / len(parsed)
        
        if valid_items:
            result['parsed'] = {'facts': [], 'rules': []}
            for item in valid_items:
                if item['type'] == 'fact':
                    result['parsed']['facts'].append(item['data'])
                else:
                    result['parsed']['rules'].append(item['data'])
            
            if result['score'] == 1.0:
                result['valid'] = True
        
        return result
    
    def _analyze_item(self, item: Any, index: int) -> Dict:
        """Analyze a single item from the array"""
        result = {'valid': False, 'parsed': None, 'issues': [], 'suggestions': [], 'close': False}
        
        if not isinstance(item, list):
            result['issues'].append(f"Item {index}: Expected array, got {type(item).__name__}")
            
            # Check if it's close (dict with rule/fact keys)
            if isinstance(item, dict):
                if 'rule' in item or 'fact' in item:
                    result['close'] = True
                    result['suggestions'].append(
                        f"Item {index}: Convert dictionary to array format"
                    )
            return result
        
        if len(item) < 2:
            result['issues'].append(f"Item {index}: Too few elements (need at least 2)")
            return result
        
        item_type = item[0]
        
        if item_type == 'fact':
            if len(item) != 2:
                result['issues'].append(f"Item {index}: Fact needs exactly 2 elements")
                result['close'] = True
            elif not isinstance(item[1], list):
                result['issues'].append(f"Item {index}: Fact data must be an array")
                result['close'] = True
            else:
                result['valid'] = True
                result['parsed'] = {'type': 'fact', 'data': item[1]}
                
        elif item_type == 'rule':
            if len(item) != 3:
                result['issues'].append(f"Item {index}: Rule needs exactly 3 elements [\"rule\", head, body]")
                result['close'] = True
                if len(item) == 2:
                    result['suggestions'].append(
                        f"Item {index}: Missing body? Should be [\"rule\", {json.dumps(item[1])}, [conditions...]]"
                    )
            else:
                head = item[1]
                body = item[2]
                
                if not isinstance(head, list):
                    result['issues'].append(f"Item {index}: Rule head must be an array")
                    result['close'] = True
                elif not isinstance(body, list):
                    result['issues'].append(f"Item {index}: Rule body must be an array")
                    result['close'] = True
                else:
                    # Check body predicates
                    invalid_preds = []
                    for j, pred in enumerate(body):
                        if not isinstance(pred, list):
                            invalid_preds.append(j)
                    
                    if invalid_preds:
                        result['issues'].append(
                            f"Item {index}: Body conditions {invalid_preds} are not arrays"
                        )
                        result['close'] = True
                        result['suggestions'].append(
                            f"Each condition in the body must be an array like [\"parent\", \"X\", \"Y\"]"
                        )
                    else:
                        result['valid'] = True
                        result['parsed'] = {'type': 'rule', 'data': [head, body]}
        else:
            result['issues'].append(f"Item {index}: Unknown type '{item_type}' (expected 'fact' or 'rule')")
            if item_type in ['parent', 'grandparent', 'sibling']:
                result['close'] = True
                result['suggestions'].append(
                    f"Item {index}: Did you mean [\"fact\", {json.dumps(item)}] or [\"rule\", ...]?"
                )
        
        return result
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text"""
        import re
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON array or object
        patterns = [
            r'\[.*\]',  # Array
            r'\{.*\}'   # Object
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
        
        return text.strip() if text.strip().startswith('[') or text.strip().startswith('{') else None
    
    def _validate_strict(self, items: List) -> Optional[Dict]:
        """Strict validation of properly formatted items"""
        facts = []
        rules = []
        
        for item in items:
            if isinstance(item, list) and len(item) >= 2:
                if item[0] == 'fact' and len(item) == 2 and isinstance(item[1], list):
                    facts.append(item[1])
                elif item[0] == 'rule' and len(item) == 3:
                    if isinstance(item[1], list) and isinstance(item[2], list):
                        if all(isinstance(p, list) for p in item[2]):
                            rules.append([item[1], item[2]])
        
        if facts or rules:
            return {'facts': facts, 'rules': rules}
        return None
    
    def generate_feedback_prompt(self, analysis: Dict) -> str:
        """Generate a helpful feedback prompt based on analysis"""
        
        if analysis['valid']:
            return ""
        
        prompt_parts = []
        
        if analysis['close_but_wrong']:
            prompt_parts.append("Your output is close but needs small corrections:")
        else:
            prompt_parts.append("Your output needs to be reformatted:")
        
        # Add specific issues
        if analysis['issues']:
            prompt_parts.append("\nIssues found:")
            for issue in analysis['issues'][:3]:  # Limit to 3 main issues
                prompt_parts.append(f"  - {issue}")
        
        # Add suggestions
        if analysis['suggestions']:
            prompt_parts.append("\nSuggestions:")
            for suggestion in analysis['suggestions'][:3]:
                prompt_parts.append(f"  - {suggestion}")
        
        # Add correct format reminder
        prompt_parts.append("\nCorrect format examples:")
        prompt_parts.append('[["fact", ["parent", "alice", "bob"]]]')
        prompt_parts.append('[["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]')
        
        prompt_parts.append("\nPlease provide your response as a JSON array in exactly this format.")
        
        return "\n".join(prompt_parts)