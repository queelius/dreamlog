"""
DreamLog Interactive REPL (Read-Eval-Print-Loop)

Provides an interactive shell for working with DreamLog knowledge bases.
Supports both standalone usage and integration into other systems.
"""

import readline
import atexit
import os
import sys
import json
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .engine import DreamLogEngine
from .prefix_parser import parse_prefix_notation, parse_s_expression, term_to_sexp, term_to_prefix_json
from .knowledge import Fact, Rule
from .terms import atom, var, compound
from .llm_hook import LLMHook
from .llm_providers import MockLLMProvider


class ReplCommand(Enum):
    """REPL commands"""
    HELP = "help"
    EXIT = "exit"
    QUIT = "quit"
    CLEAR = "clear"
    SAVE = "save"
    LOAD = "load"
    STATS = "stats"
    TRACE = "trace"
    FORMAT = "format"
    LLM = "llm"
    HISTORY = "history"
    RESET = "reset"
    EXPLAIN = "explain"


@dataclass
class ReplConfig:
    """REPL configuration"""
    prompt: str = "dreamlog> "
    history_file: str = "~/.dreamlog_history"
    max_solutions: int = 10
    auto_save: bool = False
    auto_save_file: str = "~/.dreamlog_autosave.json"
    trace_enabled: bool = False
    format: str = "sexp"  # "sexp" or "json"
    color_output: bool = True
    llm_enabled: bool = False
    llm_domain: str = "general"


class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'


class DreamLogRepl:
    """
    Interactive REPL for DreamLog
    
    Features:
    - Interactive query evaluation
    - Add facts and rules
    - Save/load knowledge bases
    - Command history
    - Tab completion
    - Syntax highlighting
    """
    
    def __init__(self, config: Optional[ReplConfig] = None,
                 output_handler: Optional[Callable[[str], None]] = None):
        """
        Initialize the REPL
        
        Args:
            config: Configuration options
            output_handler: Custom output handler (for integration)
        """
        self.config = config or ReplConfig()
        self.engine = DreamLogEngine()
        self.output_handler = output_handler or print
        self.running = False
        self.query_count = 0
        self.command_history = []
        
        # Setup LLM if enabled
        if self.config.llm_enabled:
            self._setup_llm()
        
        # Setup readline for better terminal interaction
        self._setup_readline()
    
    def _setup_llm(self):
        """Setup LLM integration"""
        llm = MockLLMProvider(knowledge_domain=self.config.llm_domain)
        self.engine.llm_hook = LLMHook(llm)
    
    def _setup_readline(self):
        """Setup readline for history and tab completion"""
        # History file
        history_file = os.path.expanduser(self.config.history_file)
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass
        
        # Save history on exit
        atexit.register(readline.write_history_file, history_file)
        
        # Tab completion
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._completer)
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for commands and functors"""
        options = []
        
        if text.startswith(":"):
            # Complete commands
            commands = [f":{cmd.value}" for cmd in ReplCommand]
            options = [cmd for cmd in commands if cmd.startswith(text)]
        else:
            # Complete functors
            functors = set()
            for fact in self.engine.kb.facts:
                if hasattr(fact.term, 'functor'):
                    functors.add(fact.term.functor)
            for rule in self.engine.kb.rules:
                if hasattr(rule.head, 'functor'):
                    functors.add(rule.head.functor)
            
            if self.config.format == "sexp":
                options = [f"({f}" for f in functors if f.startswith(text[1:] if text.startswith("(") else text)]
            else:
                options = [f'["{f}"' for f in functors if f.startswith(text[2:] if text.startswith('["') else text)]
        
        return options[state] if state < len(options) else None
    
    def _colorize(self, text: str, color: str) -> str:
        """Add color to text if colors are enabled"""
        if self.config.color_output:
            return f"{color}{text}{Colors.RESET}"
        return text
    
    def _print_banner(self):
        """Print welcome banner"""
        banner = """
╔════════════════════════════════════════╗
║         DreamLog Interactive REPL          ║
║    Logic Programming with S-expressions║
╚════════════════════════════════════════╝
        """
        self.output_handler(self._colorize(banner, Colors.CYAN))
        self.output_handler(f"Type {self._colorize(':help', Colors.YELLOW)} for commands")
        self.output_handler("")
    
    def _print_help(self):
        """Print help information"""
        help_text = """
Commands:
  :help              Show this help message
  :exit, :quit       Exit the REPL
  :clear             Clear the knowledge base
  :save <file>       Save knowledge base to file
  :load <file>       Load knowledge base from file
  :stats             Show knowledge base statistics
  :trace on/off      Enable/disable unification tracing
  :format sexp/json  Set input format (current: {})
  :llm on/off        Enable/disable LLM integration
  :history           Show command history
  :reset             Reset the REPL
  :explain <query>   Explain query resolution

Syntax:
  Facts:     (parent john mary)  or  ["parent", "john", "mary"]
  Rules:     (grandparent X Z) :- (parent X Y), (parent Y Z)
  Queries:   (parent john X)  or  ["parent", "john", "X"]
  
Variables start with uppercase letters.
Press Tab for autocompletion.
        """.format(self.config.format)
        self.output_handler(help_text)
    
    def _parse_input(self, line: str) -> Any:
        """Parse input based on current format"""
        if self.config.format == "sexp":
            if line.startswith("("):
                return parse_s_expression(line)
            else:
                # Try as atom/variable
                if line[0].isupper():
                    return var(line)
                else:
                    return atom(line)
        else:  # json format
            if line.startswith("["):
                return parse_prefix_notation(json.loads(line))
            else:
                # Try parsing as JSON string
                return parse_prefix_notation(json.loads(f'"{line}"'))
    
    def _handle_command(self, command: str, args: str) -> bool:
        """
        Handle REPL commands
        
        Returns:
            True to continue, False to exit
        """
        try:
            cmd = ReplCommand(command)
        except ValueError:
            self.output_handler(self._colorize(f"Unknown command: :{command}", Colors.RED))
            return True
        
        if cmd in (ReplCommand.EXIT, ReplCommand.QUIT):
            return False
        
        elif cmd == ReplCommand.HELP:
            self._print_help()
        
        elif cmd == ReplCommand.CLEAR:
            self.engine.clear()
            self.output_handler(self._colorize("✓ Knowledge base cleared", Colors.GREEN))
        
        elif cmd == ReplCommand.SAVE:
            if not args:
                self.output_handler(self._colorize("Usage: :save <filename>", Colors.YELLOW))
            else:
                self._save_kb(args.strip())
        
        elif cmd == ReplCommand.LOAD:
            if not args:
                self.output_handler(self._colorize("Usage: :load <filename>", Colors.YELLOW))
            else:
                self._load_kb(args.strip())
        
        elif cmd == ReplCommand.STATS:
            self._show_stats()
        
        elif cmd == ReplCommand.TRACE:
            if args.strip().lower() == "on":
                self.config.trace_enabled = True
                self.output_handler(self._colorize("✓ Tracing enabled", Colors.GREEN))
            elif args.strip().lower() == "off":
                self.config.trace_enabled = False
                self.output_handler(self._colorize("✓ Tracing disabled", Colors.GREEN))
            else:
                status = "on" if self.config.trace_enabled else "off"
                self.output_handler(f"Tracing is {status}")
        
        elif cmd == ReplCommand.FORMAT:
            if args.strip() in ("sexp", "json"):
                self.config.format = args.strip()
                self.output_handler(self._colorize(f"✓ Format set to {self.config.format}", Colors.GREEN))
            else:
                self.output_handler(f"Current format: {self.config.format}")
        
        elif cmd == ReplCommand.LLM:
            if args.strip().lower() == "on":
                self.config.llm_enabled = True
                self._setup_llm()
                self.output_handler(self._colorize("✓ LLM integration enabled", Colors.GREEN))
            elif args.strip().lower() == "off":
                self.config.llm_enabled = False
                self.engine.llm_hook = None
                self.output_handler(self._colorize("✓ LLM integration disabled", Colors.GREEN))
            else:
                status = "on" if self.config.llm_enabled else "off"
                self.output_handler(f"LLM integration is {status}")
        
        elif cmd == ReplCommand.HISTORY:
            self._show_history()
        
        elif cmd == ReplCommand.RESET:
            self.engine = DreamLogEngine()
            self.query_count = 0
            self.command_history = []
            if self.config.llm_enabled:
                self._setup_llm()
            self.output_handler(self._colorize("✓ REPL reset", Colors.GREEN))
        
        elif cmd == ReplCommand.EXPLAIN:
            if not args:
                self.output_handler(self._colorize("Usage: :explain <query>", Colors.YELLOW))
            else:
                self._explain_query(args.strip())
        
        return True
    
    def _save_kb(self, filename: str):
        """Save knowledge base to file"""
        try:
            kb_data = self.engine.save_to_prefix()
            with open(filename, 'w') as f:
                f.write(kb_data)
            self.output_handler(self._colorize(f"✓ Saved to {filename}", Colors.GREEN))
        except Exception as e:
            self.output_handler(self._colorize(f"✗ Error saving: {e}", Colors.RED))
    
    def _load_kb(self, filename: str):
        """Load knowledge base from file"""
        try:
            with open(filename, 'r') as f:
                kb_data = f.read()
            self.engine.load_from_prefix(kb_data)
            self.output_handler(self._colorize(f"✓ Loaded from {filename}", Colors.GREEN))
            self._show_stats()
        except Exception as e:
            self.output_handler(self._colorize(f"✗ Error loading: {e}", Colors.RED))
    
    def _show_stats(self):
        """Show knowledge base statistics"""
        num_facts = len(self.engine.kb.facts)
        num_rules = len(self.engine.kb.rules)
        
        functors = set()
        for fact in self.engine.kb.facts:
            if hasattr(fact.term, 'functor'):
                functors.add(fact.term.functor)
        for rule in self.engine.kb.rules:
            if hasattr(rule.head, 'functor'):
                functors.add(rule.head.functor)
        
        self.output_handler(self._colorize("📊 Knowledge Base Statistics:", Colors.CYAN))
        self.output_handler(f"  Facts: {num_facts}")
        self.output_handler(f"  Rules: {num_rules}")
        self.output_handler(f"  Functors: {len(functors)}")
        if functors:
            self.output_handler(f"    {', '.join(sorted(functors))}")
        self.output_handler(f"  Queries executed: {self.query_count}")
    
    def _show_history(self):
        """Show command history"""
        if not self.command_history:
            self.output_handler("No commands in history")
            return
        
        self.output_handler(self._colorize("📜 Command History:", Colors.CYAN))
        for i, cmd in enumerate(self.command_history[-20:], 1):
            self.output_handler(f"  {i}. {cmd}")
    
    def _explain_query(self, query_str: str):
        """Explain how a query would be resolved"""
        try:
            query_term = self._parse_input(query_str)
            
            from .unification import Unifier
            unifier = Unifier(trace=True)
            
            self.output_handler(self._colorize(f"Explaining: {term_to_sexp(query_term)}", Colors.CYAN))
            
            # Check facts
            matching_facts = []
            for fact in self.engine.kb.facts:
                result = unifier.unify(query_term, fact.term)
                if result.success:
                    matching_facts.append((fact, result))
            
            if matching_facts:
                self.output_handler(self._colorize("Matching facts:", Colors.GREEN))
                for fact, result in matching_facts:
                    self.output_handler(f"  • {term_to_sexp(fact.term)}")
                    if result.bindings:
                        bindings = ", ".join(f"{k}={v}" for k, v in result.bindings.items())
                        self.output_handler(f"    Bindings: {bindings}")
            
            # Check rules
            matching_rules = []
            for rule in self.engine.kb.rules:
                result = unifier.unify(query_term, rule.head)
                if result.success:
                    matching_rules.append((rule, result))
            
            if matching_rules:
                self.output_handler(self._colorize("Matching rules:", Colors.YELLOW))
                for rule, result in matching_rules:
                    body_str = ", ".join(term_to_sexp(b) for b in rule.body)
                    self.output_handler(f"  • {term_to_sexp(rule.head)} :- {body_str}")
                    if result.bindings:
                        bindings = ", ".join(f"{k}={v}" for k, v in result.bindings.items())
                        self.output_handler(f"    Initial bindings: {bindings}")
            
            if not matching_facts and not matching_rules:
                self.output_handler(self._colorize("No matching facts or rules found", Colors.GRAY))
        
        except Exception as e:
            self.output_handler(self._colorize(f"✗ Error: {e}", Colors.RED))
    
    def _process_statement(self, line: str):
        """Process a statement (fact, rule, or query)"""
        try:
            # Check if it's a rule (contains :-)
            if ":-" in line:
                self._process_rule(line)
            else:
                # Try to parse the input
                term = self._parse_input(line.strip())
                
                # Check if it's a query (contains variables)
                has_vars = False
                if hasattr(term, 'get_variables'):
                    has_vars = bool(term.get_variables())
                
                if has_vars or line.strip().endswith("?"):
                    # It's a query
                    self._process_query(term)
                else:
                    # It's a fact
                    self._process_fact(term)
        
        except Exception as e:
            self.output_handler(self._colorize(f"✗ Error: {e}", Colors.RED))
    
    def _process_fact(self, term):
        """Process a fact"""
        self.engine.add_fact(Fact(term))
        self.output_handler(self._colorize(f"✓ Added fact: {term_to_sexp(term)}", Colors.GREEN))
    
    def _process_rule(self, line: str):
        """Process a rule"""
        parts = line.split(":-")
        if len(parts) != 2:
            raise ValueError("Invalid rule format")
        
        head = self._parse_input(parts[0].strip())
        
        # Parse body terms
        body = []
        body_str = parts[1].strip()
        
        if self.config.format == "sexp":
            # Split by commas and parse each term
            for term_str in body_str.split(","):
                body.append(self._parse_input(term_str.strip()))
        else:
            # Parse as JSON array
            body_terms = json.loads(f"[{body_str}]")
            body = [parse_prefix_notation(t) for t in body_terms]
        
        self.engine.add_rule(Rule(head, body))
        body_repr = ", ".join(term_to_sexp(b) for b in body)
        self.output_handler(self._colorize(f"✓ Added rule: {term_to_sexp(head)} :- {body_repr}", Colors.GREEN))
    
    def _process_query(self, term):
        """Process a query"""
        self.query_count += 1
        
        solutions = []
        for i, solution in enumerate(self.engine.query([term])):
            if i >= self.config.max_solutions:
                break
            solutions.append(solution)
        
        if solutions:
            self.output_handler(self._colorize(f"Found {len(solutions)} solution(s):", Colors.BLUE))
            for i, sol in enumerate(solutions, 1):
                if sol.ground_bindings:
                    bindings = ", ".join(f"{k}={v}" for k, v in sol.ground_bindings.items())
                    self.output_handler(f"  {i}. {bindings}")
                else:
                    self.output_handler(f"  {i}. Yes")
        else:
            self.output_handler(self._colorize("No solutions found.", Colors.GRAY))
    
    def run(self):
        """Run the interactive REPL"""
        self.running = True
        self._print_banner()
        
        while self.running:
            try:
                # Get input
                line = input(self._colorize(self.config.prompt, Colors.BOLD))
                line = line.strip()
                
                if not line:
                    continue
                
                # Add to history
                self.command_history.append(line)
                
                # Check for commands
                if line.startswith(":"):
                    parts = line[1:].split(maxsplit=1)
                    command = parts[0] if parts else ""
                    args = parts[1] if len(parts) > 1 else ""
                    
                    if not self._handle_command(command, args):
                        break
                else:
                    # Process as statement
                    self._process_statement(line)
                
                # Auto-save if enabled
                if self.config.auto_save:
                    self._save_kb(os.path.expanduser(self.config.auto_save_file))
            
            except KeyboardInterrupt:
                self.output_handler("\n" + self._colorize("Use :exit to quit", Colors.YELLOW))
            except EOFError:
                break
            except Exception as e:
                self.output_handler(self._colorize(f"✗ Unexpected error: {e}", Colors.RED))
        
        self.output_handler(self._colorize("\nGoodbye!", Colors.CYAN))
    
    def execute_line(self, line: str) -> str:
        """
        Execute a single line and return the output
        
        This method is for integration with other systems (like API server)
        """
        output_lines = []
        
        def capture_output(text):
            output_lines.append(text)
        
        old_handler = self.output_handler
        self.output_handler = capture_output
        
        try:
            line = line.strip()
            
            if line.startswith(":"):
                parts = line[1:].split(maxsplit=1)
                command = parts[0] if parts else ""
                args = parts[1] if len(parts) > 1 else ""
                self._handle_command(command, args)
            else:
                self._process_statement(line)
        
        finally:
            self.output_handler = old_handler
        
        return "\n".join(output_lines)


def main():
    """Main entry point for standalone REPL"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DreamLog Interactive REPL")
    parser.add_argument("--kb", help="Load knowledge base on startup")
    parser.add_argument("--format", choices=["sexp", "json"], default="sexp",
                       help="Input format")
    parser.add_argument("--llm", action="store_true", help="Enable LLM integration")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    
    args = parser.parse_args()
    
    # Create config
    config = ReplConfig(
        format=args.format,
        llm_enabled=args.llm,
        color_output=not args.no_color
    )
    
    # Create and run REPL
    repl = DreamLogRepl(config)
    
    # Load KB if specified
    if args.kb:
        try:
            with open(args.kb, 'r') as f:
                repl.engine.load_from_prefix(f.read())
            print(f"Loaded knowledge base from {args.kb}")
        except Exception as e:
            print(f"Error loading {args.kb}: {e}")
    
    # Run the REPL
    repl.run()


if __name__ == "__main__":
    main()