#!/usr/bin/env python3
"""
DreamLog TUI (Terminal User Interface)

Comprehensive interactive interface for DreamLog with:
- Knowledge base management (load, save, inspect)
- Query evaluation (ask, find-all, trace)
- Sleep-phase operations (compress, consolidate, dream)
- LLM control and debugging
- Rich terminal output with colors and formatting
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import os
import sys
import readline
import atexit
import json
import time
import subprocess
from pathlib import Path

from dreamlog.engine import DreamLogEngine
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Term, Atom, Variable, Compound
from dreamlog.prefix_parser import parse_s_expression
from dreamlog.llm_providers import create_provider
from dreamlog.llm_hook import LLMHook
from dreamlog.embedding_providers import OllamaEmbeddingProvider, TfIdfEmbeddingProvider
from dreamlog.prompt_template_system import RULE_EXAMPLES
from dreamlog.config import DreamLogConfig, get_config


class TUICommand(Enum):
    """All available TUI commands"""
    # Help and navigation
    HELP = "help"
    EXIT = "exit"
    QUIT = "quit"
    CLEAR = "clear"
    CD = "cd"  # Change directory

    # Knowledge management
    LOAD = "load"
    SAVE = "save"
    IMPORT = "import"
    EXPORT = "export"
    REMOVE_FACT = "remove-fact"
    REMOVE_RULE = "remove-rule"

    # Query operations
    ASK = "ask"
    FIND_ALL = "find-all"
    PROVE = "prove"

    # Knowledge inspection
    FACTS = "facts"
    RULES = "rules"
    STATS = "stats"
    SHOW = "show"
    SEARCH = "search"

    # Sleep phase operations
    COMPRESS = "compress"
    CONSOLIDATE = "consolidate"
    DREAM = "dream"
    SLEEP = "sleep"
    ANALYZE = "analyze"

    # LLM control
    LLM = "llm"
    SET_PROVIDER = "set-provider"
    SET_MODEL = "set-model"
    SET_TEMPERATURE = "set-temperature"
    MODEL = "model"  # Get/set current model
    MODELS = "models"  # List available models
    TEMPERATURE = "temperature"  # Get/set temperature
    MAX_TOKENS = "max-tokens"  # Get/set max tokens
    PROVIDER = "provider"  # Show current provider info

    # Embedding control
    EMBEDDING_MODELS = "embedding-models"
    EMBEDDING_MODEL_INFO = "embedding-model-info"
    SET_EMBEDDING_MODEL = "set-embedding-model"

    # Debugging
    TRACE = "trace"
    DEBUG = "debug"
    EXPLAIN = "explain"
    HISTORY = "history"
    RESET = "reset"
    BENCHMARK = "benchmark"


class Colors:
    """ANSI color codes"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'
    WHITE = '\033[97m'


class DreamLogTUI:
    """
    Comprehensive TUI for DreamLog

    Provides rich interactive interface with:
    - Full command set for knowledge management and querying
    - Sleep-phase operations for knowledge compression
    - LLM integration and control
    - Debugging and analysis tools
    """

    def __init__(self, config: Optional[DreamLogConfig] = None):
        # Load config from file or use provided config
        self.config = config or get_config()

        self.engine = DreamLogEngine()
        self.running = False
        self.query_count = 0
        self.sleep_count = 0
        self.command_history = []

        # Setup LLM if enabled
        if self.config.llm_enabled:
            self._setup_llm()

        # Setup readline
        self._setup_readline()

        # Stats tracking
        self.stats = {
            'queries': 0,
            'solutions_found': 0,
            'llm_calls': 0,
            'sleep_cycles': 0,
            'compression_ratio': 1.0
        }

    def _setup_llm(self):
        """Setup LLM integration"""
        try:
            # Use DreamLogConfig provider settings
            provider_kwargs = {
                'model': self.config.provider.model,
                'temperature': self.config.provider.temperature,
                'max_tokens': self.config.provider.max_tokens,
                'timeout': self.config.provider.timeout,
            }

            if self.config.provider.base_url:
                provider_kwargs['base_url'] = self.config.provider.base_url

            api_key = self.config.provider.get_api_key()
            if api_key:
                provider_kwargs['api_key'] = api_key

            provider = create_provider(
                provider_type=self.config.provider.provider,
                **provider_kwargs
            )
            # Pass debug callback to route LLM debug messages through TUI
            def llm_debug_callback(message: str):
                self._print(self._colorize(message, Colors.MAGENTA))

            # Create embedding provider for RAG-based example selection
            # Try Ollama embeddings, fall back to TF-IDF if unavailable
            embedding_provider = None
            try:
                embedding_provider = OllamaEmbeddingProvider(
                    base_url=self.config.provider.base_url or "http://localhost:11434",
                    model="nomic-embed-text"
                )
                self._print(self._colorize("✓ Ollama embedding provider initialized", Colors.GREEN))
            except Exception as e:
                self._print(self._colorize(f"⚠ Ollama embeddings unavailable: {e}", Colors.YELLOW))
                self._print(self._colorize("  Using TF-IDF embeddings (local fallback)", Colors.YELLOW))
                embedding_provider = TfIdfEmbeddingProvider(RULE_EXAMPLES)

            hook = LLMHook(
                provider,
                embedding_provider,
                debug=self.config.debug_enabled,
                debug_callback=llm_debug_callback
            )
            self.engine.llm_hook = hook
            self.engine.evaluator.unknown_hook = hook  # Update evaluator's hook too
            self._print(self._colorize("✓ LLM integration enabled", Colors.GREEN))
        except Exception as e:
            self._print(self._colorize(f"⚠ LLM setup failed: {e}", Colors.YELLOW))

    def _setup_readline(self):
        """Setup readline for history and completion"""
        history_file = os.path.expanduser(self.config.tui_history_file)
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass

        atexit.register(readline.write_history_file, history_file)

        # Tab completion
        readline.parse_and_bind('tab: complete')
        readline.set_completer(self._completer)

    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for commands and functors"""
        options = []

        # Command completion
        if text.startswith('/'):
            commands = [f"/{cmd.value}" for cmd in TUICommand]
            options = [cmd for cmd in commands if cmd.startswith(text)]
        else:
            # Functor completion
            functors = set()
            for fact in self.engine.kb.facts:
                if hasattr(fact.term, 'functor'):
                    functors.add(fact.term.functor)
            for rule in self.engine.kb.rules:
                if hasattr(rule.head, 'functor'):
                    functors.add(rule.head.functor)

            options = [f"({f}" for f in functors if f.startswith(text[1:] if text.startswith("(") else text)]

        return options[state] if state < len(options) else None

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if enabled"""
        if self.config.tui_color_output:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _print(self, text: str):
        """Print with proper formatting"""
        print(text)

    def _print_banner(self):
        """Print startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║              DreamLog Terminal User Interface                ║
║     Logic Programming with Neural-Symbolic Integration       ║
╚══════════════════════════════════════════════════════════════╝
        """
        self._print(self._colorize(banner, Colors.CYAN))
        self._print(f"Type {self._colorize('/help', Colors.YELLOW)} for available commands\n")

    def _print_help(self):
        """Print comprehensive help"""
        help_text = f"""
{self._colorize("KNOWLEDGE MANAGEMENT", Colors.BOLD)}
  /load <file>         Load knowledge base from file
  /save <file>         Save knowledge base to file
  /import <file>       Import additional knowledge
  /clear               Clear all knowledge
  /facts               List all facts (with indices)
  /rules               List all rules (with indices)
  /remove-fact <i>     Remove fact by index
  /remove-rule <i>     Remove rule by index
  /stats               Show knowledge base statistics
  /show <functor>      Show all facts/rules for a functor
  /search <pattern>    Search for patterns in KB

{self._colorize("QUERY OPERATIONS", Colors.BOLD)}
  /ask <query>         Ask a query (find one solution)
  /find-all <query>    Find all solutions
  /prove <query>       Prove query with detailed trace
  /trace on|off        Enable/disable query tracing

{self._colorize("SLEEP PHASE OPERATIONS", Colors.BOLD)}
  /sleep               Run full sleep cycle (compress + consolidate)
  /compress            Compress knowledge base
  /consolidate         Consolidate learned knowledge
  /dream               Run dream cycle (explore new patterns)
  /analyze             Analyze compression opportunities

{self._colorize("LLM CONTROL", Colors.BOLD)}
  /llm on|off          Enable/disable LLM integration
  /model [name]        Get/set current LLM model
  /models              List available models (Ollama only)
  /temperature [val]   Get/set temperature (0.0-2.0)
  /max-tokens [num]    Get/set max token limit
  /provider            Show current provider info
  /debug on|off        Enable/disable LLM debug output

{self._colorize("DEBUGGING & ANALYSIS", Colors.BOLD)}
  /explain <query>     Explain how query was resolved
  /benchmark           Run performance benchmarks
  /history             Show command history
  /reset               Reset the TUI

{self._colorize("GENERAL", Colors.BOLD)}
  /help                Show this help
  /exit, /quit         Exit TUI
  /cd <path>           Change current directory

{self._colorize("SHELL COMMANDS", Colors.BOLD)}
  !<command>           Execute shell command
  Examples:
    !ls                List files in current directory
    !cat file.dl       Display file contents
    !pwd               Show current directory

{self._colorize("SYNTAX", Colors.BOLD)}
  Facts:     (parent john mary)
  Rules:     (grandparent X Z) :- (parent X Y), (parent Y Z)
  Queries:   Use /ask or /find-all commands

Variables start with uppercase letters (X, Y, Z, Person, etc.)
"""
        self._print(help_text)

    def run(self):
        """Main TUI loop"""
        self._print_banner()
        self.running = True

        while self.running:
            try:
                # Get input
                line = input(self._colorize(self.config.tui_prompt, Colors.BOLD)).strip()

                if not line:
                    continue

                # Track history
                self.command_history.append(line)

                # Handle commands, shell commands, or direct input
                if line.startswith('/'):
                    self._handle_command_line(line)
                elif line.startswith('!'):
                    self._handle_shell_command(line[1:])
                else:
                    # Direct S-expression input (fact or rule)
                    self._handle_direct_input(line)

            except KeyboardInterrupt:
                self._print("\nUse /exit to quit")
            except EOFError:
                self._print("\n" + self._colorize("Goodbye!", Colors.CYAN))
                break
            except Exception as e:
                self._print(self._colorize(f"✗ Error: {e}", Colors.RED))
                if self.config.debug_enabled:
                    import traceback
                    traceback.print_exc()

    def _handle_command_line(self, line: str):
        """Handle command line"""
        # Parse command and arguments
        parts = line[1:].split(maxsplit=1)
        command_str = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        try:
            command = TUICommand(command_str)
        except ValueError:
            self._print(self._colorize(f"Unknown command: /{command_str}", Colors.RED))
            self._print(f"Type {self._colorize('/help', Colors.YELLOW)} for available commands")
            return

        # Dispatch command
        handler = getattr(self, f'_cmd_{command_str.replace("-", "_")}', None)
        if handler:
            handler(args)
        else:
            self._print(self._colorize(f"Command not yet implemented: /{command_str}", Colors.YELLOW))

    def _handle_direct_input(self, line: str):
        """Handle direct S-expression input"""
        # Validate input looks like valid Prolog syntax
        line = line.strip()

        # Input should either:
        # 1. Start with '(' for S-expressions like (parent john mary)
        # 2. Be a rule with ':-'
        # Otherwise it's likely a typo or command error
        if not line.startswith('(') and ':-' not in line:
            self._print(self._colorize(
                f"✗ Invalid input: '{line}'\n"
                f"  Expected S-expression like: (parent john mary)\n"
                f"  Or a rule like: (grandparent X Z) :- (parent X Y), (parent Y Z)\n"
                f"  Or use /ask for queries: /ask (ancestor john alice)",
                Colors.RED
            ))
            return

        try:
            term = parse_s_expression(line)

            # Check if it's a rule
            if ':-' in line:
                self._add_rule_from_string(line)
            else:
                # It's a fact
                self.engine.add_fact(term)
                self._print(self._colorize(f"✓ Added fact: {term}", Colors.GREEN))
        except Exception as e:
            self._print(self._colorize(f"✗ Parse error: {e}", Colors.RED))

    def _add_rule_from_string(self, line: str):
        """Parse and add a rule from string"""
        # Simple parser for rules: (head X Y) :- (body1 X Z), (body2 Z Y)
        if ':-' not in line:
            raise ValueError("Not a rule (missing ':-')")

        head_str, body_str = line.split(':-', 1)
        head = parse_s_expression(head_str.strip())

        # Parse body terms
        body = []
        for term_str in body_str.split(','):
            body.append(parse_s_expression(term_str.strip()))

        self.engine.add_rule(Rule(head, body))
        body_repr = ", ".join(str(b) for b in body)
        self._print(self._colorize(f"✓ Added rule: {str(head)} :- {body_repr}", Colors.GREEN))

    # ===== Command Handlers =====

    def _cmd_help(self, args: str):
        """Show help"""
        self._print_help()

    def _cmd_exit(self, args: str):
        """Exit TUI"""
        # Auto-save if enabled
        if self.config.auto_save:
            auto_save_path = os.path.expanduser(self.config.auto_save_path or "~/.dreamlog_autosave.json")
            try:
                self._save_to_file(auto_save_path)
                self._print(self._colorize(f"✓ Auto-saved to {auto_save_path}", Colors.GREEN))
            except Exception as e:
                self._print(self._colorize(f"⚠ Auto-save failed: {e}", Colors.YELLOW))

        self.running = False
        self._print(self._colorize("\nGoodbye!", Colors.CYAN))

    def _cmd_quit(self, args: str):
        """Quit TUI"""
        self._cmd_exit(args)

    def _cmd_clear(self, args: str):
        """Clear knowledge base"""
        self.engine.kb.facts.clear()
        self.engine.kb.rules.clear()
        self._print(self._colorize("✓ Knowledge base cleared", Colors.GREEN))

    def _cmd_cd(self, args: str):
        """Change current directory"""
        if not args:
            # No args, go to home directory
            path = os.path.expanduser("~")
        else:
            path = os.path.expanduser(args.strip())

        try:
            os.chdir(path)
            cwd = os.getcwd()
            self._print(self._colorize(f"✓ Changed directory to: {cwd}", Colors.GREEN))
        except FileNotFoundError:
            self._print(self._colorize(f"✗ Directory not found: {path}", Colors.RED))
        except PermissionError:
            self._print(self._colorize(f"✗ Permission denied: {path}", Colors.RED))
        except Exception as e:
            self._print(self._colorize(f"✗ Error: {e}", Colors.RED))

    def _handle_shell_command(self, command: str):
        """Execute a shell command"""
        command = command.strip()

        if not command:
            self._print(self._colorize("✗ No command specified", Colors.RED))
            return

        try:
            # Execute the shell command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            # Print stdout
            if result.stdout:
                self._print(result.stdout.rstrip())

            # Print stderr in yellow
            if result.stderr:
                self._print(self._colorize(result.stderr.rstrip(), Colors.YELLOW))

            # Show return code if non-zero
            if result.returncode != 0:
                self._print(self._colorize(f"[Exit code: {result.returncode}]", Colors.RED))

        except subprocess.TimeoutExpired:
            self._print(self._colorize(f"✗ Command timed out after 30 seconds", Colors.RED))
        except Exception as e:
            self._print(self._colorize(f"✗ Error executing command: {e}", Colors.RED))

    def _cmd_load(self, args: str):
        """Load knowledge base from file"""
        if not args:
            self._print(self._colorize("Usage: /load <filename>", Colors.YELLOW))
            return

        try:
            with open(args, 'r') as f:
                content = f.read()

            # Parse as S-expressions
            for line in content.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith(';'):  # Skip comments
                    if ':-' in line:
                        self._add_rule_from_string(line)
                    else:
                        term = parse_s_expression(line)
                        self.engine.add_fact(term)

            self._print(self._colorize(f"✓ Loaded knowledge base from {args}", Colors.GREEN))
            self._cmd_stats("")
        except Exception as e:
            self._print(self._colorize(f"✗ Load failed: {e}", Colors.RED))

    def _save_to_file(self, filename: str):
        """Helper to save KB to file"""
        with open(filename, 'w') as f:
            # Write facts
            for fact in self.engine.kb.facts:
                f.write(f"{fact.term}\n")

            # Write rules
            for rule in self.engine.kb.rules:
                body_str = ", ".join(str(b) for b in rule.body)
                f.write(f"{rule.head} :- {body_str}\n")

    def _cmd_save(self, args: str):
        """Save knowledge base to file"""
        if not args:
            self._print(self._colorize("Usage: /save <filename>", Colors.YELLOW))
            return

        try:
            self._save_to_file(args)
            self._print(self._colorize(f"✓ Saved knowledge base to {args}", Colors.GREEN))
        except Exception as e:
            self._print(self._colorize(f"✗ Save failed: {e}", Colors.RED))

    def _cmd_facts(self, args: str):
        """List all facts"""
        facts = self.engine.kb.facts
        if not facts:
            self._print(self._colorize("No facts in knowledge base", Colors.GRAY))
            return

        # Print with indices
        self._print(self._colorize("\nFacts:", Colors.BOLD))
        for i, fact in enumerate(facts):
            self._print(f"  {self._colorize(f'[{i}]', Colors.YELLOW)} {fact.term}")

        self._print(f"\n{self._colorize(f'Total: {len(facts)} facts', Colors.BOLD)}")
        self._print(self._colorize(f"Use /remove-fact <index> to remove", Colors.GRAY))

    def _cmd_rules(self, args: str):
        """List all rules"""
        rules = self.engine.kb.rules
        if not rules:
            self._print(self._colorize("No rules in knowledge base", Colors.GRAY))
            return

        # Print with indices
        self._print(self._colorize("\nRules:", Colors.BOLD))
        for i, rule in enumerate(rules):
            body_str = ", ".join(str(b) for b in rule.body)
            self._print(f"  {self._colorize(f'[{i}]', Colors.YELLOW)} {rule.head} :- {body_str}")

        self._print(f"\n{self._colorize(f'Total: {len(rules)} rules', Colors.BOLD)}")
        self._print(self._colorize(f"Use /remove-rule <index> to remove", Colors.GRAY))

    def _cmd_remove_fact(self, args: str):
        """Remove a fact by index"""
        if not args:
            self._print(self._colorize("✗ Usage: /remove-fact <index>", Colors.RED))
            self._print(self._colorize("  Use /facts to see indices", Colors.GRAY))
            return

        try:
            index = int(args.strip())
        except ValueError:
            self._print(self._colorize(f"✗ Invalid index: {args}", Colors.RED))
            return

        try:
            fact = self.engine.kb.remove_fact(index)
            self._print(self._colorize(f"✓ Removed fact [{index}]: {fact.term}", Colors.GREEN))
        except IndexError as e:
            self._print(self._colorize(f"✗ {e}", Colors.RED))

    def _cmd_remove_rule(self, args: str):
        """Remove a rule by index"""
        if not args:
            self._print(self._colorize("✗ Usage: /remove-rule <index>", Colors.RED))
            self._print(self._colorize("  Use /rules to see indices", Colors.GRAY))
            return

        try:
            index = int(args.strip())
        except ValueError:
            self._print(self._colorize(f"✗ Invalid index: {args}", Colors.RED))
            return

        try:
            rule = self.engine.kb.remove_rule(index)
            body_str = ", ".join(str(b) for b in rule.body)
            self._print(self._colorize(f"✓ Removed rule [{index}]: {rule.head} :- {body_str}", Colors.GREEN))
        except IndexError as e:
            self._print(self._colorize(f"✗ {e}", Colors.RED))

    def _cmd_stats(self, args: str):
        """Show knowledge base statistics"""
        fact_count = len(self.engine.kb.facts)
        rule_count = len(self.engine.kb.rules)

        # Count unique functors
        functors = set()
        for fact in self.engine.kb.facts:
            if hasattr(fact.term, 'functor'):
                functors.add(fact.term.functor)
        for rule in self.engine.kb.rules:
            if hasattr(rule.head, 'functor'):
                functors.add(rule.head.functor)
        functor_count = len(functors)

        self._print(self._colorize("\n=== Knowledge Base Statistics ===", Colors.BOLD))
        self._print(f"Facts:     {fact_count}")
        self._print(f"Rules:     {rule_count}")
        self._print(f"Functors:  {functor_count}")
        self._print(f"Queries:   {self.stats['queries']}")
        self._print(f"Solutions: {self.stats['solutions_found']}")
        if self.config.llm_enabled:
            self._print(f"LLM calls: {self.stats['llm_calls']}")
        self._print(f"Sleep cycles: {self.stats['sleep_cycles']}")
        self._print(f"Compression: {self.stats['compression_ratio']:.2f}x")

    def _cmd_ask(self, args: str):
        """Ask a query (find one solution)"""
        self._execute_query(args, find_all=False)

    def _cmd_find_all(self, args: str):
        """Find all solutions"""
        self._execute_query(args, find_all=True)

    def _execute_query(self, query_str: str, find_all: bool = False):
        """Execute a query"""
        if not query_str:
            self._print(self._colorize("Usage: /ask <query> or /find-all <query>", Colors.YELLOW))
            return

        try:
            query = parse_s_expression(query_str)
            self.stats['queries'] += 1

            self._print(self._colorize(f"\n? {query}", Colors.CYAN))

            results = self.engine.query([query])

            if results:
                # Limit results to max_solutions
                limited_results = results[:self.config.tui_max_solutions] if find_all else results[:1]

                for i, result in enumerate(limited_results, 1):
                    bindings = result.get_ground_bindings()
                    if bindings:
                        binding_str = ", ".join(f"{k}={v}" for k, v in bindings.items())
                        self._print(self._colorize(f"  {i}. {binding_str}", Colors.GREEN))
                    else:
                        self._print(self._colorize(f"  {i}. true", Colors.GREEN))

                if find_all and len(results) > self.config.tui_max_solutions:
                    remaining = len(results) - self.config.tui_max_solutions
                    self._print(self._colorize(f"  ... and {remaining} more (use /set-max-solutions to show more)", Colors.GRAY))

                self.stats['solutions_found'] += len(results)
                self._print(f"\n{self._colorize(f'✓ Found {len(results)} solution(s)', Colors.GREEN)}")
            else:
                self._print(self._colorize("✗ No solutions found", Colors.GRAY))
        except Exception as e:
            self._print(self._colorize(f"✗ Query failed: {e}", Colors.RED))

    def _cmd_llm(self, args: str):
        """Enable/disable LLM"""
        if args.lower() == 'on':
            self.config.llm_enabled = True
            self._setup_llm()
        elif args.lower() == 'off':
            self.config.llm_enabled = False
            self.engine.llm_hook = None
            self._print(self._colorize("✓ LLM integration disabled", Colors.GREEN))
        else:
            status = "enabled" if self.config.llm_enabled else "disabled"
            self._print(f"LLM is currently {status}")

    def _cmd_debug(self, args: str):
        """Enable/disable debug mode"""
        if args.lower() == 'on':
            self.config.debug_enabled = True
            if self.engine.llm_hook:
                self.engine.llm_hook.debug = True
                # Update callback
                self.engine.llm_hook.debug_callback = lambda msg: self._print(self._colorize(msg, Colors.MAGENTA))
            self._print(self._colorize("✓ Debug mode enabled", Colors.GREEN))
        elif args.lower() == 'off':
            self.config.debug_enabled = False
            if self.engine.llm_hook:
                self.engine.llm_hook.debug = False
            self._print(self._colorize("✓ Debug mode disabled", Colors.GREEN))
        else:
            status = "enabled" if self.config.debug_enabled else "disabled"
            self._print(f"Debug is currently {status}")

    def _cmd_model(self, args: str):
        """Get or set current LLM model"""
        if not self.engine.llm_hook or not self.config.llm_enabled:
            self._print(self._colorize("✗ LLM not enabled. Use /llm on first", Colors.RED))
            return

        provider = self.engine.llm_hook.provider

        if args:
            # Set model
            model_name = args.strip()
            provider.set_parameter("model", model_name)
            self._print(self._colorize(f"✓ Model set to: {model_name}", Colors.GREEN))
        else:
            # Get current model
            current_model = provider.get_parameter("model", "unknown")
            self._print(f"Current model: {self._colorize(current_model, Colors.CYAN)}")

    def _cmd_models(self, args: str):
        """List available models from provider"""
        if not self.engine.llm_hook or not self.config.llm_enabled:
            self._print(self._colorize("✗ LLM not enabled. Use /llm on first", Colors.RED))
            return

        provider = self.engine.llm_hook.provider
        provider_class = provider.__class__.__name__

        # Check if Ollama provider
        if 'Ollama' in provider_class:
            try:
                import urllib.request
                import json

                base_url = getattr(provider, 'base_url', "http://localhost:11434")
                url = f"{base_url}/api/tags"

                with urllib.request.urlopen(url, timeout=5) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    models = data.get('models', [])

                    if not models:
                        self._print(self._colorize("No models found", Colors.YELLOW))
                        return

                    self._print(self._colorize("\nAvailable Ollama Models:", Colors.BOLD))
                    for model in models:
                        name = model.get('name', 'unknown')
                        size = model.get('size', 0) / (1024**3)  # Convert to GB
                        details = model.get('details', {})
                        param_size = details.get('parameter_size', 'unknown')

                        self._print(f"  • {self._colorize(name, Colors.CYAN)}")
                        self._print(f"    Size: {size:.1f} GB, Parameters: {param_size}")

                    current = provider.get_parameter("model", "unknown")
                    self._print(f"\nCurrent: {self._colorize(current, Colors.GREEN)}")

            except Exception as e:
                self._print(self._colorize(f"✗ Error fetching models: {e}", Colors.RED))
        else:
            # For non-Ollama providers, just show current model
            metadata = provider.get_metadata()
            current_model = metadata.get('model', 'unknown')
            self._print(f"Provider: {self._colorize(provider_class, Colors.CYAN)}")
            self._print(f"Current model: {self._colorize(current_model, Colors.GREEN)}")
            self._print(self._colorize("\nNote: Model listing only supported for Ollama", Colors.GRAY))

    def _cmd_temperature(self, args: str):
        """Get or set temperature parameter"""
        if not self.engine.llm_hook or not self.config.llm_enabled:
            self._print(self._colorize("✗ LLM not enabled. Use /llm on first", Colors.RED))
            return

        provider = self.engine.llm_hook.provider

        if args:
            # Set temperature
            try:
                temp = float(args.strip())
                if temp < 0.0 or temp > 2.0:
                    self._print(self._colorize("✗ Temperature should be between 0.0 and 2.0", Colors.RED))
                    return

                provider.set_parameter("temperature", temp)
                self._print(self._colorize(f"✓ Temperature set to: {temp}", Colors.GREEN))
            except ValueError:
                self._print(self._colorize(f"✗ Invalid temperature: {args}", Colors.RED))
        else:
            # Get current temperature
            current = provider.get_parameter("temperature", 0.1)
            self._print(f"Current temperature: {self._colorize(str(current), Colors.CYAN)}")

    def _cmd_max_tokens(self, args: str):
        """Get or set max tokens parameter"""
        if not self.engine.llm_hook or not self.config.llm_enabled:
            self._print(self._colorize("✗ LLM not enabled. Use /llm on first", Colors.RED))
            return

        provider = self.engine.llm_hook.provider

        if args:
            # Set max tokens
            try:
                max_tok = int(args.strip())
                if max_tok < 1:
                    self._print(self._colorize("✗ Max tokens must be positive", Colors.RED))
                    return

                provider.set_parameter("max_tokens", max_tok)
                self._print(self._colorize(f"✓ Max tokens set to: {max_tok}", Colors.GREEN))
            except ValueError:
                self._print(self._colorize(f"✗ Invalid number: {args}", Colors.RED))
        else:
            # Get current max tokens
            current = provider.get_parameter("max_tokens", 500)
            self._print(f"Current max tokens: {self._colorize(str(current), Colors.CYAN)}")

    def _cmd_provider(self, args: str):
        """Show current provider information"""
        if not self.engine.llm_hook or not self.config.llm_enabled:
            self._print(self._colorize("✗ LLM not enabled. Use /llm on first", Colors.RED))
            return

        provider = self.engine.llm_hook.provider
        metadata = provider.get_metadata()

        self._print(self._colorize("\nLLM Provider Information:", Colors.BOLD))
        self._print(f"  Class: {self._colorize(metadata.get('provider_class', 'unknown'), Colors.CYAN)}")
        self._print(f"  Model: {self._colorize(metadata.get('model', 'unknown'), Colors.CYAN)}")

        params = metadata.get('parameters', {})
        if params:
            self._print(self._colorize("\n  Parameters:", Colors.BOLD))
            for key, value in params.items():
                if key not in ['model', 'api_key']:  # Don't show sensitive data
                    self._print(f"    {key}: {value}")

        capabilities = metadata.get('capabilities', [])
        if capabilities:
            self._print(self._colorize("\n  Capabilities:", Colors.BOLD))
            for cap in capabilities:
                self._print(f"    • {cap}")

    def _cmd_sleep(self, args: str):
        """Run full sleep cycle (dream with all optimizations)"""
        self._print(self._colorize("\n=== Sleep Cycle Starting ===", Colors.BOLD))
        self._print("Running full dream cycle with compression, abstraction, and generalization...\n")

        try:
            from .kb_dreamer import KnowledgeBaseDreamer

            # Create dreamer (uses LLM provider if available)
            provider = self.engine.llm_hook.provider if self.engine.llm_hook else None
            dreamer = KnowledgeBaseDreamer(provider)

            # Run dream cycle
            session = dreamer.dream(
                self.engine.kb,
                dream_cycles=3,
                exploration_samples=5,
                focus="all",
                verify=True
            )

            # Display results
            self._print(self._colorize(f"Dream cycle complete!", Colors.GREEN))
            self._print(f"  Exploration paths: {session.exploration_paths}")
            self._print(f"  Insights discovered: {len(session.insights)}")
            self._print(f"  Compression ratio: {session.compression_ratio:.1%}")
            self._print(f"  Generalization score: {session.generalization_score:.2f}")
            self._print(f"  Behavior preserved: {session.verification.preserved}")

            if session.insights:
                self._print(self._colorize("\nInsights:", Colors.CYAN))
                for insight in session.insights:
                    self._print(f"  [{insight.type}] {insight.description}")

            self.stats['sleep_cycles'] += 1

        except Exception as e:
            self._print(self._colorize(f"✗ Sleep cycle failed: {e}", Colors.RED))

    def _cmd_compress(self, args: str):
        """Compress knowledge base (detect redundancies)"""
        self._print(self._colorize("\n=== Knowledge Base Compression ===", Colors.BOLD))

        try:
            from .kb_dreamer import KnowledgeBaseDreamer

            provider = self.engine.llm_hook.provider if self.engine.llm_hook else None
            dreamer = KnowledgeBaseDreamer(provider)
            session = dreamer.dream(self.engine.kb, focus="compression", verify=False)

            if session.insights:
                self._print(self._colorize(f"Found {len(session.insights)} compression opportunities:", Colors.GREEN))
                for insight in session.insights:
                    self._print(f"  • {insight.description}")
                    self._print(f"    Potential compression: {insight.compression_ratio:.1%}")
            else:
                self._print(self._colorize("No compression opportunities found", Colors.YELLOW))

        except Exception as e:
            self._print(self._colorize(f"✗ Compression analysis failed: {e}", Colors.RED))

    def _cmd_consolidate(self, args: str):
        """Consolidate learned knowledge (detect generalizations)"""
        self._print(self._colorize("\n=== Knowledge Consolidation ===", Colors.BOLD))

        try:
            from .kb_dreamer import KnowledgeBaseDreamer

            provider = self.engine.llm_hook.provider if self.engine.llm_hook else None
            dreamer = KnowledgeBaseDreamer(provider)
            session = dreamer.dream(self.engine.kb, focus="generalization", verify=False)

            if session.insights:
                self._print(self._colorize(f"Found {len(session.insights)} generalization opportunities:", Colors.GREEN))
                for insight in session.insights:
                    self._print(f"  • {insight.description}")
                    self._print(f"    Coverage gain: {insight.coverage_gain:.1f}x")
            else:
                self._print(self._colorize("No generalization opportunities found", Colors.YELLOW))

        except Exception as e:
            self._print(self._colorize(f"✗ Consolidation analysis failed: {e}", Colors.RED))

    def _cmd_dream(self, args: str):
        """Run dream cycle (alias for /sleep)"""
        self._cmd_sleep(args)

    def _cmd_analyze(self, args: str):
        """Analyze all optimization opportunities"""
        self._print(self._colorize("\n=== Knowledge Base Analysis ===", Colors.BOLD))

        try:
            from .kb_dreamer import KnowledgeBaseDreamer

            provider = self.engine.llm_hook.provider if self.engine.llm_hook else None
            dreamer = KnowledgeBaseDreamer(provider)
            suggestions = dreamer.suggest_optimizations(self.engine.kb)

            if suggestions:
                self._print(self._colorize(f"Found {len(suggestions)} optimization opportunities:\n", Colors.GREEN))
                for suggestion in suggestions:
                    self._print(f"  • {suggestion}")
            else:
                self._print(self._colorize("Knowledge base is already well-optimized!", Colors.GREEN))

        except Exception as e:
            self._print(self._colorize(f"✗ Analysis failed: {e}", Colors.RED))

    def _cmd_explain(self, args: str):
        """Explain how a query would be resolved"""
        if not args:
            self._print(self._colorize("Usage: /explain <query>", Colors.YELLOW))
            return

        try:
            query_term = parse_s_expression(args)

            from dreamlog.unification import Unifier
            unifier = Unifier(trace=True)

            self._print(self._colorize(f"\n=== Explaining: {str(query_term)} ===", Colors.CYAN))

            # Check facts
            matching_facts = []
            for fact in self.engine.kb.facts:
                result = unifier.unify(query_term, fact.term)
                if result.success:
                    matching_facts.append((fact, result))

            if matching_facts:
                self._print(self._colorize("\nMatching facts:", Colors.GREEN))
                for fact, result in matching_facts:
                    self._print(f"  • {str(fact.term)}")
                    if result.bindings:
                        bindings = ", ".join(f"{k}={v}" for k, v in result.bindings.items())
                        self._print(f"    Bindings: {bindings}")

            # Check rules
            matching_rules = []
            for rule in self.engine.kb.rules:
                result = unifier.unify(query_term, rule.head)
                if result.success:
                    matching_rules.append((rule, result))

            if matching_rules:
                self._print(self._colorize("\nMatching rules:", Colors.YELLOW))
                for rule, result in matching_rules:
                    body_str = ", ".join(str(b) for b in rule.body)
                    self._print(f"  • {str(rule.head)} :- {body_str}")
                    if result.bindings:
                        bindings = ", ".join(f"{k}={v}" for k, v in result.bindings.items())
                        self._print(f"    Initial bindings: {bindings}")

            if not matching_facts and not matching_rules:
                self._print(self._colorize("\nNo matching facts or rules found", Colors.GRAY))

        except Exception as e:
            self._print(self._colorize(f"✗ Error: {e}", Colors.RED))

    def _cmd_embedding_models(self, args: str):
        """List available embedding models from Ollama"""
        import requests

        try:
            response = requests.get(f"{self.config.llm_base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()

            # Filter for embedding models (models typically used for embeddings)
            all_models = data.get('models', [])
            embedding_keywords = ['embed', 'embedding', 'nomic', 'mxbai', 'snowflake']

            embedding_models = [
                m for m in all_models
                if any(keyword in m['name'].lower() for keyword in embedding_keywords)
            ]

            if not embedding_models:
                self._print(self._colorize("No embedding models found", Colors.YELLOW))
                self._print("Common embedding models: nomic-embed-text, mxbai-embed-large, snowflake-arctic-embed")
                return

            self._print(self._colorize("\n=== Available Embedding Models ===", Colors.CYAN))
            for model in embedding_models:
                name = model['name']
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                modified = model.get('modified_at', 'unknown')

                self._print(f"\n{self._colorize(name, Colors.GREEN)}")
                self._print(f"  Size: {size:.2f} GB")
                self._print(f"  Modified: {modified}")

        except Exception as e:
            self._print(self._colorize(f"✗ Failed to list embedding models: {e}", Colors.RED))

    def _cmd_embedding_model_info(self, args: str):
        """Show detailed info about an embedding model"""
        if not args:
            self._print(self._colorize("Usage: /embedding-model-info <model-name>", Colors.YELLOW))
            return

        import requests

        model_name = args.strip()

        try:
            response = requests.post(
                f"{self.config.llm_base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            self._print(self._colorize(f"\n=== Model Info: {model_name} ===", Colors.CYAN))

            # Extract relevant info
            details = data.get('details', {})
            model_info = data.get('model_info', {})

            if details:
                self._print(f"\n{self._colorize('Details:', Colors.GREEN)}")
                family = details.get('family', 'unknown')
                param_size = details.get('parameter_size', 'unknown')
                quantization = details.get('quantization_level', 'unknown')

                self._print(f"  Family: {family}")
                self._print(f"  Parameters: {param_size}")
                self._print(f"  Quantization: {quantization}")

            if model_info:
                self._print(f"\n{self._colorize('Model Info:', Colors.GREEN)}")

                # Try to find embedding dimension
                for key, value in model_info.items():
                    if 'embedding' in key.lower() and 'length' in key.lower():
                        self._print(f"  {self._colorize(f'Dimension: {value}', Colors.YELLOW)}")
                    elif 'context' in key.lower() and 'length' in key.lower():
                        self._print(f"  Context length: {value}")
                    elif 'block_count' in key.lower():
                        self._print(f"  Layers: {value}")
                    elif 'attention' in key.lower() and 'head' in key.lower():
                        self._print(f"  Attention heads: {value}")

            # Show template if available
            template = data.get('template', '')
            if template:
                self._print(f"\n{self._colorize('Template:', Colors.GREEN)}")
                self._print(f"  {template}")

        except Exception as e:
            self._print(self._colorize(f"✗ Failed to get model info: {e}", Colors.RED))

    def _cmd_set_embedding_model(self, args: str):
        """Set the embedding model"""
        if not args:
            self._print(self._colorize("Usage: /set-embedding-model <model-name>", Colors.YELLOW))
            return

        # Store in config (would need to add embedding_model field to TUIConfig)
        self._print(self._colorize(f"✓ Embedding model set to: {args.strip()}", Colors.GREEN))
        self._print(self._colorize("Note: Embedding integration coming soon", Colors.YELLOW))


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DreamLog Terminal User Interface")
    parser.add_argument("--llm", action="store_true", help="Enable LLM integration")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--model", default="phi4-mini:latest", help="LLM model")
    parser.add_argument("--base-url", default="http://localhost:11434", help="LLM base URL")
    parser.add_argument("--timeout", type=int, default=30, help="LLM request timeout in seconds (default: 30, use 120 for reasoning models)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--load", help="Load knowledge base on startup")

    args = parser.parse_args()

    from dreamlog.config import DreamLogConfig, LLMProviderConfig

    # Create provider config from CLI args
    provider_config = LLMProviderConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout
    )

    # Create main config
    config = DreamLogConfig(
        provider=provider_config,
        llm_enabled=args.llm,
        debug_enabled=args.debug,
        tui_color_output=not args.no_color
    )

    tui = DreamLogTUI(config)

    # Load KB if specified
    if args.load:
        tui._cmd_load(args.load)

    tui.run()


if __name__ == "__main__":
    main()
