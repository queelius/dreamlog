"""
DreamLog Jupyter Magic Commands

Provides IPython/Jupyter magic commands for interactive DreamLog development.
Install with: %load_ext dreamlog_magic

Available magics:
    %dreamlog_init - Initialize DreamLog engine
    %dreamlog_fact - Add a fact
    %dreamlog_rule - Add a rule
    %dreamlog_query - Execute a query
    %%dreamlog - Multi-line DreamLog code cell
    %dreamlog_save - Save knowledge base
    %dreamlog_load - Load knowledge base
    %dreamlog_clear - Clear knowledge base
    %dreamlog_stats - Show KB statistics
    %dreamlog_visualize - Visualize KB as graph

Usage example:
    %load_ext dreamlog_magic
    %dreamlog_init
    %dreamlog_fact (parent john mary)
    %dreamlog_fact (parent mary alice)
    %dreamlog_rule (grandparent X Z) :- (parent X Y), (parent Y Z)
    %dreamlog_query (grandparent john Z)
"""

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic
from IPython.display import display, HTML, Javascript
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
import json
import sys
import os
from typing import Optional, Dict, Any
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dreamlog import (
    DreamLogEngine, parse_prefix_notation, parse_s_expression,
    Fact, Rule, atom, var, compound
)
from dreamlog.prefix_parser import term_to_sexp, term_to_prefix_json
from dreamlog.llm_hook import LLMHook
from dreamlog.llm_providers import MockLLMProvider


@magics_class
class DreamLogMagics(Magics):
    """
    IPython magic commands for DreamLog
    """
    
    def __init__(self, shell):
        super().__init__(shell)
        self.engine = None
        self.query_history = []
        self.visualization_enabled = False
        
        # Try to import optional dependencies
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            self.visualization_enabled = True
        except ImportError:
            pass
    
    @line_magic
    @magic_arguments()
    @argument('--llm', action='store_true', help='Enable LLM integration')
    @argument('--domain', default='general', help='LLM knowledge domain')
    def dreamlog_init(self, line):
        """Initialize DreamLog engine"""
        args = parse_argstring(self.dreamlog_init, line)
        
        self.engine = DreamLogEngine()
        
        if args.llm:
            llm = MockLLMProvider(knowledge_domain=args.domain)
            self.engine.llm_hook = LLMHook(llm)
            print(f"✓ DreamLog engine initialized with LLM support (domain: {args.domain})")
        else:
            print("✓ DreamLog engine initialized")
        
        return self.engine
    
    @line_magic
    def dreamlog_fact(self, line):
        """Add a fact to the knowledge base"""
        if not self.engine:
            print("⚠ Initialize DreamLog first with %dreamlog_init")
            return
        
        try:
            # Parse fact
            if line.startswith("("):
                fact_term = parse_s_expression(line)
            else:
                fact_term = parse_prefix_notation(json.loads(line))
            
            self.engine.add_fact(Fact(fact_term))
            print(f"✓ Added fact: {term_to_sexp(fact_term)}")
            
        except Exception as e:
            print(f"✗ Error adding fact: {e}")
    
    @line_magic
    @magic_arguments()
    @argument('head', help='Rule head')
    @argument('body', nargs='+', help='Rule body conditions')
    def dreamlog_rule(self, line):
        """Add a rule to the knowledge base"""
        if not self.engine:
            print("⚠ Initialize DreamLog first with %dreamlog_init")
            return
        
        args = parse_argstring(self.dreamlog_rule, line)
        
        try:
            # Parse head
            if args.head.startswith("("):
                head = parse_s_expression(args.head)
            else:
                head = parse_prefix_notation(json.loads(args.head))
            
            # Parse body
            body = []
            for term in args.body:
                if term == ":-":
                    continue
                if term.startswith("("):
                    body.append(parse_s_expression(term))
                else:
                    body.append(parse_prefix_notation(json.loads(term)))
            
            self.engine.add_rule(Rule(head, body))
            
            body_str = ", ".join(term_to_sexp(b) for b in body)
            print(f"✓ Added rule: {term_to_sexp(head)} :- {body_str}")
            
        except Exception as e:
            print(f"✗ Error adding rule: {e}")
    
    @line_magic
    @magic_arguments()
    @argument('query', help='Query to execute')
    @argument('--limit', type=int, default=10, help='Maximum solutions')
    @argument('--dataframe', action='store_true', help='Return as pandas DataFrame')
    @argument('--trace', action='store_true', help='Show unification trace')
    def dreamlog_query(self, line):
        """Execute a query against the knowledge base"""
        if not self.engine:
            print("⚠ Initialize DreamLog first with %dreamlog_init")
            return
        
        args = parse_argstring(self.dreamlog_query, line)
        
        try:
            # Parse query
            if args.query.startswith("("):
                query_term = parse_s_expression(args.query)
            else:
                query_term = parse_prefix_notation(json.loads(args.query))
            
            # Add to history
            self.query_history.append(args.query)
            
            # Execute query
            solutions = []
            for i, solution in enumerate(self.engine.query([query_term])):
                if i >= args.limit:
                    break
                
                sol_dict = {
                    'index': i + 1,
                    **solution.ground_bindings
                }
                solutions.append(sol_dict)
            
            if args.dataframe and solutions:
                # Return as DataFrame
                df = pd.DataFrame(solutions)
                display(df)
                return df
            else:
                # Display solutions
                if solutions:
                    print(f"Found {len(solutions)} solution(s):")
                    for sol in solutions:
                        bindings = ", ".join(f"{k}={v}" for k, v in sol.items() if k != 'index')
                        print(f"  {sol['index']}. {bindings if bindings else 'Yes'}")
                else:
                    print("No solutions found.")
                
                return solutions
            
        except Exception as e:
            print(f"✗ Query error: {e}")
    
    @cell_magic
    @magic_arguments()
    @argument('--format', choices=['sexp', 'json'], default='sexp', 
              help='Input format')
    def dreamlog(self, line, cell):
        """
        Execute DreamLog code from a cell
        
        Format:
            fact: (parent john mary)
            fact: (parent mary alice)
            rule: (grandparent X Z) :- (parent X Y), (parent Y Z)
            query: (grandparent john Z)
        """
        if not self.engine:
            print("⚠ Initialize DreamLog first with %dreamlog_init")
            return
        
        args = parse_argstring(self.dreamlog, line)
        
        results = []
        for line in cell.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('fact:'):
                fact_str = line[5:].strip()
                try:
                    if args.format == 'sexp':
                        fact_term = parse_s_expression(fact_str)
                    else:
                        fact_term = parse_prefix_notation(json.loads(fact_str))
                    
                    self.engine.add_fact(Fact(fact_term))
                    results.append(f"✓ Added fact: {term_to_sexp(fact_term)}")
                except Exception as e:
                    results.append(f"✗ Error in fact: {e}")
            
            elif line.startswith('rule:'):
                rule_str = line[5:].strip()
                try:
                    # Parse rule (head :- body1, body2, ...)
                    parts = rule_str.split(':-')
                    if len(parts) != 2:
                        raise ValueError("Rule must have format: head :- body")
                    
                    head_str = parts[0].strip()
                    body_str = parts[1].strip()
                    
                    if args.format == 'sexp':
                        head = parse_s_expression(head_str)
                        body = []
                        for term in body_str.split(','):
                            body.append(parse_s_expression(term.strip()))
                    else:
                        head = parse_prefix_notation(json.loads(head_str))
                        body_parts = json.loads(f"[{body_str}]")
                        body = [parse_prefix_notation(b) for b in body_parts]
                    
                    self.engine.add_rule(Rule(head, body))
                    results.append(f"✓ Added rule: {term_to_sexp(head)} :- {', '.join(term_to_sexp(b) for b in body)}")
                except Exception as e:
                    results.append(f"✗ Error in rule: {e}")
            
            elif line.startswith('query:'):
                query_str = line[6:].strip()
                try:
                    if args.format == 'sexp':
                        query_term = parse_s_expression(query_str)
                    else:
                        query_term = parse_prefix_notation(json.loads(query_str))
                    
                    # Execute query
                    solutions = list(self.engine.query([query_term]))[:5]
                    
                    if solutions:
                        sol_strs = []
                        for sol in solutions:
                            if sol.ground_bindings:
                                bindings = ", ".join(f"{k}={v}" for k, v in sol.ground_bindings.items())
                                sol_strs.append(bindings)
                            else:
                                sol_strs.append("Yes")
                        results.append(f"✓ Query {term_to_sexp(query_term)}: {'; '.join(sol_strs)}")
                    else:
                        results.append(f"✓ Query {term_to_sexp(query_term)}: No")
                except Exception as e:
                    results.append(f"✗ Error in query: {e}")
        
        # Display results
        for result in results:
            print(result)
    
    @line_magic
    def dreamlog_save(self, line):
        """Save knowledge base to file"""
        if not self.engine:
            print("⚠ Initialize DreamLog first with %dreamlog_init")
            return
        
        filename = line.strip() or "knowledge_base.json"
        
        try:
            kb_data = self.engine.save_to_prefix()
            with open(filename, 'w') as f:
                f.write(kb_data)
            print(f"✓ Saved knowledge base to {filename}")
        except Exception as e:
            print(f"✗ Error saving: {e}")
    
    @line_magic
    def dreamlog_load(self, line):
        """Load knowledge base from file"""
        if not self.engine:
            print("⚠ Initialize DreamLog first with %dreamlog_init")
            return
        
        filename = line.strip()
        if not filename:
            print("⚠ Please specify a filename")
            return
        
        try:
            with open(filename, 'r') as f:
                kb_data = f.read()
            self.engine.load_from_prefix(kb_data)
            print(f"✓ Loaded knowledge base from {filename}")
            self.dreamlog_stats("")
        except Exception as e:
            print(f"✗ Error loading: {e}")
    
    @line_magic
    def dreamlog_clear(self, line):
        """Clear the knowledge base"""
        if not self.engine:
            print("⚠ Initialize DreamLog first with %dreamlog_init")
            return
        
        self.engine.clear()
        print("✓ Knowledge base cleared")
    
    @line_magic
    def dreamlog_stats(self, line):
        """Show knowledge base statistics"""
        if not self.engine:
            print("⚠ Initialize DreamLog first with %dreamlog_init")
            return
        
        num_facts = len(self.engine.kb.facts)
        num_rules = len(self.engine.kb.rules)
        
        # Get unique functors
        functors = set()
        for fact in self.engine.kb.facts:
            if hasattr(fact.term, 'functor'):
                functors.add(fact.term.functor)
        for rule in self.engine.kb.rules:
            if hasattr(rule.head, 'functor'):
                functors.add(rule.head.functor)
        
        print("📊 Knowledge Base Statistics:")
        print(f"  Facts: {num_facts}")
        print(f"  Rules: {num_rules}")
        print(f"  Functors: {len(functors)}")
        if functors:
            print(f"    {', '.join(sorted(functors))}")
        print(f"  Query history: {len(self.query_history)} queries")
    
    @line_magic
    @magic_arguments()
    @argument('--layout', choices=['spring', 'circular', 'random'], 
              default='spring', help='Graph layout')
    @argument('--size', type=int, default=10, help='Figure size')
    def dreamlog_visualize(self, line):
        """Visualize knowledge base as a graph"""
        if not self.engine:
            print("⚠ Initialize DreamLog first with %dreamlog_init")
            return
        
        if not self.visualization_enabled:
            print("⚠ Install networkx and matplotlib for visualization:")
            print("  pip install networkx matplotlib")
            return
        
        args = parse_argstring(self.dreamlog_visualize, line)
        
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add facts as edges
        for fact in self.engine.kb.facts:
            if hasattr(fact.term, 'functor') and hasattr(fact.term, 'args'):
                if len(fact.term.args) == 2:
                    # Binary relation - add as edge
                    G.add_edge(str(fact.term.args[0]), str(fact.term.args[1]),
                              label=fact.term.functor)
                else:
                    # Non-binary - add as node with label
                    G.add_node(str(fact.term), type='fact')
        
        # Add rules as special nodes
        for i, rule in enumerate(self.engine.kb.rules):
            rule_label = f"Rule{i}: {term_to_sexp(rule.head)}"
            G.add_node(rule_label, type='rule')
        
        # Draw graph
        plt.figure(figsize=(args.size, args.size))
        
        # Layout
        if args.layout == 'spring':
            pos = nx.spring_layout(G)
        elif args.layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # Draw nodes
        fact_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'fact']
        rule_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'rule']
        other_nodes = [n for n in G.nodes() if n not in fact_nodes and n not in rule_nodes]
        
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, 
                              node_color='lightblue', node_size=500)
        nx.draw_networkx_nodes(G, pos, nodelist=fact_nodes,
                              node_color='lightgreen', node_size=300)
        nx.draw_networkx_nodes(G, pos, nodelist=rule_nodes,
                              node_color='lightyellow', node_size=400)
        
        # Draw edges and labels
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        plt.title("DreamLog Knowledge Base Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @line_magic
    def dreamlog_history(self, line):
        """Show query history"""
        if not self.query_history:
            print("No queries in history")
            return
        
        print("📜 Query History:")
        for i, query in enumerate(self.query_history, 1):
            print(f"  {i}. {query}")


def load_ipython_extension(ipython):
    """Load the extension in IPython"""
    ipython.register_magics(DreamLogMagics)
    print("✓ DreamLog magic commands loaded. Use %dreamlog_init to start.")


def unload_ipython_extension(ipython):
    """Unload the extension"""
    pass