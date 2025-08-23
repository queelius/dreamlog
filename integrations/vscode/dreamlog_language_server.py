"""
DreamLog Language Server Protocol (LSP) Implementation

Provides IDE support for DreamLog files through the Language Server Protocol.
Features:
- Syntax highlighting and validation
- Auto-completion for functors and variables
- Hover information
- Go to definition
- Find references
- Code formatting
- Diagnostics (errors and warnings)

Installation:
1. Install this as a Python package
2. Configure VS Code to use this language server for .dreamlog files
"""

import asyncio
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

# LSP imports (requires pygls)
try:
    from pygls.server import LanguageServer
    from pygls.lsp.methods import (
        COMPLETION, TEXT_DOCUMENT_DID_CHANGE, TEXT_DOCUMENT_DID_OPEN,
        TEXT_DOCUMENT_DID_SAVE, HOVER, DEFINITION, REFERENCES,
        FORMATTING, WORKSPACE_DID_CHANGE_CONFIGURATION, INITIALIZE
    )
    from pygls.lsp.types import (
        CompletionItem, CompletionList, CompletionParams,
        DidChangeTextDocumentParams, DidOpenTextDocumentParams,
        DidSaveTextDocumentParams, HoverParams, Hover,
        DefinitionParams, Location, Range, Position,
        ReferencesParams, DocumentFormattingParams,
        TextEdit, Diagnostic, DiagnosticSeverity,
        MarkupContent, MarkupKind, CompletionItemKind,
        InitializeParams, InitializeResult
    )
except ImportError:
    print("Please install pygls: pip install pygls")
    sys.exit(1)

# Add parent directory for DreamLog imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dreamlog import (
    DreamLogEngine, parse_prefix_notation, parse_s_expression,
    Fact, Rule, atom, var, compound
)
from dreamlog.prefix_parser import term_to_sexp, term_to_prefix_json


class TokenType(Enum):
    """Token types for syntax highlighting"""
    FUNCTOR = "functor"
    VARIABLE = "variable"
    ATOM = "atom"
    OPERATOR = "operator"
    COMMENT = "comment"
    KEYWORD = "keyword"
    NUMBER = "number"
    STRING = "string"
    PARENTHESIS = "parenthesis"
    BRACKET = "bracket"


@dataclass
class Token:
    """Represents a token in the source code"""
    type: TokenType
    value: str
    line: int
    column: int
    length: int


@dataclass
class Symbol:
    """Represents a symbol (functor, variable, etc.) in the code"""
    name: str
    type: str  # "functor", "variable", "rule", "fact"
    line: int
    column: int
    arity: Optional[int] = None
    definition_line: Optional[int] = None


class DreamLogLanguageServer(LanguageServer):
    """
    Language Server for DreamLog files
    """
    
    def __init__(self):
        super().__init__("dreamlog-language-server", "v1.0.0")
        self.engine = DreamLogEngine()
        self.symbols: Dict[str, List[Symbol]] = {}  # file -> symbols
        self.diagnostics_cache: Dict[str, List[Diagnostic]] = {}
        
        # Configuration
        self.config = {
            "format": "sexp",  # or "json"
            "enableLLM": False,
            "maxCompletions": 20,
            "enableDiagnostics": True,
            "enableFormatting": True
        }
    
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize DreamLog source code"""
        tokens = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            col = 0
            
            # Skip comments
            if line.strip().startswith('%') or line.strip().startswith('#'):
                tokens.append(Token(
                    TokenType.COMMENT, line, line_num, 0, len(line)
                ))
                continue
            
            # Tokenize based on format
            if self.config["format"] == "sexp":
                tokens.extend(self._tokenize_sexp_line(line, line_num))
            else:
                tokens.extend(self._tokenize_json_line(line, line_num))
        
        return tokens
    
    def _tokenize_sexp_line(self, line: str, line_num: int) -> List[Token]:
        """Tokenize S-expression format line"""
        tokens = []
        i = 0
        
        while i < len(line):
            char = line[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Parentheses
            if char in '()':
                tokens.append(Token(
                    TokenType.PARENTHESIS, char, line_num, i, 1
                ))
                i += 1
            
            # Operators
            elif line[i:i+2] == ':-':
                tokens.append(Token(
                    TokenType.OPERATOR, ':-', line_num, i, 2
                ))
                i += 2
            
            # Variables (uppercase start)
            elif char.isupper():
                j = i
                while j < len(line) and (line[j].isalnum() or line[j] == '_'):
                    j += 1
                tokens.append(Token(
                    TokenType.VARIABLE, line[i:j], line_num, i, j - i
                ))
                i = j
            
            # Atoms/functors (lowercase start)
            elif char.islower():
                j = i
                while j < len(line) and (line[j].isalnum() or line[j] == '_'):
                    j += 1
                
                # Check if it's a functor (followed by parenthesis)
                if j < len(line) and line[j:j+1] in ' \t':
                    # Skip whitespace
                    k = j
                    while k < len(line) and line[k] in ' \t':
                        k += 1
                    if k < len(line) and line[k] == '(':
                        token_type = TokenType.FUNCTOR
                    else:
                        token_type = TokenType.ATOM
                elif j < len(line) and line[j] == '(':
                    token_type = TokenType.FUNCTOR
                else:
                    token_type = TokenType.ATOM
                
                tokens.append(Token(
                    token_type, line[i:j], line_num, i, j - i
                ))
                i = j
            
            # Numbers
            elif char.isdigit() or (char == '-' and i + 1 < len(line) and line[i + 1].isdigit()):
                j = i + 1 if char == '-' else i
                while j < len(line) and (line[j].isdigit() or line[j] == '.'):
                    j += 1
                tokens.append(Token(
                    TokenType.NUMBER, line[i:j], line_num, i, j - i
                ))
                i = j
            
            # Strings
            elif char == '"':
                j = i + 1
                while j < len(line) and line[j] != '"':
                    if line[j] == '\\' and j + 1 < len(line):
                        j += 2
                    else:
                        j += 1
                if j < len(line):
                    j += 1
                tokens.append(Token(
                    TokenType.STRING, line[i:j], line_num, i, j - i
                ))
                i = j
            
            else:
                i += 1
        
        return tokens
    
    def _tokenize_json_line(self, line: str, line_num: int) -> List[Token]:
        """Tokenize JSON format line"""
        tokens = []
        # Simplified JSON tokenization
        if line.strip().startswith('['):
            tokens.append(Token(
                TokenType.BRACKET, '[', line_num, line.index('['), 1
            ))
        if line.strip().endswith(']'):
            tokens.append(Token(
                TokenType.BRACKET, ']', line_num, line.rindex(']'), 1
            ))
        return tokens
    
    def extract_symbols(self, text: str, uri: str) -> List[Symbol]:
        """Extract symbols (functors, variables, etc.) from source"""
        symbols = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            
            # Parse based on format
            try:
                if self.config["format"] == "sexp":
                    if ":-" in line:
                        # It's a rule
                        parts = line.split(":-")
                        head = parts[0].strip()
                        if head.startswith("("):
                            term = parse_s_expression(head)
                            if hasattr(term, 'functor'):
                                symbols.append(Symbol(
                                    term.functor, "rule", line_num, 0,
                                    len(term.args) if hasattr(term, 'args') else 0
                                ))
                    elif line.startswith("("):
                        # It's a fact
                        term = parse_s_expression(line)
                        if hasattr(term, 'functor'):
                            symbols.append(Symbol(
                                term.functor, "fact", line_num, 0,
                                len(term.args) if hasattr(term, 'args') else 0
                            ))
                else:
                    # JSON format
                    if line.startswith("["):
                        data = json.loads(line)
                        if isinstance(data, list) and len(data) > 0:
                            if data[0] == "rule":
                                # Rule in JSON
                                if len(data) > 1 and isinstance(data[1], list):
                                    functor = data[1][0] if data[1] else None
                                    if functor:
                                        symbols.append(Symbol(
                                            functor, "rule", line_num, 0,
                                            len(data[1]) - 1
                                        ))
                            elif data[0] == "fact":
                                # Fact in JSON
                                if len(data) > 1 and isinstance(data[1], list):
                                    functor = data[1][0] if data[1] else None
                                    if functor:
                                        symbols.append(Symbol(
                                            functor, "fact", line_num, 0,
                                            len(data[1]) - 1
                                        ))
                            elif isinstance(data[0], str):
                                # Direct fact
                                symbols.append(Symbol(
                                    data[0], "fact", line_num, 0,
                                    len(data) - 1
                                ))
            except Exception:
                # Ignore parsing errors for symbol extraction
                pass
        
        return symbols
    
    def validate_document(self, text: str) -> List[Diagnostic]:
        """Validate DreamLog document and return diagnostics"""
        diagnostics = []
        lines = text.split('\n')
        
        # Track defined functors
        defined_functors = set()
        used_functors = set()
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            
            # Try to parse the line
            try:
                if self.config["format"] == "sexp":
                    if ":-" in line:
                        # Parse rule
                        parts = line.split(":-")
                        if len(parts) != 2:
                            diagnostics.append(Diagnostic(
                                Range(Position(line_num, 0), Position(line_num, len(line))),
                                "Invalid rule format. Expected: head :- body",
                                DiagnosticSeverity.Error
                            ))
                            continue
                        
                        head = parse_s_expression(parts[0].strip())
                        if hasattr(head, 'functor'):
                            defined_functors.add(head.functor)
                        
                        # Parse body
                        for term_str in parts[1].split(','):
                            term = parse_s_expression(term_str.strip())
                            if hasattr(term, 'functor'):
                                used_functors.add(term.functor)
                    
                    elif line.startswith("("):
                        # Parse fact or query
                        term = parse_s_expression(line)
                        if hasattr(term, 'functor'):
                            # Check if it's a query (has variables)
                            if hasattr(term, 'get_variables') and term.get_variables():
                                used_functors.add(term.functor)
                            else:
                                defined_functors.add(term.functor)
                
                else:
                    # JSON format validation
                    data = json.loads(line)
                    if not isinstance(data, list):
                        diagnostics.append(Diagnostic(
                            Range(Position(line_num, 0), Position(line_num, len(line))),
                            "Expected JSON array",
                            DiagnosticSeverity.Error
                        ))
            
            except json.JSONDecodeError as e:
                diagnostics.append(Diagnostic(
                    Range(Position(line_num, 0), Position(line_num, len(line))),
                    f"JSON syntax error: {e}",
                    DiagnosticSeverity.Error
                ))
            
            except Exception as e:
                diagnostics.append(Diagnostic(
                    Range(Position(line_num, 0), Position(line_num, len(line))),
                    f"Parse error: {e}",
                    DiagnosticSeverity.Error
                ))
        
        # Check for undefined functors (warnings)
        undefined = used_functors - defined_functors
        if undefined and self.config["enableDiagnostics"]:
            for line_num, line in enumerate(lines):
                for functor in undefined:
                    if functor in line:
                        diagnostics.append(Diagnostic(
                            Range(Position(line_num, 0), Position(line_num, len(line))),
                            f"Functor '{functor}' is used but not defined",
                            DiagnosticSeverity.Warning
                        ))
        
        return diagnostics
    
    def get_completions(self, text: str, line: int, column: int) -> List[CompletionItem]:
        """Get completion items for the current position"""
        completions = []
        
        # Get all symbols
        symbols = self.extract_symbols(text, "")
        
        # Add functor completions
        functors = set()
        for symbol in symbols:
            if symbol.type in ("fact", "rule"):
                functors.add(symbol.name)
        
        for functor in sorted(functors)[:self.config["maxCompletions"]]:
            item = CompletionItem(
                label=functor,
                kind=CompletionItemKind.Function,
                detail=f"Functor defined in this file",
                documentation=f"Predicate: {functor}"
            )
            completions.append(item)
        
        # Add common keywords
        keywords = ["fact", "rule", "query"]
        for kw in keywords:
            item = CompletionItem(
                label=kw,
                kind=CompletionItemKind.Keyword,
                detail="DreamLog keyword"
            )
            completions.append(item)
        
        # Add variable templates
        var_templates = ["X", "Y", "Z", "Var", "Result", "List", "Head", "Tail"]
        for var in var_templates:
            item = CompletionItem(
                label=var,
                kind=CompletionItemKind.Variable,
                detail="Variable template"
            )
            completions.append(item)
        
        return completions
    
    def format_document(self, text: str) -> List[TextEdit]:
        """Format DreamLog document"""
        edits = []
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('%') or line.startswith('#'):
                formatted_lines.append(line)
                continue
            
            # Format based on type
            if self.config["format"] == "sexp":
                # Add spacing around operators
                if ":-" in line:
                    parts = line.split(":-")
                    if len(parts) == 2:
                        head = parts[0].strip()
                        body = parts[1].strip()
                        # Format body with proper comma spacing
                        body_terms = [t.strip() for t in body.split(',')]
                        body = ", ".join(body_terms)
                        line = f"{head} :- {body}"
                
                formatted_lines.append(line)
            else:
                # JSON format - ensure proper JSON formatting
                try:
                    data = json.loads(line)
                    formatted_lines.append(json.dumps(data, separators=(',', ': ')))
                except:
                    formatted_lines.append(line)
        
        # Create single edit to replace entire document
        new_text = '\n'.join(formatted_lines)
        if new_text != text:
            edits.append(TextEdit(
                Range(Position(0, 0), Position(len(lines) - 1, len(lines[-1]))),
                new_text
            ))
        
        return edits


# Create the language server instance
server = DreamLogLanguageServer()


@server.feature(INITIALIZE)
async def initialize(params: InitializeParams) -> InitializeResult:
    """Initialize the language server"""
    return InitializeResult(
        capabilities={
            "textDocumentSync": 1,
            "completionProvider": {"triggerCharacters": ["(", "[", " "]},
            "hoverProvider": True,
            "definitionProvider": True,
            "referencesProvider": True,
            "documentFormattingProvider": True
        }
    )


@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(params: DidOpenTextDocumentParams):
    """Handle document open"""
    uri = params.text_document.uri
    text = params.text_document.text
    
    # Extract symbols
    server.symbols[uri] = server.extract_symbols(text, uri)
    
    # Validate and send diagnostics
    diagnostics = server.validate_document(text)
    server.publish_diagnostics(uri, diagnostics)


@server.feature(TEXT_DOCUMENT_DID_CHANGE)
async def did_change(params: DidChangeTextDocumentParams):
    """Handle document change"""
    uri = params.text_document.uri
    
    # Get the new text (assuming full document sync)
    if params.content_changes:
        text = params.content_changes[0].text
        
        # Re-extract symbols
        server.symbols[uri] = server.extract_symbols(text, uri)
        
        # Re-validate and send diagnostics
        diagnostics = server.validate_document(text)
        server.publish_diagnostics(uri, diagnostics)


@server.feature(COMPLETION)
async def completions(params: CompletionParams) -> CompletionList:
    """Provide completion suggestions"""
    uri = params.text_document.uri
    position = params.position
    
    # Get document text
    document = server.workspace.get_document(uri)
    text = document.source
    
    # Get completions
    items = server.get_completions(text, position.line, position.character)
    
    return CompletionList(is_incomplete=False, items=items)


@server.feature(HOVER)
async def hover(params: HoverParams) -> Optional[Hover]:
    """Provide hover information"""
    uri = params.text_document.uri
    position = params.position
    
    # Find symbol at position
    if uri in server.symbols:
        for symbol in server.symbols[uri]:
            if symbol.line == position.line:
                # Create hover content
                content = f"**{symbol.name}**\n\n"
                content += f"Type: {symbol.type}\n"
                if symbol.arity is not None:
                    content += f"Arity: {symbol.arity}\n"
                
                return Hover(
                    contents=MarkupContent(
                        kind=MarkupKind.Markdown,
                        value=content
                    ),
                    range=Range(
                        Position(position.line, 0),
                        Position(position.line, 100)
                    )
                )
    
    return None


@server.feature(DEFINITION)
async def definition(params: DefinitionParams) -> Optional[List[Location]]:
    """Go to definition"""
    uri = params.text_document.uri
    position = params.position
    
    # Find symbol at position
    document = server.workspace.get_document(uri)
    line = document.lines[position.line]
    
    # Extract word at position
    word_start = position.character
    word_end = position.character
    
    while word_start > 0 and (line[word_start - 1].isalnum() or line[word_start - 1] == '_'):
        word_start -= 1
    
    while word_end < len(line) and (line[word_end].isalnum() or line[word_end] == '_'):
        word_end += 1
    
    word = line[word_start:word_end]
    
    # Find definitions
    locations = []
    if uri in server.symbols:
        for symbol in server.symbols[uri]:
            if symbol.name == word and symbol.type in ("fact", "rule"):
                locations.append(Location(
                    uri=uri,
                    range=Range(
                        Position(symbol.line, 0),
                        Position(symbol.line, 100)
                    )
                ))
    
    return locations if locations else None


@server.feature(REFERENCES)
async def references(params: ReferencesParams) -> Optional[List[Location]]:
    """Find all references"""
    uri = params.text_document.uri
    position = params.position
    
    # Similar to definition, but find all occurrences
    document = server.workspace.get_document(uri)
    line = document.lines[position.line]
    
    # Extract word at position
    word_start = position.character
    word_end = position.character
    
    while word_start > 0 and (line[word_start - 1].isalnum() or line[word_start - 1] == '_'):
        word_start -= 1
    
    while word_end < len(line) and (line[word_end].isalnum() or line[word_end] == '_'):
        word_end += 1
    
    word = line[word_start:word_end]
    
    # Find all references
    locations = []
    for line_num, line_text in enumerate(document.lines):
        if word in line_text:
            locations.append(Location(
                uri=uri,
                range=Range(
                    Position(line_num, 0),
                    Position(line_num, len(line_text))
                )
            ))
    
    return locations if locations else None


@server.feature(FORMATTING)
async def formatting(params: DocumentFormattingParams) -> Optional[List[TextEdit]]:
    """Format document"""
    uri = params.text_document.uri
    document = server.workspace.get_document(uri)
    
    edits = server.format_document(document.source)
    return edits if edits else None


@server.feature(WORKSPACE_DID_CHANGE_CONFIGURATION)
async def did_change_configuration(params):
    """Handle configuration changes"""
    settings = params.settings.get("dreamlog", {})
    
    # Update configuration
    if "format" in settings:
        server.config["format"] = settings["format"]
    if "enableLLM" in settings:
        server.config["enableLLM"] = settings["enableLLM"]
    if "maxCompletions" in settings:
        server.config["maxCompletions"] = settings["maxCompletions"]
    if "enableDiagnostics" in settings:
        server.config["enableDiagnostics"] = settings["enableDiagnostics"]
    if "enableFormatting" in settings:
        server.config["enableFormatting"] = settings["enableFormatting"]


def main():
    """Main entry point"""
    server.start_io()


if __name__ == "__main__":
    main()