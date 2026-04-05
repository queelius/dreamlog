"""
DreamLog MCP Server — persistent, self-compressing knowledge store.

Exposes DreamLog as an MCP tool for agentic LLMs. The KB persists to disk,
compresses via the dream cycle, and learns over time.

Usage:
    dreamlog-mcp                          # Default store at ~/.dreamlog/stores/default.json
    DREAMLOG_STORE=./my.json dreamlog-mcp # Custom store path

Configure in .mcp.json:
    {
      "mcpServers": {
        "dreamlog": {
          "command": "dreamlog-mcp",
          "env": {"DREAMLOG_STORE": "~/.dreamlog/stores/project.json"}
        }
      }
    }
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP, Context
from pydantic import Field

# Ensure dreamlog is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integrations.mcp.knowledge_store import KnowledgeStore


@asynccontextmanager
async def lifespan(server):
    store_path = Path(os.environ.get(
        "DREAMLOG_STORE", "~/.dreamlog/stores/default.json"
    )).expanduser()
    budget = float(os.environ.get("DREAMLOG_LLM_BUDGET", "0.50"))
    threshold = int(os.environ.get("DREAMLOG_DREAM_THRESHOLD", "50"))

    store = KnowledgeStore(store_path, llm_budget_usd=budget,
                           dream_threshold=threshold)
    try:
        yield {"store": store}
    finally:
        store._dirty = True
        store.save()


mcp = FastMCP("dreamlog", lifespan=lifespan)


# ── Tools ─────────────────────────────────────────────────────────

@mcp.tool()
def dreamlog_assert(
    fact: Annotated[str, Field(
        description=(
            "Fact or rule in S-expression format. "
            "Facts: (predicate arg1 arg2). "
            "Rules: (head X Y) :- (body1 X Z), (body2 Z Y). "
            "Variables are UPPERCASE (X, Y, Z). "
            "Constants are lowercase (john, mary). "
            "Examples: '(parent john mary)', "
            "'(ancestor X Z) :- (parent X Y), (ancestor Y Z)'"
        ),
    )],
    ctx: Context = None,
) -> dict:
    """Add a fact or rule to the persistent knowledge base.

    The knowledge base persists across sessions. After enough assertions,
    call dreamlog_dream to compress the KB and discover generalizations.
    """
    store: KnowledgeStore = ctx.request_context.lifespan_context["store"]
    return store.assert_fact(fact)


@mcp.tool()
def dreamlog_query(
    query: Annotated[str, Field(
        description=(
            "Goal in S-expression format with variables. "
            "Examples: '(parent john X)' finds john's children, "
            "'(ancestor X mary)' finds mary's ancestors. "
            "Use uppercase for variables (unknowns), lowercase for constants."
        ),
    )],
    limit: Annotated[int, Field(
        description="Maximum number of solutions to return.",
        default=10,
    )] = 10,
    ctx: Context = None,
) -> dict:
    """Query the knowledge base using SLD resolution.

    Returns variable bindings for each solution. Supports negation-as-failure,
    transitive closure, and all rules discovered during dream cycles.
    """
    store: KnowledgeStore = ctx.request_context.lifespan_context["store"]
    return store.query(query, limit=limit)


@mcp.tool()
def dreamlog_dream(
    dry_run: Annotated[bool, Field(
        description="If true, preview compression without applying changes.",
        default=False,
    )] = False,
    ctx: Context = None,
) -> dict:
    """Run the sleep/compression cycle on the knowledge base.

    Discovers generalizations, removes redundant facts, invents predicates,
    and optionally uses an LLM to find cross-predicate rules. All changes
    are verified against a test suite — no behavioral changes are introduced.

    Call this after accumulating facts to compress and learn. Check
    dreamlog_status to see if dreaming is recommended.
    """
    store: KnowledgeStore = ctx.request_context.lifespan_context["store"]
    return store.dream(dry_run=dry_run)


@mcp.tool()
def dreamlog_explain(
    query: Annotated[str, Field(
        description=(
            "Goal to explain, in S-expression format. "
            "Example: '(grandparent john carol)'"
        ),
    )],
    ctx: Context = None,
) -> dict:
    """Explain how a query resolves — show matching facts and rules."""
    store: KnowledgeStore = ctx.request_context.lifespan_context["store"]
    return store.explain(query)


@mcp.tool()
def dreamlog_status(
    ctx: Context = None,
) -> dict:
    """Show knowledge base status, LLM budget, and dream readiness.

    Returns fact/rule counts, predicate list, assertion/query history,
    dream recommendation, and LLM usage/budget information.
    """
    store: KnowledgeStore = ctx.request_context.lifespan_context["store"]
    return store.status()


# ── Resources ─────────────────────────────────────────────────────

@mcp.resource("dreamlog://kb")
def get_kb(ctx: Context = None) -> str:
    """Current knowledge base in prefix-notation JSON."""
    store: KnowledgeStore = ctx.request_context.lifespan_context["store"]
    return store.kb.to_prefix()


@mcp.resource("dreamlog://stats")
def get_stats(ctx: Context = None) -> str:
    """Knowledge base statistics as JSON."""
    store: KnowledgeStore = ctx.request_context.lifespan_context["store"]
    return json.dumps(store.status(), indent=2)


# ── Entry point ───────────────────────────────────────────────────

def main():
    mcp.run()


if __name__ == "__main__":
    main()
