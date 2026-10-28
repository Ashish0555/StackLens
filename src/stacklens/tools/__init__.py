"""Tool registry.

A tool is a plain function the model may decide to call. The model never sees
the body -- only the name, the signature and the description. That description
is the contract, which is why it is written in the same language as the
answers and built from the configured knowledge domain.

Tools are assembled by a function rather than held in a constant because the
search tool's description depends on settings read at runtime.

To add a tool: write it in its own module, then register it in build_tools().
"""

from langchain_core.tools import BaseTool

from stacklens.tools.calculator import build_calculator_tool, calculate
from stacklens.tools.documentation import build_search_tool, search_documentation


def build_tools() -> list[BaseTool]:
    """Assemble every tool the agent may call."""
    return [build_search_tool(), build_calculator_tool()]


__all__ = [
    "build_calculator_tool",
    "build_search_tool",
    "build_tools",
    "calculate",
    "search_documentation",
]
