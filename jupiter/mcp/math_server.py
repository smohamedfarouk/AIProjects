
from datetime import datetime

from mcp.server.fastmcp import FastMCP
#MCP - To connect LLM Models with Tools - Functions(Tools) or External Data

#MCP Host - The AI Application that manages model 
#MCP Server - Code which exposes functions(Tools) or external data
#MCP Client - Is a component inside MCP host, which connects host with server
#MCP Transport - Mode of Communication channel

mcp = FastMCP("BeginnerTools")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    mcp.run(transport="stdio")
