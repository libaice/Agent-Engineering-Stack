from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator-server")


# uv run mcp dev src/mcp/mcp_calculator_server.py


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


if __name__ == "__main__":
    mcp.run()