"""stock-advisor-mcp: a standalone MCP server exposing KRX stock data
(via yfinance + the unauthenticated KRX master files) for Claude Desktop.

Deliberately independent of the `invest-project` trading
repository: no shared code, no credentials, no database. See
`D:\\Investment_Project\\handoff\\knowledge\\stock-advisor-mcp-tool-spec-2026-09-21.md`
for the design this implements.
"""

__version__ = "0.1.0"
