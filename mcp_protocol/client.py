"""
Minimal MCPClient (tools-only)

- server_source: FastMCP | TransportConfig
- transports: STDIO (StdioTransport), HTTP (StreamableHttpTransport), SSE (SSETransport)
- no resources / prompts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from fastmcp import Client, FastMCP
    from fastmcp.client.transports import StdioTransport, SSETransport, StreamableHttpTransport
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    Client = None
    FastMCP = None
    StdioTransport = None
    SSETransport = None
    StreamableHttpTransport = None


class TransportKind(str, Enum):
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


@dataclass(frozen=True)
class TransportConfig:
    kind: TransportKind

    # ---- STDIO ----
    command: Optional[str] = None          # e.g. "python", "uv", "node"
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    keep_alive: Optional[bool] = None      # FastMCP stdio session persistence (optional) :contentReference[oaicite:1]{index=1}

    # ---- HTTP / SSE ----
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    auth: Optional[Any] = None

    # pass-through kwargs to transport constructors
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind == TransportKind.STDIO:
            if not self.command:
                raise ValueError("STDIO requires command.")
        elif self.kind in (TransportKind.HTTP, TransportKind.SSE):
            if not self.url:
                raise ValueError(f"{self.kind.value.upper()} requires url.")
        else:
            raise ValueError(f"Unsupported transport kind: {self.kind}")


ServerSource = Union["FastMCP", TransportConfig]


class MCPClient:
    """tools-only MCP client wrapper."""

    def __init__(self, server_source: ServerSource):
        if not FASTMCP_AVAILABLE:
            raise ImportError("This MCPClient requires fastmcp>=2.0.0 (pip install fastmcp>=2.0.0).")

        self._server_or_transport = self._normalize(server_source)
        self.client: Optional[Client] = None
        self._ctx = None

    def _normalize(self, server_source: ServerSource) -> Any:
        # memory transport: FastMCP instance
        if isinstance(server_source, FastMCP):
            return server_source

        # explicit transport config
        if isinstance(server_source, TransportConfig):
            server_source.validate()
            return self._build_transport(server_source)

        raise TypeError("server_source must be a FastMCP instance or a TransportConfig.")

    def _build_transport(self, cfg: TransportConfig) -> Any:
        if cfg.kind == TransportKind.STDIO:
            kwargs = dict(cfg.extra)
            if cfg.keep_alive is not None:
                kwargs["keep_alive"] = cfg.keep_alive
            return StdioTransport(
                command=cfg.command,
                args=cfg.args,
                env=cfg.env,
                cwd=cfg.cwd,
                **kwargs,
            )

        if cfg.kind == TransportKind.HTTP:
            return StreamableHttpTransport(
                url=cfg.url,
                headers=cfg.headers,
                auth=cfg.auth,
                **cfg.extra,
            )

        if cfg.kind == TransportKind.SSE:
            return SSETransport(
                url=cfg.url,
                headers=cfg.headers,
                auth=cfg.auth,
                **cfg.extra,
            )

        raise ValueError(f"Unsupported transport kind: {cfg.kind}")

    async def __aenter__(self) -> "MCPClient":
        """异步上下文管理器入口"""
        self.client = Client(self._server_or_transport)
        self._ctx = self.client
        await self._ctx.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器退出"""
        if self._ctx:
            await self._ctx.__aexit__(exc_type, exc_val, exc_tb)
        self.client = None
        self._ctx = None

    def _require_connected(self) -> None:
        if not self.client:
            raise RuntimeError("Client not connected. Use: async with MCPClient(...) as c:")

    async def ping(self) -> bool:
        self._require_connected()
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        self._require_connected()
        result = await self.client.list_tools()
        tools = result.tools if hasattr(result, "tools") else (result or [])
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": getattr(t, "inputSchema", {}) or {},
            }
            for t in tools
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        self._require_connected()
        result = await self.client.call_tool(name, arguments)

        # ToolResult(content=[...]) -> return common payloads
        if hasattr(result, "content") and result.content:
            if len(result.content) == 1:
                c = result.content[0]
                if hasattr(c, "text"):
                    return c.text
                if hasattr(c, "data"):
                    return c.data
                return str(c)

            out = []
            for c in result.content:
                out.append(getattr(c, "text", getattr(c, "data", str(c))))
            return out

        return None

    def transport_info(self) -> Dict[str, Any]:
        if not self.client:
            return {"status": "not_connected"}
        t = getattr(self.client, "transport", None)
        return {
            "status": "connected",
            "transport_type": type(t).__name__ if t else None,
            "transport_repr": str(t) if t else None,
        }