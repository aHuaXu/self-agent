import pytest
import asyncio
import sys
import textwrap

from mcp_protocol.client import MCPClient, TransportConfig, TransportKind
from mcp_protocol.server import MCPServer

fastmcp = pytest.importorskip("fastmcp")


@pytest.mark.asyncio
async def test_memory_list_tools_and_call_tool():
    # 1) 构造 server，并注册一个 tool
    server = MCPServer(name="TestServer")

    def add(a: int, b: int) -> str:
        """Add two integers and return a string."""
        return str(a + b)

    server.add_tool(add, name="add", description="Add two integers")

    # 2) memory transport：直接把 FastMCP 实例传给 client
    async with MCPClient(server.mcp) as client:
        tools = await client.list_tools()
        tool_names = {t["name"] for t in tools}

        assert "add" in tool_names

        # 3) 调用工具
        out = await client.call_tool("add", {"a": 2, "b": 3})
        assert out == "5"


@pytest.mark.asyncio
async def test_memory_ping():
    server = MCPServer(name="PingServer")

    async with MCPClient(server.mcp) as client:
        ok = await client.ping()
        assert ok is True

def test_server_get_info():
    server = MCPServer(name="InfoServer", description="desc")
    info = server.get_info()
    assert info["name"] == "InfoServer"
    assert info["description"] == "desc"
    assert info["protocol"] == "MCP"

def _has_stdio_transport() -> bool:
    try:
        from fastmcp.client.transports import StdioTransport  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_stdio_list_tools_and_call_tool(tmp_path):
    if not _has_stdio_transport():
        pytest.skip("fastmcp.client.transports.StdioTransport not available in this fastmcp version")

    # 1) 写一个最小 server 脚本（子进程执行）
    server_py = tmp_path / "mcp_stdio_server.py"
    server_py.write_text(
        textwrap.dedent(
            """
            from fastmcp import FastMCP

            mcp = FastMCP("StdioTestServer")

            @mcp.tool(name="add", description="Add two integers")
            def add(a: int, b: int) -> str:
                return str(a + b)

            if __name__ == "__main__":
                # stdio 方式运行：通过 stdin/stdout 与 client 通信
                mcp.run(transport="stdio")
            """
        ).lstrip(),
        encoding="utf-8",
    )

    # 2) client 用 STDIO transport 启动子进程并连接
    # 用 sys.executable + "-u" 确保 python 不缓冲 stdout（降低握手卡住概率）
    cfg = TransportConfig(
        kind=TransportKind.STDIO,
        command=sys.executable,
        args=["-u", str(server_py)],
        env={**dict(**{"PYTHONUNBUFFERED": "1"})},
        cwd=str(tmp_path),
        keep_alive=False,  # 单测里不需要长期保活
    )

    async def _run():
        async with MCPClient(cfg) as client:
            tools = await client.list_tools()
            tool_names = {t["name"] for t in tools}
            assert "add" in tool_names

            out = await client.call_tool("add", {"a": 2, "b": 3})
            assert out == "5"

            assert await client.ping() is True

    # 3) 超时兜底，避免 CI/本机卡死
    await asyncio.wait_for(_run(), timeout=15)