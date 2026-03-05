from typing import Optional, Dict, Any, Callable
try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "fastmcp is required for MCP server functionality. "
        "Install it with: pip install fastmcp"
    )

class MCPServer:
    """基于 fastmcp 库的 MCP 服务器"""
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None
    ):
        """
        初始化 MCP 服务器
        
        Args:
            name: 服务器名称
            description: 服务器描述
        """
        self.mcp = FastMCP(name=name)
        self.name = name
        self.description = description or f"{name} MCP Server"
        
    def add_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        添加工具到服务器
        
        Args:
            func: 工具函数
            name: 工具名称（可选，默认使用函数名）
            description: 工具描述（可选，默认使用函数文档字符串）
        """
        # 使用装饰器注册工具
        if name or description:
            self.mcp.tool(name=name, description=description)(func)
        else:
            self.mcp.tool()(func)
        
    def run(self, transport: str = "stdio", **kwargs):
        """运行服务器

        Args:
            transport: 传输方式 ("stdio", "http", "sse")
            **kwargs: 传输特定的参数
                - host: HTTP 服务器主机（默认 "127.0.0.1"）
                - port: HTTP 服务器端口（默认 8000）
                - 其他 FastMCP.run() 支持的参数

        Examples:
            # Stdio 传输（默认）
            server.run()

            # HTTP 传输
            server.run(transport="http", host="0.0.0.0", port=8081)

            # SSE 传输
            server.run(transport="sse", host="0.0.0.0", port=8081)
        """
        self.mcp.run(transport=transport, **kwargs)
        
    def get_info(self) -> Dict[str, Any]:
        """
        获取服务器信息
        
        Returns:
            服务器信息字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "protocol": "MCP"
        }
