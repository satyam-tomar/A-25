import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client
from logger import get_logger

log = get_logger("mcp_client")

MCP_URL = "http://localhost:8003/sse"

async def _call_tool_async(tool_name: str, arguments: dict):
    log.debug(f"Connecting to MCP SSE at {MCP_URL}")
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            log.debug(f"MCP session initialized, calling tool='{tool_name}'")
            result = await session.call_tool(tool_name, arguments)
            log.debug(f"MCP tool='{tool_name}' call complete")
            return result

def call_tool(tool_name: str, arguments: dict):
    return asyncio.run(_call_tool_async(tool_name, arguments))

def _parse_tool_response(result) -> list:
    if result is None:
        return []
    if hasattr(result, "content"):
        items = result.content
    elif isinstance(result, dict):
        items = result.get("content", [])
    elif isinstance(result, list):
        items = result
    else:
        return []

    docs = []
    for item in items:
        text = item.text if hasattr(item, "text") else item.get("text", "") if isinstance(item, dict) else str(item)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                docs.extend(parsed)
            else:
                docs.append(parsed)
        except (json.JSONDecodeError, TypeError):
            docs.append({"text": text, "score": 0.0})
    return docs

def search_documents(query: str) -> list:
    short = query[:60] + ("..." if len(query) > 60 else "")
    log.info(f"search_documents query='{short}'")
    try:
        result = call_tool("search_documents", {"query": query})
        docs = _parse_tool_response(result)
        log.info(f"search_documents returned {len(docs)} result(s)")
        return docs
    except Exception as e:
        log.error(f"search_documents failed: {e}")
        return []