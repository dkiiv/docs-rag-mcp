"""End to end over stdio: start the real MCP server as a subprocess and talk to
it with the MCP client, the same way an agent does."""

import json
import os
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {"search_docs", "read_doc", "ask_docs", "recent_changes", "index_info"}


def _payload(result):
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    return json.loads(result.content[0].text)


async def _session_run(env, fn):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "docsrag.mcp_server"],
        env={**os.environ, "DOCSRAG_EMBED_URL": "hash://", "DOCSRAG_DB": str(env["settings"].db_path),
             "DOCSRAG_NAME": "testdocs", "DOCSRAG_DESCRIPTION": "test runbooks"})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def test_tools_and_calls_over_stdio(env):
    async def scenario(session):
        tools = {tool.name for tool in (await session.list_tools()).tools}
        assert tools == EXPECTED_TOOLS

        found = _payload(await session.call_tool("search_docs", {"query": "restore a snapshot", "k": 2}))
        top = found["results"][0]
        assert top["path"] == "runbooks/backup.md"

        start, end = top["lines"].split("-")
        read = _payload(await session.call_tool("read_doc", {"path": top["path"], "start_line": int(start),
                                                             "end_line": int(end)}))
        assert "restore-snapshot" in read["text"]

        refused = _payload(await session.call_tool("read_doc", {"path": "../../etc/passwd"}))
        assert "error" in refused

        info = _payload(await session.call_tool("index_info", {}))
        assert info["name"] == "testdocs" and info["documents"] == 3
        return True

    assert anyio.run(_session_run, env, scenario)
