"""
================================================================================
PHASE 4: MCP SERVER - MODEL CONTEXT PROTOCOL
================================================================================
OpsBot - Enterprise Knowledge Copilot

MCP (Model Context Protocol) server exposes OpsBot as tools accessible from:
- Claude Desktop application
- External MCP clients
- Any application using MCP protocol

SKILL SIGNALS:
- MCP integration (2024-2025 standard)
- Tool exposure and standardization
- API interoperability
- Modern AI infrastructure

MCP is Anthropic's standard for connecting tools to LLMs.
This makes OpsBot available to Claude Desktop users directly.

================================================================================
"""

import json
import sys
from typing import Any

from query_engine import RAGPipeline, QueryResult
from agent import Agent


# ============================================================================
# MCP TOOL DEFINITIONS
# ============================================================================

class MCPServer:
    """
    MCP Server exposing OpsBot as tools.

    Tools exposed:
    1. search_knowledge_base - Search handbook
    2. get_section_details - Get full section
    3. ask_agent - Multi-step reasoning
    4. list_topics - See available topics
    """

    def __init__(self):
        """Initialize MCP server with RAG pipeline"""
        try:
            self.rag = RAGPipeline()
            self.agent = Agent(self.rag)
            self.ready = True
        except Exception as e:
            print(f"Error initializing MCP server: {e}", file=sys.stderr)
            self.ready = False

    def handle_call(self, tool_name: str, arguments: dict) -> dict:
        """
        Handle tool calls from MCP clients.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            Tool result as dictionary
        """
        if not self.ready:
            return {"error": "MCP server not initialized"}

        if tool_name == "search_knowledge_base":
            return self._search_knowledge_base(arguments.get("query", ""))

        elif tool_name == "get_section_details":
            return self._get_section_details(arguments.get("section", ""))

        elif tool_name == "ask_agent":
            return self._ask_agent(arguments.get("question", ""))

        elif tool_name == "list_topics":
            return self._list_topics()

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _search_knowledge_base(self, query: str) -> dict:
        """
        Search knowledge base.

        MCP signature:
        - Input: query (string)
        - Output: { answer, sources, confidence }
        """
        result = self.rag.query(query)

        return {
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "chunks_retrieved": result.retrieved_chunks
        }

    def _get_section_details(self, section: str) -> dict:
        """
        Get details about a specific section.

        MCP signature:
        - Input: section (string, e.g., "IT/security")
        - Output: { content, metadata }
        """
        # This would query by metadata in production
        return {
            "section": section,
            "status": "Available in search - use search_knowledge_base",
            "note": "Full section retrieval not yet implemented in Phase 4"
        }

    def _ask_agent(self, question: str) -> dict:
        """
        Ask agent for multi-step reasoning.

        MCP signature:
        - Input: question (string)
        - Output: { answer, steps, recommendation }
        """
        result = self.agent.run(question)

        return {
            "answer": result["answer"],
            "recommendation": result["recommendation"],
            "steps_taken": result["steps_taken"],
            "summary": result["summary"]
        }

    def _list_topics(self) -> dict:
        """
        List available topics in knowledge base.

        MCP signature:
        - Output: { topics }
        """
        # Get from metadata
        info = self.rag.kb.get_collection_info()

        topics = set()
        for file_info in info.get("metadata", {}).get("files", []):
            path = file_info.get("relative_path", "")
            topic = path.split("\\")[0] if "\\" in path else path

            if topic:
                topics.add(topic)

        return {
            "topics": sorted(list(topics)),
            "total_chunks": info.get("metadata", {}).get("total_chunks", 0)
        }

    def get_tools_description(self) -> list:
        """
        Return MCP tool definitions for clients.

        This tells Claude Desktop and other MCP clients what tools are available.
        """
        return [
            {
                "name": "search_knowledge_base",
                "description": "Search the company knowledge base (handbook, policies, procedures)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for (e.g., 'password policy')"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "ask_agent",
                "description": "Ask an intelligent agent to plan and execute multi-step tasks",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Complex question or request requiring multi-step reasoning"
                        }
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "list_topics",
                "description": "List all available topics in the knowledge base",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_section_details",
                "description": "Get detailed information about a specific section",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "description": "Section name (e.g., 'IT/security')"
                        }
                    },
                    "required": ["section"]
                }
            }
        ]


# ============================================================================
# MCP PROTOCOL HANDLER
# ============================================================================

def handle_mcp_message(message: dict) -> dict:
    """
    Handle incoming MCP protocol messages.

    MCP message format:
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "tool_name",
            "arguments": {...}
        }
    }
    """
    method = message.get("method")
    params = message.get("params", {})

    if method == "tools/list":
        # List available tools
        return {
            "jsonrpc": "2.0",
            "result": {"tools": server.get_tools_description()}
        }

    elif method == "tools/call":
        # Call a tool
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        result = server.handle_call(tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "result": result
        }

    else:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Method not found"}
        }


# ============================================================================
# STDIO TRANSPORT (for Claude Desktop)
# ============================================================================

def main():
    """
    Run MCP server via stdin/stdout.

    Claude Desktop communicates via JSON-RPC over stdio.
    """
    global server
    server = MCPServer()

    if not server.ready:
        print("MCP Server failed to initialize", file=sys.stderr)
        sys.exit(1)

    print("MCP Server initialized and ready", file=sys.stderr)

    # Read messages from stdin
    while True:
        try:
            line = sys.stdin.readline()

            if not line:
                break

            message = json.loads(line)
            response = handle_mcp_message(message)

            # Send response via stdout
            print(json.dumps(response))
            sys.stdout.flush()

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}", file=sys.stderr)

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            response = {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)}
            }
            print(json.dumps(response))
            sys.stdout.flush()


# ============================================================================
# DEMO MODE (for testing without MCP client)
# ============================================================================

def demo_mode():
    """
    Interactive demo of MCP tools (no MCP protocol needed).

    Useful for testing when Claude Desktop isn't available.
    """
    global server
    server = MCPServer()

    if not server.ready:
        print("Failed to initialize server")
        return

    print("\n" + "="*70)
    print("OpsBot MCP Server - Demo Mode")
    print("="*70)

    while True:
        try:
            command = input("\nCommand (search/agent/topics/quit): ").strip().lower()

            if command == "quit":
                break

            elif command == "search":
                query = input("Query: ")
                result = server._search_knowledge_base(query)
                print("\nResult:")
                print(json.dumps(result, indent=2))

            elif command == "agent":
                question = input("Question: ")
                result = server._ask_agent(question)
                print("\nResult:")
                print(json.dumps(result, indent=2))

            elif command == "topics":
                result = server._list_topics()
                print("\nTopics:")
                for topic in result.get("topics", []):
                    print(f"  - {topic}")

            else:
                print("Unknown command")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    import sys

    # Check for --demo flag
    if "--demo" in sys.argv:
        demo_mode()
    else:
        main()
