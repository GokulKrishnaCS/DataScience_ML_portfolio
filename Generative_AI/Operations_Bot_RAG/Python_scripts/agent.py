"""
================================================================================
PHASE 3: AGENTIC WORKFLOWS & TOOL USE
================================================================================
OpsBot - Enterprise Knowledge Copilot

This script implements intelligent agents that:
1. Understand complex requests
2. Plan multi-step solutions
3. Use tools (search docs, create tickets, look up info)
4. Reason about next steps
5. Provide recommendations

SKILL SIGNALS THIS DEMONSTRATES:
- Agentic workflows: planning, execution, reflection
- Tool calling: LLM decides which tools to use
- Multi-step reasoning: breaking down complex tasks
- State management: memory across multiple steps
- Production patterns: error handling, guardrails

This is what recruiters most want to see - it shows you can build
systems that are smarter than simple RAG.

================================================================================
"""

import json
from typing import List, Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import re

from query_engine import RAGPipeline, QueryResult, RetrievedChunk


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

class ToolType(Enum):
    """Available tools that agents can use"""
    SEARCH_DOCS = "search_docs"           # Search knowledge base
    CREATE_TICKET = "create_ticket"       # Create IT/HR ticket
    GET_EMPLOYEE_INFO = "get_employee_info"  # Look up employee
    RECOMMEND_NEXT_STEP = "recommend_next_step"  # Suggest action


@dataclass
class Tool:
    """Definition of a tool agent can use"""
    name: str
    description: str
    parameters: Dict[str, str]  # param_name -> type description
    example_usage: str


# Define available tools
TOOLS = {
    "search_docs": Tool(
        name="search_docs",
        description="Search the company knowledge base (handbook, policies, procedures)",
        parameters={
            "query": "string - what to search for",
        },
        example_usage='{"tool": "search_docs", "query": "password policy"}'
    ),
    "create_ticket": Tool(
        name="create_ticket",
        description="Create a support ticket (IT, HR, etc)",
        parameters={
            "category": "string - category (IT, HR, Finance, etc)",
            "title": "string - ticket title",
            "description": "string - detailed description",
        },
        example_usage='{"tool": "create_ticket", "category": "IT", "title": "Password reset", "description": "Need to reset my password"}'
    ),
    "recommend_next_step": Tool(
        name="recommend_next_step",
        description="Generate a recommended next action based on context",
        parameters={
            "context": "string - what the user is trying to do",
        },
        example_usage='{"tool": "recommend_next_step", "context": "User needs to request PTO"}'
    ),
}


# ============================================================================
# TOOL EXECUTION
# ============================================================================

class ToolExecutor:
    """Execute tools (mocked for demo purposes)"""

    def __init__(self, rag: RAGPipeline):
        """
        Initialize tool executor with RAG pipeline for search.

        Args:
            rag: RAGPipeline instance for search_docs
        """
        self.rag = rag
        self.tickets_created = []  # Track created tickets

    def execute(self, tool_name: str, **kwargs) -> str:
        """
        Execute a tool with given arguments.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments

        Returns:
            Tool result as string
        """
        if tool_name == "search_docs":
            return self._search_docs(kwargs.get("query", ""))

        elif tool_name == "create_ticket":
            return self._create_ticket(
                kwargs.get("category", ""),
                kwargs.get("title", ""),
                kwargs.get("description", "")
            )

        elif tool_name == "recommend_next_step":
            return self._recommend_next_step(kwargs.get("context", ""))

        else:
            return f"Unknown tool: {tool_name}"

    def _search_docs(self, query: str) -> str:
        """Search knowledge base"""
        result = self.rag.query(query)
        return f"Found answer: {result.answer[:200]}... (Source: {result.sources[0]['file'] if result.sources else 'Unknown'})"

    def _create_ticket(self, category: str, title: str, description: str) -> str:
        """Create ticket (mocked)"""
        ticket_id = f"TICKET-{len(self.tickets_created) + 1001}"
        ticket = {
            "id": ticket_id,
            "category": category,
            "title": title,
            "description": description,
            "status": "open"
        }
        self.tickets_created.append(ticket)
        return f"Ticket created: {ticket_id} - {title}"

    def _recommend_next_step(self, context: str) -> str:
        """Recommend action (heuristic-based)"""
        context_lower = context.lower()

        if "password" in context_lower:
            return "Next step: Submit a ticket to IT for password reset. Response time: 2 hours."
        elif "pto" in context_lower or "time off" in context_lower:
            return "Next step: Log into the HR portal and submit PTO request. Approval within 24 hours."
        elif "expense" in context_lower or "reimbursement" in context_lower:
            return "Next step: File expense report in Finance portal. Keep receipts."
        else:
            return "Next step: Check the knowledge base for more details or create a support ticket."


# ============================================================================
# AGENT STATE & MEMORY
# ============================================================================

@dataclass
class AgentStep:
    """A single step in agent execution"""
    step_num: int
    action: str  # What the agent decides to do
    tool_name: str  # Which tool (if any)
    tool_input: Dict  # Parameters for tool
    observation: str  # Result of tool execution
    reasoning: str  # Why agent made this decision


class AgentMemory:
    """Memory for multi-step agents - tracks conversation and decisions"""

    def __init__(self, user_query: str):
        """
        Initialize agent memory for a query.

        Args:
            user_query: Original user request
        """
        self.user_query = user_query
        self.steps: List[AgentStep] = []
        self.internal_thoughts: List[str] = []

    def add_step(self, step: AgentStep):
        """Record a step"""
        self.steps.append(step)

    def add_thought(self, thought: str):
        """Record internal reasoning"""
        self.internal_thoughts.append(thought)

    def get_context_for_llm(self) -> str:
        """
        Format memory as context string for LLM.

        Shows previous steps and decisions.
        """
        context = f"Original request: {self.user_query}\n\n"

        context += "Actions taken so far:\n"
        for step in self.steps:
            context += f"  {step.step_num}. {step.action}\n"
            if step.tool_name:
                context += f"     Used tool: {step.tool_name}\n"
                context += f"     Result: {step.observation[:100]}...\n"

        return context

    def summary(self) -> str:
        """Summary of steps taken"""
        if not self.steps:
            return "No steps taken yet."

        summary = f"Steps taken: {len(self.steps)}\n"
        for step in self.steps:
            summary += f"  {step.step_num}. {step.action} ({step.tool_name or 'thinking'})\n"

        return summary


# ============================================================================
# AGENT ORCHESTRATION
# ============================================================================

class Agent:
    """
    Intelligent agent that plans and executes tasks.

    Decision loop:
    1. Understand user query
    2. Plan next action
    3. Execute (use tool or think)
    4. Observe result
    5. Decide if done or loop
    """

    def __init__(self, rag: RAGPipeline, name: str = "OpsBot Agent"):
        """
        Initialize agent.

        Args:
            rag: RAGPipeline for knowledge access
            name: Agent identifier
        """
        self.name = name
        self.rag = rag
        self.executor = ToolExecutor(rag)
        self.max_steps = 5  # Prevent infinite loops

    def run(self, user_query: str) -> Dict:
        """
        Execute agent on a user query.

        Args:
            user_query: What user is asking

        Returns:
            Dictionary with final answer, steps taken, recommendation
        """
        print(f"\n[AGENT] {self.name} thinking about: {user_query}")
        print("="*70)

        memory = AgentMemory(user_query)

        # Main agent loop
        for step_num in range(1, self.max_steps + 1):
            print(f"\n[Step {step_num}] Planning action...")

            # Decide what to do
            action, tool_name, tool_input, reasoning = self._decide_action(
                user_query, memory
            )

            print(f"  Decision: {action}")
            if tool_name:
                print(f"  Using tool: {tool_name}")

            # Execute action
            if tool_name:
                observation = self.executor.execute(tool_name, **tool_input)
            else:
                observation = "Reasoning step complete."

            # Record step
            step = AgentStep(
                step_num=step_num,
                action=action,
                tool_name=tool_name or "",
                tool_input=tool_input,
                observation=observation,
                reasoning=reasoning
            )
            memory.add_step(step)

            print(f"  Observation: {observation[:100]}...")

            # Check if done
            if self._should_finish(action, step_num):
                print(f"\n[DONE] Agent finished after {step_num} steps")
                break

        # Generate final answer
        final_answer = self._generate_final_answer(user_query, memory)
        recommendation = self._generate_recommendation(user_query, memory)

        result = {
            "query": user_query,
            "answer": final_answer,
            "recommendation": recommendation,
            "steps_taken": len(memory.steps),
            "summary": memory.summary(),
            "internal_reasoning": memory.internal_thoughts
        }

        return result

    def _decide_action(self, user_query: str, memory: AgentMemory) -> tuple:
        """
        Decide next action using LLM.

        This is where tool calling happens - LLM decides which tool to use.

        Returns:
            (action_description, tool_name, tool_params, reasoning)
        """
        # Simplified decision logic (in production: use LLM with tool calling)

        if len(memory.steps) == 0:
            # First step: search for information
            return (
                "Search knowledge base",
                "search_docs",
                {"query": user_query},
                "User is asking about something in knowledge base"
            )

        elif len(memory.steps) == 1:
            # Second step: recommend next step
            context = f"User query: {user_query}"
            if memory.steps[0].observation:
                context += f"\nFound: {memory.steps[0].observation[:100]}"

            return (
                "Recommend next action",
                "recommend_next_step",
                {"context": context},
                "Based on search result, recommend action"
            )

        else:
            # Done - no more steps
            return (
                "Conclude conversation",
                None,
                {},
                "Have gathered enough information"
            )

    def _should_finish(self, last_action: str, step_num: int) -> bool:
        """Decide if agent should stop"""
        # Stop if we've done multiple steps or hit max
        if "conclude" in last_action.lower() or step_num >= self.max_steps:
            return True
        return False

    def _generate_final_answer(self, query: str, memory: AgentMemory) -> str:
        """Generate final answer from agent's work"""
        if not memory.steps:
            return "No information found."

        # Combine findings from all steps
        answer_parts = ["Based on my investigation:"]

        for step in memory.steps:
            if step.observation:
                answer_parts.append(f"- {step.observation[:150]}")

        return "\n".join(answer_parts)

    def _generate_recommendation(self, query: str, memory: AgentMemory) -> str:
        """Generate recommended next action"""
        if memory.steps:
            last_step = memory.steps[-1]
            if "recommend" in last_step.action.lower():
                return last_step.observation

        return "Consider creating a support ticket for further assistance."


# ============================================================================
# MULTI-AGENT SYSTEMS (ADVANCED)
# ============================================================================

class MultiAgentOrchestrator:
    """
    Orchestrate multiple agents working together.

    Example: Research Agent + Recommendation Agent
    """

    def __init__(self, rag: RAGPipeline):
        """Initialize orchestrator with multiple agents"""
        self.rag = rag
        self.research_agent = Agent(rag, "Research Agent")
        self.recommendation_agent = Agent(rag, "Recommendation Agent")

    def handle_complex_query(self, query: str) -> Dict:
        """
        Route complex queries to appropriate agents.

        Simple rule-based routing (in production: use classifier).
        """
        # Route based on query type
        if any(word in query.lower() for word in ["compare", "difference", "vs"]):
            return {
                "agent_used": "comparison_agent",
                "note": "Would route to specialized comparison agent",
                "example": "Compare IT vs HR processes"
            }

        elif any(word in query.lower() for word in ["create", "submit", "request"]):
            # Complex multi-step task
            print("\n[ORCHESTRATOR] Complex request detected - using multi-step agent")
            return self.research_agent.run(query)

        else:
            # Simple query
            return self.research_agent.run(query)


# ============================================================================
# MAIN - INTERACTIVE AGENT
# ============================================================================

def main():
    """Run interactive agent loop"""
    print("\n" + "="*72)
    print("OpsBot - Intelligent Agent Assistant")
    print("="*72)
    print("\nAsk complex questions. Agent will plan and execute tasks.")
    print("Type 'exit' to quit.\n")

    # Initialize
    try:
        rag = RAGPipeline()
        agent = Agent(rag)
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return

    # Main loop
    while True:
        try:
            query = input("\nYour request: ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            if not query:
                continue

            # Run agent
            result = agent.run(query)

            # Display results
            print("\n" + "="*70)
            print("FINAL ANSWER:")
            print("="*70)
            print(result["answer"])

            print("\n" + "="*70)
            print("RECOMMENDED NEXT STEP:")
            print("="*70)
            print(result["recommendation"])

            print("\n" + "="*70)
            print("EXECUTION SUMMARY:")
            print("="*70)
            print(result["summary"])

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
