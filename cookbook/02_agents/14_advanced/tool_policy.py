"""
Tool Policy
=============================

Example demonstrating deterministic tool-call authorization with Agno's
``ToolPolicy``. Every tool call is checked against the policy before it
executes; denied calls are never run and the model receives a structured
denial message instead.

The policy is evaluated synchronously with plain ``fnmatch`` pattern
matching - no LLM is involved in the authorization path.

Run:
    .venvs/demo/bin/python cookbook/02_agents/14_advanced/tool_policy.py
"""

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import ToolPolicy


def search_customer(name: str) -> str:
    """Search the customer database for a customer by name."""
    return f"Found customer profile for {name}"


def get_customer_transactions(customer_id: str) -> str:
    """Get the recent transactions for a customer."""
    return f"Transactions for {customer_id}: 3 recent purchases"


def delete_customer(customer_id: str) -> str:
    """Permanently delete a customer record. Irreversible."""
    return f"Customer {customer_id} deleted"


# ---------------------------------------------------------------------------
# Create Agent with a deterministic tool policy
# ---------------------------------------------------------------------------
def main():
    """Demonstrate allowlist/denylist tool authorization."""
    print("Tool Policy Demo")
    print("=" * 50)

    # Read-only agent: read tools are allowed, destructive tools are blocked.
    # Wildcards are supported, so "search_*" covers every search tool.
    policy = ToolPolicy(
        allowlist=["search_*", "get_customer_transactions"],
        denylist=["delete_*"],
    )

    agent = Agent(
        name="Governed Customer Agent",
        model=OpenAIResponses(id="gpt-5-mini"),
        tools=[search_customer, get_customer_transactions, delete_customer],
        tool_policy=policy,
        instructions="Use the tools to answer questions about customers.",
        show_tool_calls=True,
    )

    # Test 1: Allowed tool call (search_* matches the allowlist)
    print("\n[TEST 1] Search customer - should execute")
    print("-" * 30)
    agent.print_response("Find the customer profile for Alice Johnson")

    # Test 2: Denied tool call (delete_* matches the denylist)
    # The tool is NOT executed; the model sees a denial message and answers
    # without the destructive side effect.
    print("\n[TEST 2] Delete customer - should be denied before execution")
    print("-" * 30)
    agent.print_response("Delete customer CUST-0042")


# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
