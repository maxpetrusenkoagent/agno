"""Deterministic tool-call authorization policy for agents.

A ``ToolPolicy`` is evaluated synchronously before a tool executes. It never
calls an LLM: allowlist/denylist matching is plain ``fnmatch`` pattern
matching, so the decision is deterministic and fast (<1ms).

Example:
    from agno.tools import ToolPolicy

    policy = ToolPolicy(
        allowlist=["search_*", "get_customer"],
        denylist=["delete_*", "transfer_funds"],
    )

    agent = Agent(..., tools=[...], tool_policy=policy)

Rules:
- ``allowlist``: only tools matching one of these patterns may execute.
  When set to an empty list, every tool call is denied.
- ``denylist``: tools matching one of these patterns never execute.
  A denylist match always wins over an allowlist match.
- When both lists are unset the policy is a no-op and every tool runs.
"""

import fnmatch
from typing import List, Optional

__all__ = ["ToolPolicy"]


class ToolPolicy:
    """Allowlist/denylist authorization evaluated before tool execution.

    Args:
        allowlist: Optional list of tool-name patterns that are permitted.
            Supports ``fnmatch`` wildcards (``*``, ``?``, ``[...]``).
        denylist: Optional list of tool-name patterns that are forbidden.
            A denylist match denies the call even when the allowlist matches.
        deny_message: Message prefix attached to structured denial reasons.
    """

    def __init__(
        self,
        allowlist: Optional[List[str]] = None,
        denylist: Optional[List[str]] = None,
        deny_message: str = "Tool call denied by agent tool policy.",
    ) -> None:
        self.allowlist = list(allowlist) if allowlist is not None else None
        self.denylist = list(denylist) if denylist is not None else None
        self.deny_message = deny_message

    def check(self, tool_name: str) -> Optional[str]:
        """Return a denial reason if ``tool_name`` is not permitted, else None.

        The reason names the tool and the list that denied it so callers can
        surface structured, auditable denials.
        """
        if self.denylist is not None and any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.denylist):
            return f"{self.deny_message} Tool '{tool_name}' matches the policy denylist."
        if self.allowlist is not None and not any(
            fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.allowlist
        ):
            return f"{self.deny_message} Tool '{tool_name}' is not in the policy allowlist."
        return None

    def to_dict(self) -> dict:
        """Serialize to a dict for agent configuration round-trips."""
        return {
            "allowlist": self.allowlist,
            "denylist": self.denylist,
            "deny_message": self.deny_message,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["ToolPolicy"]:
        """Reconstruct a ToolPolicy from ``to_dict`` output (or None)."""
        if data is None:
            return None
        return cls(
            allowlist=data.get("allowlist"),
            denylist=data.get("denylist"),
            deny_message=data.get("deny_message", "Tool call denied by agent tool policy."),
        )

    def __repr__(self) -> str:
        return f"ToolPolicy(allowlist={self.allowlist}, denylist={self.denylist})"
