"""Unit tests for ToolPolicy — deterministic tool-call authorization."""

from agno.tools.policy import ToolPolicy


def test_no_policy_allows_all_tools():
    policy = ToolPolicy()
    assert policy.check("search_docs") is None
    assert policy.check("delete_account") is None
    assert policy.check("anything") is None


def test_allowlist_allows_only_listed_tools():
    policy = ToolPolicy(allowlist=["search_docs", "get_customer"])
    assert policy.check("search_docs") is None
    assert policy.check("get_customer") is None
    reason = policy.check("delete_account")
    assert reason is not None
    assert "delete_account" in reason
    assert "allowlist" in reason


def test_denylist_blocks_listed_tools():
    policy = ToolPolicy(denylist=["delete_account", "transfer_funds"])
    assert policy.check("search_docs") is None
    reason = policy.check("transfer_funds")
    assert reason is not None
    assert "transfer_funds" in reason
    assert "denylist" in reason


def test_wildcard_allowlist_patterns():
    policy = ToolPolicy(allowlist=["search_*", "list_*"])
    assert policy.check("search_docs") is None
    assert policy.check("list_customers") is None
    assert policy.check("delete_account") is not None


def test_wildcard_denylist_patterns():
    policy = ToolPolicy(denylist=["delete_*"])
    assert policy.check("search_docs") is None
    assert policy.check("delete_account") is not None
    assert policy.check("delete_all_customers") is not None


def test_denylist_wins_over_allowlist():
    policy = ToolPolicy(allowlist=["*"], denylist=["delete_*"])
    assert policy.check("search_docs") is None
    reason = policy.check("delete_account")
    assert reason is not None
    assert "denylist" in reason


def test_pattern_matching_is_case_sensitive():
    policy = ToolPolicy(denylist=["Delete_Account"])
    assert policy.check("Delete_Account") is not None
    assert policy.check("delete_account") is None


def test_empty_allowlist_denies_everything():
    # An allowlist that is set but empty denies all tools: explicit and safe.
    policy = ToolPolicy(allowlist=[])
    assert policy.check("search_docs") is not None


def test_deny_message_can_be_customized():
    policy = ToolPolicy(denylist=["delete_account"], deny_message="Blocked by compliance policy.")
    reason = policy.check("delete_account")
    assert reason is not None
    assert reason.startswith("Blocked by compliance policy.")
