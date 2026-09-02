"""Tool policy enforcement at the model execution chokepoint and in a full agent run.

A denied tool call must never execute its entrypoint, must not emit a
tool-call-started event, and must produce a structured denial message the
model can see. An allowed call behaves as before.
"""

import json
from typing import Any, AsyncIterator, Iterator, List

import pytest

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse, ModelResponseEvent
from agno.tools.function import Function, FunctionCall
from agno.tools.policy import ToolPolicy


class SequenceModel(Model):
    """Offline model that replays canned responses (tool calls, then text)."""

    def __init__(self, responses: List[ModelResponse]):
        super().__init__(id="test-model", name="test-model", provider="test")
        self.instructions = None
        self._responses = list(responses)

    def get_instructions_for_model(self, *args, **kwargs):
        return None

    def get_system_message_for_model(self, *args, **kwargs):
        return None

    async def aget_instructions_for_model(self, *args, **kwargs):
        return None

    async def aget_system_message_for_model(self, *args, **kwargs):
        return None

    def parse_args(self, *args, **kwargs):
        return {}

    def invoke(self, *args, **kwargs) -> ModelResponse:
        return self._responses.pop(0)

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        return self._responses.pop(0)

    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        yield self._responses.pop(0)

    async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        yield self._responses.pop(0)
        return

    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


def _tool_call(tool_name: str, call_id: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": tool_name, "arguments": json.dumps(args)},
    }


def _agent_with_policy(**policy_kwargs) -> Agent:
    return Agent(name="policy-test", tool_policy=ToolPolicy(**policy_kwargs))


def test_denied_tool_call_is_not_executed_sync():
    executed: List[int] = []

    def dangerous_tool(value: int) -> str:
        executed.append(value)
        return f"executed {value}"

    function = Function.from_callable(dangerous_tool, name="dangerous_tool")
    agent = _agent_with_policy(denylist=["dangerous_*"])
    function._agent = agent  # what tool preparation does before model.run
    function_call = FunctionCall(function=function, arguments={"value": 1}, call_id="call_1")

    model = SequenceModel([])
    results: List[Message] = []
    events = list(model.run_function_call(function_call=function_call, function_call_results=results))

    assert executed == []  # never executed
    assert events == []  # no tool-call-started event for a denied call
    assert len(results) == 1
    denial = results[0]
    assert denial.tool_call_error is True
    assert denial.tool_name == "dangerous_tool"
    assert denial.tool_call_id == "call_1"
    assert denial.content is not None
    assert "denied" in denial.content
    assert "dangerous_tool" in denial.content


def test_allowed_tool_call_executes_sync():
    executed: List[int] = []

    def safe_tool(value: int) -> str:
        executed.append(value)
        return f"executed {value}"

    function = Function.from_callable(safe_tool, name="safe_tool")
    agent = _agent_with_policy(allowlist=["safe_*"])
    function._agent = agent
    function_call = FunctionCall(function=function, arguments={"value": 2}, call_id="call_2")

    model = SequenceModel([])
    results: List[Message] = []
    events = list(model.run_function_call(function_call=function_call, function_call_results=results))

    assert executed == [2]
    assert len(events) == 2  # tool-call-started + tool-call-completed events still emitted
    assert events[0].event == ModelResponseEvent.tool_call_started.value
    assert events[1].event == ModelResponseEvent.tool_call_completed.value
    assert len(results) == 1
    assert results[0].tool_call_error is False
    assert results[0].content is not None
    assert "executed 2" in results[0].content


@pytest.mark.asyncio
async def test_denied_tool_call_is_not_executed_async():
    executed: List[int] = []

    def dangerous_tool(value: int) -> str:
        executed.append(value)
        return f"executed {value}"

    function = Function.from_callable(dangerous_tool, name="dangerous_tool")
    agent = _agent_with_policy(denylist=["dangerous_*"])
    function._agent = agent
    function_call = FunctionCall(function=function, arguments={"value": 3}, call_id="call_3")

    model = SequenceModel([])
    success, _timer, fc, _result = await model.arun_function_call(function_call=function_call)

    assert executed == []
    assert success is False
    assert fc.error is not None
    assert "denied" in fc.error


def test_agent_run_denies_tool_call_and_completes():
    denied_executions: List[int] = []
    allowed_executions: List[int] = []

    def dangerous_tool(value: int) -> str:
        denied_executions.append(value)
        return f"danger {value}"

    def safe_tool(value: int) -> str:
        allowed_executions.append(value)
        return f"safe {value}"

    model = SequenceModel(
        [
            ModelResponse(
                content="",
                role="assistant",
                tool_calls=[
                    _tool_call("dangerous_tool", "call_1", {"value": 1}),
                    _tool_call("safe_tool", "call_2", {"value": 2}),
                ],
            ),
            ModelResponse(content="final answer", role="assistant"),
        ]
    )
    agent = Agent(
        name="policy-e2e",
        model=model,
        db=InMemoryDb(),
        telemetry=False,
        tool_policy=ToolPolicy(denylist=["dangerous_*"]),
        tools=[dangerous_tool, safe_tool],
    )

    run = agent.run("use the tools")

    assert denied_executions == []  # policy vetoed before execution
    assert allowed_executions == [2]
    assert run.content == "final answer"
    # The denial is fed back to the model as a tool error message
    assert any(m.tool_name == "dangerous_tool" and m.tool_call_error for m in (run.messages or []))


def test_denied_confirmation_tool_is_not_paused():
    """A policy-denied call must never pause for approval: the policy veto is
    authoritative even when the tool is marked requires_confirmation."""

    def dangerous_tool(value: int) -> str:
        return f"executed {value}"

    function = Function.from_callable(dangerous_tool, name="dangerous_tool")
    function.requires_confirmation = True
    agent = _agent_with_policy(denylist=["dangerous_*"])
    function._agent = agent
    function_call = FunctionCall(function=function, arguments={"value": 1}, call_id="call_6")

    model = SequenceModel([])
    results: List[Message] = []
    events = list(model.run_function_calls(function_calls=[function_call], function_call_results=results))

    assert events == []  # no ToolCallPaused, no ToolCallStarted
    assert len(results) == 1
    assert results[0].tool_call_error is True
    assert results[0].content is not None
    assert "denied" in results[0].content


@pytest.mark.asyncio
async def test_denied_tool_call_emits_no_started_event_async():
    """Async path: a denied call must not emit ToolCallStarted before denial."""

    def dangerous_tool(value: int) -> str:
        return f"executed {value}"

    function = Function.from_callable(dangerous_tool, name="dangerous_tool")
    agent = _agent_with_policy(denylist=["dangerous_*"])
    function._agent = agent
    function_call = FunctionCall(function=function, arguments={"value": 1}, call_id="call_7")

    model = SequenceModel([])
    results: List[Message] = []
    events = []
    async for event in model.arun_function_calls(function_calls=[function_call], function_call_results=results):
        events.append(event)

    assert all(
        getattr(e, "event", None) != ModelResponseEvent.tool_call_started.value for e in events
    )
    assert len(results) == 1
    assert results[0].tool_call_error is True
    assert results[0].content is not None
    assert "denied" in results[0].content


def test_framework_offload_read_back_tool_is_exempt_from_policy():
    """Framework read-back tools (provenance-marked) stay callable under a
    strict allowlist, mirroring their tool-call-limit exemption."""

    def read_result(result_id: str) -> str:
        return f"result {result_id}"

    function = Function.from_callable(read_result, name="read_result")
    function._is_offload_read_back = True  # marker set by offload/tools.py factories
    agent = _agent_with_policy(allowlist=["search_*"])
    function._agent = agent
    function_call = FunctionCall(function=function, arguments={"result_id": "r1"}, call_id="call_5")

    model = SequenceModel([])
    results: List[Message] = []
    list(model.run_function_call(function_call=function_call, function_call_results=results))

    assert len(results) == 1
    assert results[0].tool_call_error is False


def test_user_tool_named_read_result_is_denied():
    """A user-defined tool named like a framework read-back tool must NOT be
    exempt: the exemption is provenance-based, not name-based."""

    def read_result(result_id: str) -> str:
        return f"result {result_id}"

    function = Function.from_callable(read_result, name="read_result")  # no marker
    agent = _agent_with_policy(allowlist=["search_*"])
    function._agent = agent
    function_call = FunctionCall(function=function, arguments={"result_id": "r1"}, call_id="call_8")

    model = SequenceModel([])
    results: List[Message] = []
    list(model.run_function_call(function_call=function_call, function_call_results=results))

    assert len(results) == 1
    assert results[0].tool_call_error is True
    assert results[0].content is not None
    assert "denied" in results[0].content


def test_agent_streaming_run_completes_after_denial():
    """Regression: a denied tool marked requires_confirmation must not end a
    streaming run early - the model gets the denial and answers."""

    def dangerous_tool(value: int) -> str:
        return f"danger {value}"

    denied_calls: List[int] = []

    model = SequenceModel(
        [
            ModelResponse(
                content="",
                role="assistant",
                tool_calls=[_tool_call("dangerous_tool", "call_s2", {"value": 1})],
            ),
            ModelResponse(content="streaming final answer", role="assistant"),
        ]
    )
    # Mark the tool requires_confirmation so the streaming loop must not break
    # on the original call flags after the policy denied it
    agent = Agent(
        name="policy-stream",
        model=model,
        db=InMemoryDb(),
        telemetry=False,
        tool_policy=ToolPolicy(denylist=["dangerous_*"]),
    )
    # Register the tool through a function object with the confirmation flag
    function = Function.from_callable(dangerous_tool, name="dangerous_tool")
    function.requires_confirmation = True
    agent.tools = [function]

    run = agent.run("use the tools", stream=True)

    assert denied_calls == []
    streamed_content = "".join(
        str(getattr(e, "content", "") or "") for e in run if getattr(e, "event", "") == "RunContent"
    )
    assert "streaming final answer" in streamed_content


@pytest.mark.asyncio
async def test_agent_astreaming_run_completes_after_denial():
    """Async streaming parity: same as test_agent_streaming_run_completes_after_denial."""

    def dangerous_tool(value: int) -> str:
        return f"danger {value}"

    denied_calls: List[int] = []

    model = SequenceModel(
        [
            ModelResponse(
                content="",
                role="assistant",
                tool_calls=[_tool_call("dangerous_tool", "call_s3", {"value": 1})],
            ),
            ModelResponse(content="async streaming final answer", role="assistant"),
        ]
    )
    agent = Agent(
        name="policy-astream",
        model=model,
        db=InMemoryDb(),
        telemetry=False,
        tool_policy=ToolPolicy(denylist=["dangerous_*"]),
    )
    function = Function.from_callable(dangerous_tool, name="dangerous_tool")
    function.requires_confirmation = True
    agent.tools = [function]

    assert denied_calls == []
    streamed_content = ""
    async for e in agent.arun("use the tools", stream=True):
        if getattr(e, "event", "") == "RunContent":
            streamed_content += str(getattr(e, "content", "") or "")
    assert "async streaming final answer" in streamed_content


def test_no_policy_leaves_tool_execution_unchanged():
    executed: List[int] = []

    def plain_tool(value: int) -> str:
        executed.append(value)
        return f"executed {value}"

    function = Function.from_callable(plain_tool, name="plain_tool")
    agent = Agent(name="no-policy")
    function._agent = agent
    function_call = FunctionCall(function=function, arguments={"value": 4}, call_id="call_4")

    model = SequenceModel([])
    results: List[Message] = []
    list(model.run_function_call(function_call=function_call, function_call_results=results))

    assert executed == [4]
    assert len(results) == 1
    assert results[0].tool_call_error is False
