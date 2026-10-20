import asyncio
import json
from types import SimpleNamespace

from a2a.helpers import (
    get_data_parts,
    new_data_message,
    new_text_message,
)
from a2a.types import (
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
import pytest

from src.a2a_adapter.executor import EvidenceAgentExecutor
from src.a2a_adapter.client import (
    A2AEvidenceClient,
    A2AEvidenceClientError,
)
from src.a2a_adapter.protocol import (
    A2AEvidenceRequestError,
    message_to_evidence_request,
)
from src.a2a_adapter.server import build_a2a_app, build_agent_card


def request_payload():
    return {
        "request_id": "req-a2a-1",
        "requester": "research-agent",
        "mode": "stance_research",
        "topic": "竞争优势是否持续",
        "direction": "support",
        "research_focus": ["渠道", "产品"],
        "max_deepening_steps": 1,
    }


def bundle_payload():
    return {
        "request_id": "req-a2a-1",
        "mode": "stance_research",
        "status": "sufficient",
        "evidence": [{
            "evidence_id": "E-local-a2a",
            "source_kind": "local",
            "doc_id": "report-1",
            "chunk_id": "report-1:p1:c0",
            "text": "可追溯证据",
        }],
    }


class CollectingEventQueue:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


def test_message_adapter_prefers_structured_data_part():
    message = new_data_message(
        request_payload(),
        role=Role.ROLE_USER,
    )

    request = message_to_evidence_request(message)

    assert request.request_id == "req-a2a-1"
    assert request.research_focus == ["渠道", "产品"]


def test_message_adapter_accepts_json_text_for_debugging():
    message = new_text_message(
        json.dumps(request_payload(), ensure_ascii=False),
        role=Role.ROLE_USER,
    )

    request = message_to_evidence_request(message)

    assert request.mode == "stance_research"


def test_message_adapter_rejects_plain_natural_language():
    message = new_text_message("帮我找证据", role=Role.ROLE_USER)

    with pytest.raises(A2AEvidenceRequestError, match="合法"):
        message_to_evidence_request(message)


def test_executor_returns_evidence_bundle_as_data_artifact():
    captured = []

    def request_evidence(*, request):
        captured.append(request)
        return bundle_payload()

    message = new_data_message(
        request_payload(),
        role=Role.ROLE_USER,
    )
    context = SimpleNamespace(message=message, current_task=None)
    queue = CollectingEventQueue()
    executor = EvidenceAgentExecutor(request_evidence)

    asyncio.run(executor.execute(context, queue))

    assert len(captured) == 1
    assert isinstance(queue.events[0], Task)
    assert isinstance(queue.events[1], TaskStatusUpdateEvent)
    assert queue.events[1].status.state == TaskState.TASK_STATE_WORKING
    assert isinstance(queue.events[2], TaskArtifactUpdateEvent)
    result = get_data_parts(queue.events[2].artifact.parts)[0]
    assert result["request_id"] == "req-a2a-1"
    assert result["evidence"][0]["evidence_id"] == "E-local-a2a"
    assert queue.events[3].status.state == TaskState.TASK_STATE_COMPLETED


def test_executor_marks_invalid_request_failed_without_calling_runtime():
    calls = []
    executor = EvidenceAgentExecutor(
        lambda **kwargs: calls.append(kwargs)
    )
    context = SimpleNamespace(
        message=new_text_message("not-json", role=Role.ROLE_USER),
        current_task=None,
    )
    queue = CollectingEventQueue()

    asyncio.run(executor.execute(context, queue))

    assert calls == []
    assert queue.events[-1].status.state == TaskState.TASK_STATE_FAILED


def test_agent_card_and_routes_advertise_evidence_skill():
    card = build_agent_card("http://127.0.0.1:9999/a2a")
    app = build_a2a_app(
        lambda **_kwargs: bundle_payload(),
        public_url="http://127.0.0.1:9999/a2a",
    )

    assert card.name == "FinEvidence"
    assert card.skills[0].id == "investigate_evidence"
    assert card.supported_interfaces[0].protocol_binding == "JSONRPC"
    paths = {route.path for route in app.routes}
    assert "/.well-known/agent-card.json" in paths
    assert "/a2a" in paths
    assert "/health" in paths


def test_client_adapter_validates_and_records_remote_bundle():
    captured = []

    async def transport(request):
        captured.append(request)
        return bundle_payload()

    client = A2AEvidenceClient(
        "http://127.0.0.1:9999",
        transport=transport,
    )

    bundle = client.request_evidence(request=request_payload())

    assert captured[0].request_id == "req-a2a-1"
    assert bundle.status == "sufficient"
    assert "E-local-a2a" in client.registry.snapshot()
    assert client.snapshot_metrics()["success_count"] == 1
    assert client.snapshot_requests()[0]["result"]["mode"] == "stance_research"


def test_client_adapter_rejects_mismatched_response_identity():
    async def transport(_request):
        payload = bundle_payload()
        payload["request_id"] = "wrong-request"
        return payload

    client = A2AEvidenceClient(
        "http://127.0.0.1:9999",
        transport=transport,
    )

    with pytest.raises(A2AEvidenceClientError, match="request_id"):
        client.request_evidence(request=request_payload())

    assert client.snapshot_metrics()["failure_count"] == 1


def test_client_sync_api_refuses_to_block_running_event_loop():
    client = A2AEvidenceClient(
        "http://127.0.0.1:9999",
        transport=lambda _request: None,
    )

    async def call_sync_api():
        with pytest.raises(RuntimeError, match="async_request_evidence"):
            client.request_evidence(request=request_payload())

    asyncio.run(call_sync_api())
