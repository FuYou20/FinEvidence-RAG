"""A2A AgentExecutor backed by the existing EvidenceRuntime."""

import asyncio
import logging

from a2a.helpers import (
    new_task_from_user_message,
    new_text_message,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils.errors import UnsupportedOperationError

from src.evidence_protocol import EvidenceBundle

from .protocol import (
    A2AEvidenceRequestError,
    evidence_bundle_to_parts,
    message_to_evidence_request,
)


logger = logging.getLogger(__name__)


class EvidenceAgentExecutor(AgentExecutor):
    """Expose a synchronous evidence gateway as an asynchronous A2A task."""

    def __init__(self, request_evidence, *, max_concurrency=1):
        if not callable(request_evidence):
            raise TypeError("request_evidence 必须是可调用对象")
        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
            or max_concurrency <= 0
        ):
            raise ValueError("max_concurrency 必须是正整数")
        self._request_evidence = request_evidence
        # EvidenceRuntime currently owns mutable registry/metrics state.
        # Keep the default at one until those structures are made thread-safe.
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if context.message is None:
            raise ValueError("A2A RequestContext 缺少 message")
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message(
                "FinEvidence 正在调查并验证证据。",
                context_id=task.context_id,
                task_id=task.id,
            ),
        )
        try:
            request = message_to_evidence_request(context.message)
        except A2AEvidenceRequestError as exc:
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                new_text_message(
                    f"INVALID_ARGUMENT: {exc}",
                    context_id=task.context_id,
                    task_id=task.id,
                ),
            )
            return
        try:
            async with self._semaphore:
                raw_bundle = await asyncio.to_thread(
                    self._request_evidence,
                    request=request,
                )
            bundle = EvidenceBundle.model_validate(raw_bundle)
        except Exception as exc:
            logger.exception(
                "A2A 证据调查失败, request_id=%s",
                request.request_id,
            )
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                new_text_message(
                    f"EVIDENCE_INVESTIGATION_FAILED: {type(exc).__name__}",
                    context_id=task.context_id,
                    task_id=task.id,
                ),
            )
            return
        await updater.add_artifact(
            parts=evidence_bundle_to_parts(bundle),
            name="evidence_bundle",
            metadata={
                "request_id": bundle.request_id,
                "status": bundle.status,
                "mode": bundle.mode,
            },
            last_chunk=True,
        )
        await updater.update_status(
            TaskState.TASK_STATE_COMPLETED,
            new_text_message(
                "证据调查完成。",
                context_id=task.context_id,
                task_id=task.id,
            ),
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        del context, event_queue
        raise UnsupportedOperationError(
            message="FinEvidence 暂不支持取消正在运行的同步调查"
        )
