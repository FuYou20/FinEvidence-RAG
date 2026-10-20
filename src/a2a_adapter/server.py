"""A2A Agent Card and Starlette application construction."""

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from .executor import EvidenceAgentExecutor
from .protocol import (
    EVIDENCE_BUNDLE_MEDIA_TYPE,
    EVIDENCE_REQUEST_MEDIA_TYPE,
)


def build_agent_card(public_url):
    if not isinstance(public_url, str) or not public_url.strip():
        raise ValueError("public_url 必须是非空字符串")
    public_url = public_url.rstrip("/")
    return AgentCard(
        name="FinEvidence",
        description=(
            "面向证券研报的证据调查 Agent：执行混合检索、证据补全、"
            "交叉验证，并在材料不足时返回明确缺口。"
        ),
        version="0.1.0",
        supported_interfaces=[
            AgentInterface(
                url=public_url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        default_input_modes=[
            EVIDENCE_REQUEST_MEDIA_TYPE,
            "application/json",
            "text/plain",
        ],
        default_output_modes=[
            EVIDENCE_BUNDLE_MEDIA_TYPE,
            "application/json",
        ],
        skills=[
            AgentSkill(
                id="investigate_evidence",
                name="调查与验证证券研究证据",
                description=(
                    "接受 stance_research 或 verify_evidence 请求，返回"
                    "可追溯 EvidenceBundle；证据不足时返回缺失信息。"
                ),
                tags=[
                    "agentic-rag",
                    "financial-research",
                    "evidence",
                    "verification",
                ],
                examples=[
                    (
                        '{"request_id":"req-1","requester":"research-agent",'
                        '"mode":"stance_research","topic":"传音控股的非洲竞争优势",'
                        '"direction":"support"}'
                    )
                ],
                input_modes=[
                    EVIDENCE_REQUEST_MEDIA_TYPE,
                    "application/json",
                ],
                output_modes=[EVIDENCE_BUNDLE_MEDIA_TYPE],
            )
        ],
    )


async def _health(_request):
    return JSONResponse({"status": "ok", "agent": "FinEvidence"})


def build_a2a_app(
    request_evidence,
    *,
    public_url="http://127.0.0.1:9999/a2a",
    rpc_path="/a2a",
    max_concurrency=1,
):
    if not isinstance(rpc_path, str) or not rpc_path.startswith("/"):
        raise ValueError("rpc_path 必须是以 / 开头的路径")
    card = build_agent_card(public_url)
    executor = EvidenceAgentExecutor(
        request_evidence,
        max_concurrency=max_concurrency,
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [Route("/health", _health, methods=["GET"])]
    routes.extend(create_agent_card_routes(card))
    routes.extend(create_jsonrpc_routes(handler, rpc_url=rpc_path))
    app = Starlette(routes=routes)
    app.state.agent_card = card
    app.state.evidence_executor = executor
    return app
