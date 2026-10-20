"""Send one structured EvidenceRequest to the FinEvidence A2A server."""

import asyncio
import json
import os
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_data_parts, new_data_message
from a2a.types import Role, SendMessageRequest
import httpx


async def run():
    base_url = os.getenv("A2A_BASE_URL", "http://127.0.0.1:9999")
    payload = {
        "request_id": f"a2a-demo-{uuid4().hex[:8]}",
        "requester": "a2a-demo-client",
        "mode": "stance_research",
        "topic": "传音控股在非洲市场具备哪些竞争优势？",
        "direction": "support",
        "research_focus": ["本地化", "渠道", "产品"],
        "max_deepening_steps": 1,
    }
    async with httpx.AsyncClient(timeout=180.0) as http_client:
        card = await A2ACardResolver(
            httpx_client=http_client,
            base_url=base_url,
        ).get_agent_card()
        client = await create_client(
            agent=card,
            client_config=ClientConfig(
                streaming=True,
                httpx_client=http_client,
            ),
        )
        request = SendMessageRequest(
            message=new_data_message(
                payload,
                media_type="application/vnd.finevidence.evidence-request+json",
                role=Role.ROLE_USER,
            )
        )
        async for response in client.send_message(request):
            if response.HasField("status_update"):
                print(
                    "status:",
                    response.status_update.status.state,
                )
            if response.HasField("artifact_update"):
                data = get_data_parts(
                    response.artifact_update.artifact.parts
                )
                print(json.dumps(data[0], ensure_ascii=False, indent=2))
        await client.close()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
