"""Start FinEvidence as an A2A 1.0 JSON-RPC server."""

import os

from dotenv import load_dotenv
import uvicorn

from src.a2a_adapter.server import build_a2a_app
from src.a2a_adapter.service import (
    PROJECT_ROOT,
    create_evidence_service_from_env,
)


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    host = os.getenv("A2A_HOST", "127.0.0.1")
    port = int(os.getenv("A2A_PORT", "9999"))
    public_url = os.getenv(
        "A2A_PUBLIC_URL",
        f"http://127.0.0.1:{port}/a2a",
    )
    max_concurrency = int(os.getenv("A2A_MAX_CONCURRENCY", "1"))
    service = create_evidence_service_from_env()
    app = build_a2a_app(
        service.request_evidence,
        public_url=public_url,
        max_concurrency=max_concurrency,
    )
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        service.close()


if __name__ == "__main__":
    main()
