"""A2A protocol adapter for the FinEvidence service."""

from .executor import EvidenceAgentExecutor
from .server import build_agent_card, build_a2a_app

__all__ = [
    "EvidenceAgentExecutor",
    "build_agent_card",
    "build_a2a_app",
]
