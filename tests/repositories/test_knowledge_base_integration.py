"""Retrieval smoke test with the real Giga-Embeddings model.

Gated twice: the ``integration`` marker and the ``RUN_INTEGRATION=1`` env var,
because the model is a ~14 GB download and wants a GPU. Run with:

    RUN_INTEGRATION=1 uv run pytest -m integration -q
"""

import os

import pytest

from src.config import get_settings
from src.rag.embedder import GigaEmbedder
from src.repositories.knowledge_base import FaissKnowledgeBase


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run the real-embedder test (~14 GB model, GPU)",
    ),
]


# The model's trust_remote_code forward path calls torch.backends.cuda.sdp_kernel(),
# deprecated since torch 2.3 → FutureWarning. That is third-party remote model code
# we cannot fix, so the narrowest possible ignore goes on this test only;
# filterwarnings=["error"] stays in force for everything else.
@pytest.mark.filterwarnings("ignore:.*sdp_kernel.*:FutureWarning")
async def test_leave_question_retrieves_leave_policy_with_real_embedder():
    settings = get_settings()
    embedder = GigaEmbedder(
        settings.embedding_model_name,
        device="auto",
        hf_token=settings.hf_token,
        batch_size=settings.embedding_batch_size,
    )
    knowledge_base = FaissKnowledgeBase(embedder, settings.data_dir)

    await knowledge_base.load()
    results = await knowledge_base.search(
        "За сколько дней нужно предупреждать об отпуске?", k=3
    )

    assert len(results) == 3
    assert results[0].chunk.source == "leave_policy.md"
    # min_retrieval_score in src/config.py was calibrated on this corpus:
    # legitimate questions score >= 0.58 with this model.
    assert results[0].score >= 0.5
