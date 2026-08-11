"""The one test that loads the real embedding model.

Everything else in the suite substitutes ``FakeEmbedder``, a bag-of-stems
stand-in that can prove the retrieval *wiring* is right but says nothing about
whether ``ai-sage/Giga-Embeddings-instruct`` actually ranks this corpus
sensibly — the fake matches literal stems, the model matches meaning. This test
is the one that answers that question, and it costs a 3B model (~13 GB) plus a
GPU to answer it, so it is gated off by default.

Run it with::

    RUN_INTEGRATION=1 uv run pytest -m integration -q

Expect roughly a minute: the model loads lazily on the first ``embed``.
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
        reason="set RUN_INTEGRATION=1 to run the real-embedder test (~13 GB model, GPU)",
    ),
]

PAYMENTS_QUESTION = "Хочу оплатить подписку другой банковской картой, что делать?"
RECOVERY_QUESTION = "Потерял доступ к аккаунту и забыл пароль, как его вернуть?"


# ``Giga-Embeddings-instruct`` ships remote code that calls the deprecated
# ``torch.backends.cuda.sdp_kernel()``. Warnings are errors suite-wide, and that
# is the right default — the exemption is scoped to the one test that loads it.
@pytest.mark.filterwarnings("ignore:.*sdp_kernel.*:FutureWarning")
async def test_the_real_model_ranks_the_real_corpus_by_meaning() -> None:
    settings = get_settings()
    # Built directly rather than through ``get_embedder()``: this test wants the
    # real model regardless of how the application happens to be wired.
    embedder = GigaEmbedder(
        settings.embedding_model_name,
        device=settings.embedding_device,
        hf_token=settings.hf_token,
        batch_size=settings.embedding_batch_size,
        max_length=settings.embedding_max_length,
    )
    knowledge_base = FaissKnowledgeBase(embedder, settings.knowledge_base_dir)

    await knowledge_base.load()
    payments = await knowledge_base.search(PAYMENTS_QUESTION, k=settings.retrieval_top_k)
    recovery = await knowledge_base.search(RECOVERY_QUESTION, k=settings.retrieval_top_k)

    assert await knowledge_base.ping() is True
    assert payments[0].chunk.source == "payment_methods.md", (
        f"top hit was {payments[0].chunk.source}:{payments[0].chunk.heading} "
        f"at {payments[0].score:.3f}"
    )
    assert recovery[0].chunk.source == "account_security.md", (
        f"top hit was {recovery[0].chunk.source}:{recovery[0].chunk.heading} "
        f"at {recovery[0].score:.3f}"
    )
