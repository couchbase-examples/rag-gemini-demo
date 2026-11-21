"""Optional live Gemini smoke test. Needs GOOGLE_API_KEY; Couchbase is not used.

The models and the embedding setup come from chat_with_pdf.py, so this checks
exactly what the app calls.
"""

import math
import os

import pytest

from chat_with_pdf import EMBEDDING_DIMENSIONS, LLM_MODEL, create_embeddings

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY is not set; skipping live Gemini provider smoke test",
)


def test_embeddings():
    vector = create_embeddings().embed_query("Couchbase")
    # Must match the Search index's vector field (demoSearchIndex.json)
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert all(isinstance(x, float) and math.isfinite(x) for x in vector)
    # The index uses dot_product, so vectors must be unit length.
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-6)


def test_llm():
    from langchain_google_genai import GoogleGenerativeAI

    llm = GoogleGenerativeAI(model=LLM_MODEL, temperature=0, max_retries=1)
    response = llm.invoke("Reply with the single word: pong")
    assert response.strip()
