"""Optional live Gemini smoke test. Needs GOOGLE_API_KEY; Couchbase is not used.

The model names are read from chat_with_pdf.py, so this checks the exact
models the app calls.
"""

import ast
import math
import os
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "chat_with_pdf.py"

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY is not set; skipping live Gemini provider smoke test",
)


def _app_models(class_name):
    """Return the model= string literals passed to `class_name` in the app."""
    models = []
    for node in ast.walk(ast.parse(APP.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == class_name
        ):
            for kw in node.keywords:
                if kw.arg == "model" and isinstance(kw.value, ast.Constant):
                    models.append(kw.value.value)
    assert models, f"no {class_name}(model=...) call found in {APP.name}"
    return models


@pytest.mark.parametrize("model", _app_models("GoogleGenerativeAIEmbeddings"))
def test_embeddings(model):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    vector = GoogleGenerativeAIEmbeddings(model=model).embed_query("Couchbase")
    assert len(vector) > 0
    assert all(isinstance(x, float) and math.isfinite(x) for x in vector)


@pytest.mark.parametrize("model", _app_models("GoogleGenerativeAI"))
def test_llm(model):
    from langchain_google_genai import GoogleGenerativeAI

    llm = GoogleGenerativeAI(model=model, temperature=0, max_retries=1)
    response = llm.invoke("Reply with the single word: pong")
    assert response.strip()
