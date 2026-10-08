"""Run chat_with_pdf.py with Couchbase and Gemini swapped for in-memory fakes.

Streamlit executes this file instead of the app so the UI can be smoke tested
without a Couchbase cluster or a GOOGLE_API_KEY. Only the external services are
faked; the app code, Streamlit, LangChain and pypdf run for real.
"""

import os
import runpy
from pathlib import Path

import couchbase.cluster
import langchain_couchbase
import langchain_google_genai
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeStreamingListLLM
from langchain_core.vectorstores import InMemoryVectorStore

APP = Path(__file__).resolve().parent.parent / "chat_with_pdf.py"

# Placeholders only, so the app's environment checks pass.
for name in (
    "GOOGLE_API_KEY",
    "DB_CONN_STR",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_BUCKET",
    "DB_SCOPE",
    "DB_COLLECTION",
    "INDEX_NAME",
):
    os.environ[name] = "smoke-placeholder"


class FakeCluster:
    def __init__(self, connection_string, options):
        pass

    def wait_until_ready(self, timeout):
        pass


def fake_embeddings(model, **kwargs):
    return DeterministicFakeEmbedding(size=768)


def fake_llm(model, **kwargs):
    return FakeStreamingListLLM(responses=[f"Smoke answer from {model}"])


def fake_vector_store(embedding, **kwargs):
    return InMemoryVectorStore(embedding=embedding)


couchbase.cluster.Cluster = FakeCluster
langchain_google_genai.GoogleGenerativeAIEmbeddings = fake_embeddings
langchain_google_genai.GoogleGenerativeAI = fake_llm
langchain_couchbase.CouchbaseSearchVectorStore = fake_vector_store

runpy.run_path(str(APP), run_name="__main__")
