"""No-secret unit tests for the embedding adapter in chat_with_pdf.py."""

import json
import math
from pathlib import Path

import pytest

from chat_with_pdf import EMBEDDING_DIMENSIONS, FixedDimensionEmbeddings

INDEX = Path(__file__).resolve().parent.parent / "demoSearchIndex.json"


class RecordingEmbeddings:
    """Stands in for GoogleGenerativeAIEmbeddings and records the size requested."""

    def __init__(self, vector):
        self.vector = vector
        self.requested = []

    def embed_documents(self, texts, output_dimensionality=None):
        self.requested.append(output_dimensionality)
        return [list(self.vector) for _ in texts]

    def embed_query(self, text, output_dimensionality=None):
        self.requested.append(output_dimensionality)
        return list(self.vector)


def _norm(vector):
    return math.sqrt(sum(x * x for x in vector))


def _index_vector_dims():
    """Return the dims of every vector field in the Search index definition."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "vector" and "dims" in node:
                found.append(node["dims"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(INDEX.read_text()))
    return found


def test_dimensions_match_search_index():
    assert _index_vector_dims() == [EMBEDDING_DIMENSIONS]


def test_embed_query_requests_size_and_normalizes():
    inner = RecordingEmbeddings([3.0, 4.0])
    vector = FixedDimensionEmbeddings(inner, EMBEDDING_DIMENSIONS).embed_query("q")
    assert inner.requested == [EMBEDDING_DIMENSIONS]
    assert vector == pytest.approx([0.6, 0.8])


def test_embed_documents_requests_size_and_normalizes():
    inner = RecordingEmbeddings([1.0, 2.0, 2.0])
    vectors = FixedDimensionEmbeddings(inner, EMBEDDING_DIMENSIONS).embed_documents(
        ["a", "b"]
    )
    assert inner.requested == [EMBEDDING_DIMENSIONS]
    assert len(vectors) == 2
    for vector in vectors:
        assert _norm(vector) == pytest.approx(1.0)


def test_zero_vector_is_returned_unchanged():
    inner = RecordingEmbeddings([0.0, 0.0])
    assert FixedDimensionEmbeddings(inner, 2).embed_query("q") == [0.0, 0.0]
