from types import SimpleNamespace

import pytest

from chunkbuster import Query
from chunkbuster.core.models import as_query


@pytest.mark.parametrize("metadata", [{"market": "EC"}, SimpleNamespace(market="EC")])
def test_query_preserves_metadata_and_instance(metadata):
    query = Query("query", metadata=metadata)

    assert query.metadata is metadata
    assert as_query(query) is query


def test_string_query_is_normalized_and_default_metadata_is_private():
    first = as_query("query")
    second = as_query("query")

    assert first.text == "query"
    assert first.metadata == {}
    assert first.metadata is not second.metadata
