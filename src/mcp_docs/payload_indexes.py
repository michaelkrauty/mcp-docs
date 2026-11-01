"""Payload indexes on the document collection."""

from qdrant_client.models import (
    PayloadSchemaType,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
)
from vector_core import PayloadIndexSchema

# keyword_search matches whole words and phrases case-insensitively. Qdrant
# only does that on a field with a full-text index: without one, text and
# phrase conditions fall back to a case-sensitive substring scan. Phrase
# matching must be enabled on the index, or phrase conditions match nothing.
KEYWORD_TEXT_INDEX = TextIndexParams(
    type=TextIndexType.TEXT,
    tokenizer=TokenizerType.WORD,
    lowercase=True,
    phrase_matching=True,
)

KEYWORD_SEARCH_INDEXES: list[tuple[str, PayloadIndexSchema]] = [
    ("content", KEYWORD_TEXT_INDEX),
    ("filename", KEYWORD_TEXT_INDEX),
]

DOCUMENT_PAYLOAD_INDEXES: list[tuple[str, PayloadIndexSchema]] = [
    ("type", PayloadSchemaType.KEYWORD),
    ("document_id", PayloadSchemaType.KEYWORD),
    ("content_hash", PayloadSchemaType.KEYWORD),
    ("doc_type", PayloadSchemaType.KEYWORD),
    ("tags", PayloadSchemaType.KEYWORD),
    *KEYWORD_SEARCH_INDEXES,
]
