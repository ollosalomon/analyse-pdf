# Tests pour le découpage du texte
import pytest
from langchain.schema import Document
from app.ingestion.chunker import create_hierarchical_chunks

def test_chunking_simple_document():
    doc = Document(page_content="CHAPITRE 1: Introduction\n1.1. Contexte\n\nTexte ici.", metadata={"page": 1})
    chunks = create_hierarchical_chunks([doc])
    assert len(chunks) >= 1
    assert isinstance(chunks[0].page_content, str)
    assert "chapter" in chunks[0].metadata["type"]
