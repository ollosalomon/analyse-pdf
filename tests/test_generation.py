# Tests pour la génération de rapport
import pytest
from unittest.mock import MagicMock
from app.report.generator import ReportGenerator

def test_report_generator_minimal_structure():
    vector_db_mock = MagicMock()
    vector_db_mock.hybrid_search.return_value = []
    vector_db_mock.db.similarity_search.return_value = []

    generator = ReportGenerator(vector_db_mock, llm_model=MagicMock())
    structure = generator.extract_document_structure()
    
    assert isinstance(structure, dict)
    assert "chapters" in structure
