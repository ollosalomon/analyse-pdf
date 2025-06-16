# Tests pour le chargement PDF
import os
import fitz
from langchain.schema import Document
from app.ingestion.pdf_loader import process_large_pdf

def test_process_large_pdf(tmp_path):
    # Création d’un petit PDF fictif
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "CHAPITRE 1: Extrait de test")
    doc.save(pdf_path)
    doc.close()

    result = process_large_pdf(str(pdf_path), batch_size=1)
    assert isinstance(result, str)
    assert "Traitement terminé" in result
