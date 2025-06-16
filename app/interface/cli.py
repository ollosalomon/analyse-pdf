# Interface ligne de commande - Version corrigée
import os
import argparse
from app.ingestion.chunker import create_hierarchical_chunks
from app.ingestion.image_extractor import extract_images_from_page
from app.vectorstore.embeddings import get_embedding_model
from app.vectorstore.chroma_db import EnhancedVectorDB
from app.report.generator import ReportGenerator
from langchain.schema import Document
import fitz
import json
from datetime import datetime
import logging
from tqdm import tqdm
import time
import traceback

logger = logging.getLogger("pdf_cli")
logging.basicConfig(level=logging.INFO)

def ensure_directory_structure(output_dir):
    """Crée la structure de dossiers nécessaire"""
    directories = [
        output_dir,
        os.path.join(output_dir, "extracted_text"),
        os.path.join(output_dir, "extracted_images"),
        os.path.join(output_dir, "metadata"),
        os.path.join(output_dir, "vector_db")
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Dossier créé/vérifié: {directory}")
        except Exception as e:
            logger.error(f"Erreur création dossier {directory}: {e}")
            raise

def save_text_content(text, page_num, output_dir):
    """Sauvegarde le contenu textuel d'une page"""
    try:
        logger.info(f"Texte à sauvegarder: {text}")
        
        text_file = os.path.join(output_dir, "extracted_text", f"page_{page_num}.txt")
        logger.info(f"text_file__text_file: {text_file}")
        
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)
        logger.debug(f"Texte sauvegardé: {text_file}")
        return True
    except Exception as e:
        logger.error(f"Erreur sauvegarde texte page {page_num}: {e}")
        return False

def save_processing_stats(stats, output_dir):
    """Sauvegarde les statistiques de traitement"""
    try:
        stats_file = os.path.join(output_dir, "metadata", "processing_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.debug(f"Statistiques sauvegardées: {stats_file}")
        return True
    except Exception as e:
        logger.error(f"Erreur sauvegarde statistiques: {e}")
        return False

def wait_for_ollama_recovery(max_retries=3, wait_time=10):
    """Attend la récupération d'Ollama après un timeout"""
    for i in range(max_retries):
        logger.info(f"Tentative de récupération Ollama {i+1}/{max_retries}")
        time.sleep(wait_time)
        try:
            # Test simple de connexion
            embedding_model = get_embedding_model()
            return True
        except Exception as e:
            logger.warning(f"Ollama toujours indisponible: {e}")
    return False


def process_pdf_in_batches(pdf_input, output_dir="output", batch_size=20, overlap=5):
    """
    Traite un PDF par lots
    Args:
        pdf_input: Peut être un chemin (str) ou un UploadedFile de Streamlit
    """
    logger.info("=== DÉBUT DU TRAITEMENT PDF ===")
    
    # Créer la structure de dossiers
    try:
        ensure_directory_structure(output_dir)
    except Exception as e:
        logger.error(f"Impossible de créer la structure de dossiers: {e}")
        raise

    # Initialiser les statistiques
    doc_stats = {
        "filename": getattr(pdf_input, "name", os.path.basename(str(pdf_input))),
        "total_pages": 0,
        "process_start": datetime.now().isoformat(),
        "batch_size": batch_size,
        "pages_processed": 0,
        "images_extracted": 0,
        "chunks_created": 0,
        "text_files_saved": 0,
        "errors": []
    }

    # Ouvrir le PDF
    try:
        # Gestion différente selon le type d'entrée
        if hasattr(pdf_input, "read"):  # UploadedFile de Streamlit
            pdf_content = pdf_input.read()
            doc = fitz.open(stream=pdf_content, filetype="pdf")
        else:  # Chemin de fichier classique
            doc = fitz.open(pdf_input)
            
        total_pages = len(doc)
        doc_stats["total_pages"] = total_pages
        logger.info(f"PDF ouvert: {total_pages} pages")
    except Exception as e:
        logger.error(f"Erreur ouverture PDF: {e}")
        raise

    # Initialiser le modèle d'embedding avec retry
    embedding_model = None
    vector_db = None
    
    try:
        logger.info("Initialisation du modèle d'embedding...")
        embedding_model = get_embedding_model()
        
        logger.info("Initialisation de la base vectorielle...")
        vector_db = EnhancedVectorDB(
            embedding_model, 
            persist_directory=os.path.join(output_dir, "vector_db")
        )
        vector_db.initialize_db()
        logger.info("Base vectorielle initialisée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur initialisation embedding/vectordb: {e}")
        logger.info("Continuation sans vectorisation...")

    all_documents = []
    successful_batches = 0

    # Traitement par batch
    for batch_start in tqdm(range(0, total_pages, batch_size - overlap), desc="Traitement des batches"):
        batch_end = min(batch_start + batch_size, total_pages)
        batch_documents = []
        batch_images = 0
        batch_errors = []

        logger.info(f"=== BATCH {batch_start}-{batch_end} ===")

        # Traitement des pages du batch
        for page_num in range(batch_start, batch_end):
            try:
                page = doc[page_num]
                text = page.get_text()
                logger.info(f"Texte extrait: {text}")
                # Gestion du texte
                if not text or not text.strip():
                    logger.warning(f"[Page {page_num+1}] Aucun texte extrait")
                    text = f"[Page {page_num+1} - Contenu vide ou uniquement images]"
                else:
                    logger.info(f"[Page {page_num+1}] Texte extrait: {len(text)} caractères")

                # Sauvegarder le texte immédiatement
                if save_text_content(text, page_num + 1, output_dir):
                    doc_stats["text_files_saved"] += 1

                # Créer le document de base
                page_doc = Document(
                    page_content=text,
                    metadata={
                        "source": getattr(pdf_input, "name", os.path.basename(str(pdf_input))),
                        "page": page_num + 1,
                        "batch": f"{batch_start}-{batch_end}",
                        "has_content": bool(text.strip()),
                        "content_length": len(text)
                    }
                )

                # Extraction des images
                try:
                    images = extract_images_from_page(page)
                    batch_images += len(images)
                    
                    if images:
                        # Traitement des images et OCR
                        image_paths = []
                        image_descriptions = []
                        ocr_texts = []
                        
                        for img in images:
                            image_paths.append(img.get("path", ""))
                            image_descriptions.append(img.get("description", "N/A"))
                            ocr_text = img.get("ocr_text", "")
                            if ocr_text and ocr_text.strip():
                                ocr_texts.append(ocr_text)

                        # Ajouter les métadonnées d'images (en tant que strings)
                        page_doc.metadata["images_count"] = len(images)
                        page_doc.metadata["has_images"] = True
                        
                        # Combiner le texte OCR avec le contenu de la page
                        if ocr_texts:
                            ocr_combined = "\n".join(ocr_texts).strip()
                            page_doc.page_content += f"\n\n[CONTENU IMAGES - OCR]\n{ocr_combined}"
                            
                            logger.info(f"[Page {page_num+1}] OCR ajouté: {len(ocr_combined)} caractères")

                        logger.info(f"[Page {page_num+1}] {len(images)} image(s) traitée(s)")
                    else:
                        page_doc.metadata["has_images"] = False
                        page_doc.metadata["images_count"] = 0

                except Exception as e:
                    logger.warning(f"Erreur extraction images page {page_num+1}: {e}")
                    page_doc.metadata["image_extraction_error"] = str(e)
                    batch_errors.append(f"Page {page_num+1} images: {e}")

                batch_documents.append(page_doc)

            except Exception as e:
                logger.error(f"Erreur traitement page {page_num+1}: {e}")
                batch_errors.append(f"Page {page_num+1}: {e}")
                
                # Créer un document d'erreur pour maintenir la cohérence
                error_doc = Document(
                    page_content=f"[ERREUR PAGE {page_num+1}] {str(e)}",
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "batch": f"{batch_start}-{batch_end}",
                        "has_content": False,
                        "has_error": True,
                        "error": str(e)
                    }
                )
                batch_documents.append(error_doc)

        # Mise à jour des statistiques du batch
        doc_stats["pages_processed"] += len(batch_documents)
        doc_stats["images_extracted"] += batch_images
        doc_stats["errors"].extend(batch_errors)

        all_documents.extend(batch_documents)

        # Tentative de vectorisation avec gestion d'erreur robuste
        if vector_db and batch_documents:
            try:
                # Filtrer les documents avec contenu significatif pour la vectorisation
                documents_for_vectorization = [
                    doc for doc in batch_documents 
                    if doc.page_content.strip() and len(doc.page_content.strip()) > 10
                ]
                
                if documents_for_vectorization:
                    logger.info(f"chunks pour {documents_for_vectorization} documents...")
                    logger.info(f"Création des chunks pour {len(documents_for_vectorization)} documents...")
                    
                    # Créer les chunks avec gestion d'erreur
                    try:
                        hierarchical_chunks = create_hierarchical_chunks(documents_for_vectorization)
                        
                        if hierarchical_chunks:
                            logger.info(f"{len(hierarchical_chunks)} chunks créés")
                            
                            # Tentative d'ajout à la base vectorielle avec retry
                            max_vector_retries = 2
                            for retry in range(max_vector_retries):
                                try:
                                    logger.info(f"Ajout de: {hierarchical_chunks} chunks à la base vectorielle (tentative)")
                                    vector_db.add_documents(hierarchical_chunks)
                                    logger.info(f"ajout reussi: {vector_db}")
                                    
                                    doc_stats["chunks_created"] += len(hierarchical_chunks)
                                    logger.info(f"Chunks ajoutés à la base vectorielle (tentative {retry+1})")
                                    break
                                    
                                except Exception as vector_error:
                                    logger.warning(f"Erreur vectorisation (tentative {retry+1}): {vector_error}")
                                    if retry < max_vector_retries - 1:
                                        if "timeout" in str(vector_error).lower():
                                            logger.info("Timeout détecté, attente de récupération...")
                                            wait_for_ollama_recovery()
                                    else:
                                        logger.error(f"Échec définitif vectorisation batch {batch_start}-{batch_end}")
                                        batch_errors.append(f"Vectorisation: {vector_error}")
                        else:
                            logger.warning(f"Aucun chunk généré pour le batch {batch_start}-{batch_end}")
                            
                    except Exception as chunk_error:
                        logger.error(f"Erreur création chunks: {chunk_error}")
                        batch_errors.append(f"Chunking: {chunk_error}")
                else:
                    logger.warning(f"Aucun document vectorisable dans le batch {batch_start}-{batch_end}")
                    
            except Exception as e:
                logger.error(f"Erreur générale vectorisation batch {batch_start}-{batch_end}: {e}")
                batch_errors.append(f"Vectorisation générale: {e}")

        # Sauvegarder les statistiques intermédiaires
        doc_stats["last_processed_page"] = batch_end
        doc_stats["last_update"] = datetime.now().isoformat()
        doc_stats["successful_batches"] = successful_batches + (1 if not batch_errors else 0)
        
        save_processing_stats(doc_stats, output_dir)
        
        if not batch_errors:
            successful_batches += 1
            logger.info(f"✓ Batch {batch_start}-{batch_end} traité avec succès")
        else:
            logger.warning(f"⚠ Batch {batch_start}-{batch_end} traité avec {len(batch_errors)} erreur(s)")

    # Fermeture et finalisation
    doc.close()

    # Statistiques finales
    doc_stats["process_end"] = datetime.now().isoformat()
    doc_stats["status"] = "completed" if successful_batches > 0 else "completed_with_errors"
    doc_stats["total_documents"] = len(all_documents)
    doc_stats["successful_batches"] = successful_batches
    doc_stats["total_batches"] = len(range(0, total_pages, batch_size - overlap))
    
    # Sauvegarde finale
    save_processing_stats(doc_stats, output_dir)

    # Résumé final
    logger.info("=== RÉSUMÉ DU TRAITEMENT ===")
    logger.info(f"Pages traitées: {doc_stats['pages_processed']}/{doc_stats['total_pages']}")
    logger.info(f"Fichiers texte sauvegardés: {doc_stats['text_files_saved']}")
    logger.info(f"Images extraites: {doc_stats['images_extracted']}")
    logger.info(f"Chunks créés: {doc_stats['chunks_created']}")
    logger.info(f"Batches réussis: {successful_batches}/{doc_stats['total_batches']}")
    logger.info(f"Erreurs: {len(doc_stats['errors'])}")
    
    if doc_stats['errors']:
        logger.warning("Erreurs rencontrées:")
        for error in doc_stats['errors'][:5]:  # Afficher les 5 premières erreurs
            logger.warning(f"  - {error}")
        if len(doc_stats['errors']) > 5:
            logger.warning(f"  ... et {len(doc_stats['errors']) - 5} autres erreurs")

    return vector_db, doc_stats

def get_document_structure(all_documents):
    """Extrait la structure du document à partir des documents traités"""
    try:
        logger.info("dans la fonction get_document_structure")
        
        structure = {
            "chapters": [],
            "sections": [],
            "total_pages": len(all_documents),
            "extraction_date": datetime.now().isoformat()
        }
        
        # Recherche de patterns de structure dans le contenu
        chapter_patterns = [
            r'CHAPITRE\s+\d+',
            r'PARTIE\s+\d+', 
            r'SECTION\s+\d+',
            r'\d+\.\s+[A-Z][^.]*$'
        ]
        
        for doc in all_documents:
            content = doc.page_content
            page_num = doc.metadata.get('page', 0)
            
            # Recherche de chapitres/sections
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 5 and len(line) < 100:  # Titres potentiels
                    for pattern in chapter_patterns:
                        import re
                        if re.search(pattern, line, re.IGNORECASE):
                            structure["chapters"].append({
                                "title": line,
                                "page": page_num,
                                "pattern": pattern
                            })
                            break
        
        logger.info(f"Structure extraite: {len(structure['chapters'])} éléments trouvés")
        return structure
        
    except Exception as e:
        logger.error(f"Erreur extraction structure: {e}")
        return {"chapters": [], "sections": [], "error": str(e)}

def generate_report_from_pdf(pdf_path, output_dir="output", batch_size=20, report_file="rapport_final.md"):
    """Génère un rapport complet à partir d'un PDF avec gestion d'erreur robuste"""
    try:
        logger.info("=== GÉNÉRATION DU RAPPORT ===")
        
        # Traitement du PDF
        vector_db, stats = process_pdf_in_batches(pdf_path, output_dir, batch_size)
        
        # Tentative de génération du rapport
        if vector_db:
            try:
                logger.info("Initialisation du générateur de rapport...")
                generator = ReportGenerator(vector_db)
                
                # Extraction de structure
                try:
                    generator.extract_document_structure()
                    document_structure = generator.context_memory.get("document_structure", {})
                    chapters = document_structure.get("chapters", [])
                    
                    if chapters:
                        logger.info(f"Structure détectée: {len(chapters)} chapitres")
                        for chapter in tqdm(chapters[:10], desc="Analyse des chapitres"):  # Limiter à 10 pour éviter les timeouts
                            try:
                                if isinstance(chapter, dict):
                                    chapter_title = chapter.get("title", "Chapitre sans titre")
                                    generator.analyze_section(chapter_title)
                                else:
                                    generator.analyze_section(str(chapter))
                            except Exception as e:
                                logger.warning(f"Erreur analyse chapitre {chapter}: {e}")
                    else:
                        logger.info("Aucune structure spécifique détectée")
                        
                except Exception as e:
                    logger.warning(f"Erreur extraction structure: {e}")

                # Génération du rapport final
                logger.info("Génération du rapport final...")
                final_report = generator.generate_full_report()
                
            except Exception as e:
                logger.error(f"Erreur génération rapport avancé: {e}")
                # Génération d'un rapport basique
                final_report = generate_basic_report(stats)
        else:
            logger.warning("Pas de base vectorielle disponible, génération d'un rapport basique")
            final_report = generate_basic_report(stats)

        # Sauvegarder le rapport
        report_path = os.path.join(output_dir, report_file)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        logger.info(f"✓ Rapport généré: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Erreur génération rapport: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def generate_basic_report(stats):
    """Génère un rapport basique à partir des statistiques"""
    report = f"""# Rapport de traitement PDF

## Informations générales
- **Fichier**: {stats.get('filename', 'N/A')}
- **Pages totales**: {stats.get('total_pages', 0)}
- **Date de traitement**: {stats.get('process_start', 'N/A')}
- **Statut**: {stats.get('status', 'Unknown')}

## Résultats du traitement
- **Pages traitées**: {stats.get('pages_processed', 0)}
- **Fichiers texte créés**: {stats.get('text_files_saved', 0)}
- **Images extraites**: {stats.get('images_extracted', 0)}
- **Chunks vectorisés**: {stats.get('chunks_created', 0)}

## Performances
- **Batches réussis**: {stats.get('successful_batches', 0)}/{stats.get('total_batches', 0)}
- **Taille des batches**: {stats.get('batch_size', 0)}

## Erreurs
"""
    
    errors = stats.get('errors', [])
    if errors:
        report += f"- **Nombre d'erreurs**: {len(errors)}\n"
        report += "- **Principales erreurs**:\n"
        for error in errors[:10]:
            report += f"  - {error}\n"
    else:
        report += "- Aucune erreur détectée\n"
    
    report += f"""
## Fichiers générés
- Textes extraits: `extracted_text/`
- Images: `extracted_images/`  
- Métadonnées: `metadata/`
- Base vectorielle: `vector_db/`

---
*Rapport généré automatiquement le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Traiter un PDF volumineux et générer un rapport")
    parser.add_argument("pdf_path", help="Chemin vers le fichier PDF à traiter")
    parser.add_argument("--output", "-o", default="output", help="Dossier de sortie")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Taille des lots de traitement")
    parser.add_argument("--report", "-r", default="rapport_final.md", help="Nom du rapport final")
    parser.add_argument("--verbose", "-v", action="store_true", help="Activer les logs détaillés")

    args = parser.parse_args()

    # Configuration du logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.pdf_path):
        logger.error(f"Fichier introuvable: {args.pdf_path}")
        return 1

    try:
        generate_report_from_pdf(args.pdf_path, args.output, args.batch_size, args.report)
        logger.info("✓ Traitement terminé avec succès!")
        return 0
    except Exception as e:
        logger.error(f"✗ Erreur fatale: {e}")
        logger.error(f"Traceback complet: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())