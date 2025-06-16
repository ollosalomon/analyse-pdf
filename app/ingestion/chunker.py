from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.ingestion.structure_parser import get_document_structure
import logging

logger = logging.getLogger("chunker")

def create_hierarchical_chunks(documents):
    """
    Crée des chunks hiérarchiques avec fallback si aucune structure n'est détectée.
    """
    logger.info("dans la fonction create_hierarchical_chunks")
    structure = get_document_structure(documents)
    logger.info(f"structure détectée : {structure}")
    
    all_chunks = []
    
    if not structure:
        logger.warning("Aucune structure détectée (pas de chapitres trouvés). Utilisation du chunking brut.")
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        # Protection contre les documents sans texte
        docs = [doc for doc in documents if doc.page_content.strip()]
        logger.info(f"documents valides pour chunking: {docs}")
        if not docs:
            logger.error("Aucun contenu textuel à chunker.")
            return []
        return fallback_splitter.split_documents(docs)

    # Définition des splitters avec des paramètres adaptés à la hiérarchie
    chapter_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=200,
        length_function=len
    )
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=150,
        length_function=len
    )
    detail_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    for chapter_idx, chapter in enumerate(structure):
        chapter_start = chapter["start_page"]
        chapter_end = structure[chapter_idx + 1]["start_page"] if chapter_idx + 1 < len(structure) else None

        # Récupération des documents du chapitre
        chapter_docs = [doc for doc in documents
                        if doc.metadata.get('page') >= chapter_start and
                        (chapter_end is None or doc.metadata.get('page') < chapter_end)
                        and doc.metadata.get("type", "") != "ocr_image"]

        if not chapter_docs:
            logger.warning(f"Aucun document trouvé pour le chapitre: {chapter['title']}")
            continue

        chapter_text = "\n".join([doc.page_content for doc in chapter_docs])
        
        if not chapter_text.strip():
            logger.warning(f"Chapitre vide: {chapter['title']}")
            continue

        # 1. CHUNK GLOBAL DU CHAPITRE (résumé/vue d'ensemble)
        if len(chapter_text) > 8000:
            # Si le chapitre est très long, on le divise avec chapter_splitter
            logger.info(f"Chapitre long détecté ({len(chapter_text)} chars): {chapter['title']}")
            chapter_chunks = chapter_splitter.create_documents(
                [chapter_text],
                metadatas=[{
                    "type": "chapter",
                    "chapter_type": chapter["type"],
                    "title": chapter["title"],
                    "start_page": chapter_start,
                    "level": "summary",
                    "chunk_index": i
                } for i in range(len(chapter_text) // 8000 + 1)]
            )
            all_chunks.extend(chapter_chunks)
        else:
            # Chapitre court, un seul chunk
            chapter_chunk = Document(
                page_content=chapter_text,
                metadata={
                    "type": "chapter",
                    "chapter_type": chapter["type"],
                    "title": chapter["title"],
                    "start_page": chapter_start,
                    "level": "summary"
                }
            )
            all_chunks.append(chapter_chunk)

        # 2. TRAITEMENT DES SECTIONS DU CHAPITRE
        if chapter.get("sections"):
            logger.info(f"Traitement de {len(chapter['sections'])} sections pour: {chapter['title']}")
            
            for section in chapter["sections"]:
                # Essayer de récupérer le texte spécifique à la section
                section_text = extract_section_text(section, chapter_docs, chapter_text)
                
                if section_text and len(section_text.strip()) > 50:  # Minimum de contenu
                    # Chunking des sections
                    if len(section_text) > 2000:
                        # Section longue, diviser avec section_splitter
                        section_chunks = section_splitter.create_documents(
                            [section_text],
                            metadatas=[{
                                "type": "section",
                                "section_type": section["type"],
                                "chapter": chapter["title"],
                                "section_title": section["title"],
                                "start_page": chapter_start,
                                "section_page": section.get("page", chapter_start),
                                "level": "section",
                                "chunk_index": i
                            } for i in range(len(section_text) // 2000 + 1)]
                        )
                        all_chunks.extend(section_chunks)
                    else:
                        # Section courte, un seul chunk
                        section_chunk = Document(
                            page_content=section_text,
                            metadata={
                                "type": "section",
                                "section_type": section["type"],
                                "chapter": chapter["title"],
                                "section_title": section["title"],
                                "start_page": chapter_start,
                                "section_page": section.get("page", chapter_start),
                                "level": "section"
                            }
                        )
                        all_chunks.append(section_chunk)

                    # 3. TRAITEMENT DES SOUS-SECTIONS
                    if section.get("subsections"):
                        for subsection in section["subsections"]:
                            subsection_text = extract_subsection_text(subsection, chapter_docs, section_text)
                            
                            if subsection_text and len(subsection_text.strip()) > 30:
                                detail_chunks = detail_splitter.create_documents(
                                    [subsection_text],
                                    metadatas=[{
                                        "type": "subsection",
                                        "subsection_type": subsection["type"],
                                        "chapter": chapter["title"],
                                        "section_title": section["title"],
                                        "subsection_title": subsection["title"],
                                        "page": subsection.get("page", chapter_start),
                                        "level": "detail"
                                    }]
                                )
                                all_chunks.extend(detail_chunks)
        else:
            # Pas de sections détectées, chunking détaillé du chapitre entier
            logger.info(f"Pas de sections détectées pour {chapter['title']}, chunking détaillé")
            detail_chunks = detail_splitter.create_documents(
                [chapter_text],
                metadatas=[{
                    "type": "chapter_detail",
                    "chapter": chapter["title"],
                    "start_page": chapter_start,
                    "level": "detail"
                }]
            )
            all_chunks.extend(detail_chunks)

    # 4. TRAITEMENT DES DOCUMENTS OCR HORS STRUCTURE
    ocr_docs = [doc for doc in documents if doc.metadata.get("type") == "ocr_image"]
    if ocr_docs:
        logger.info(f"{len(ocr_docs)} documents OCR ajoutés au chunking.")
        ocr_chunks = detail_splitter.split_documents(ocr_docs)
        # Ajout de métadonnées spécifiques pour les chunks OCR
        for chunk in ocr_chunks:
            chunk.metadata.update({
                "level": "ocr",
                "type": "ocr_content"
            })
        all_chunks.extend(ocr_chunks)

    logger.info(f"Chunking terminé: {len(all_chunks)} chunks créés")
    return all_chunks

def extract_section_text(section, chapter_docs, chapter_text):
    """
    Extrait le texte spécifique à une section.
    Stratégie: chercher le titre de la section dans le texte et extraire le contenu qui suit.
    """
    section_title = section["title"]
    
    # Essayer de trouver la section dans le texte
    # Chercher différentes variations du titre
    possible_patterns = [
        rf"^[a-z]\)\s+{re.escape(section_title)}",  # a) TITRE
        rf"^\d+\.\d+\.?\s+{re.escape(section_title)}",  # 1.1. TITRE
        rf"^{re.escape(section_title)}"  # TITRE direct
    ]
    
    import re
    for pattern in possible_patterns:
        match = re.search(pattern, chapter_text, re.MULTILINE | re.IGNORECASE)
        if match:
            start_pos = match.start()
            # Trouver la fin de la section (début de la section suivante ou fin du chapitre)
            next_section_match = re.search(r'\n[a-z]\)\s+[A-Z]|\n\d+\.\d+', chapter_text[start_pos + len(match.group(0)):])
            if next_section_match:
                end_pos = start_pos + len(match.group(0)) + next_section_match.start()
                return chapter_text[start_pos:end_pos].strip()
            else:
                # Dernière section, prendre jusqu'à la fin
                return chapter_text[start_pos:].strip()
    
    # Si pas trouvé, retourner une partie du texte du chapitre
    return chapter_text[:1000] if len(chapter_text) > 1000 else chapter_text

def extract_subsection_text(subsection, chapter_docs, section_text):
    """
    Extrait le texte spécifique à une sous-section.
    """
    # Logique similaire à extract_section_text mais pour les sous-sections
    subsection_title = subsection["title"]
    
    import re
    # Chercher la sous-section dans le texte de la section
    pattern = rf"^{re.escape(subsection_title)}"
    match = re.search(pattern, section_text, re.MULTILINE | re.IGNORECASE)
    
    if match:
        start_pos = match.start()
        # Prendre les 500 premiers caractères après le titre
        return section_text[start_pos:start_pos + 500].strip()
    
    # Fallback: retourner une partie du texte de la section
    return section_text[:300] if len(section_text) > 300 else section_text



# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from app.ingestion.structure_parser import get_document_structure
# import logging

# logger = logging.getLogger("chunker")

# def create_hierarchical_chunks(documents):
#     """
#     Crée des chunks hiérarchiques avec fallback si aucune structure n’est détectée.
#     """
#     logger.info("dans la fonction create_hierarchical_chunks")
#     structure = get_document_structure(documents)
#     logger.info(f"structure détectée : {structure}")
    
#     all_chunks = []
    
#     if not structure:
#         logger.warning("Aucune structure détectée (pas de chapitres trouvés). Utilisation du chunking brut.")
#         fallback_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=100,
#             length_function=len
#         )
#         # Protection contre les documents sans texte
#         docs = [doc for doc in documents if doc.page_content.strip()]
#         if not docs:
#             logger.error("Aucun contenu textuel à chunker.")
#             return []
#         return fallback_splitter.split_documents(docs)

#     chapter_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=8000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     section_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100,
#         length_function=len
#     )

#     for chapter_idx, chapter in enumerate(structure):
#         chapter_start = chapter["start_page"]
#         chapter_end = structure[chapter_idx + 1]["start_page"] if chapter_idx + 1 < len(structure) else None

#         chapter_docs = [doc for doc in documents
#                         if doc.metadata.get('page') >= chapter_start and
#                         (chapter_end is None or doc.metadata.get('page') < chapter_end)
#                         and doc.metadata.get("type", "") != "ocr_image"]

#         chapter_text = "\n".join([doc.page_content for doc in chapter_docs])

#         # Chunk global du chapitre
#         chapter_chunk = Document(
#             page_content=chapter_text,
#             metadata={
#                 "type": "chapter",
#                 "title": chapter["title"],
#                 "start_page": chapter_start,
#                 "level": "summary"
#             }
#         )
#         all_chunks.append(chapter_chunk)

#         # Sub-chunking des sections
#         section_chunks = section_splitter.create_documents(
#             [chapter_text],
#             metadatas=[{
#                 "type": "section",
#                 "chapter": chapter["title"],
#                 "start_page": chapter_start,
#                 "level": "detail"
#             }]
#         )
#         all_chunks.extend(section_chunks)

#     # 🔽 Traitement des documents OCR hors structure
#     ocr_docs = [doc for doc in documents if doc.metadata.get("type") == "ocr_image"]
#     if ocr_docs:
#         logger.info(f"{len(ocr_docs)} documents OCR ajoutés au chunking.")
#         ocr_chunks = section_splitter.split_documents(ocr_docs)
#         all_chunks.extend(ocr_chunks)

#     return all_chunks
