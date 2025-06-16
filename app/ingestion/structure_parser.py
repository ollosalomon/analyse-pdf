# Parsing générique des chapitres/sections
import re
import logging
logger = logging.getLogger("pdf_cli")
logging.basicConfig(level=logging.INFO)

def get_document_structure(documents):
    """
    Extrait la structure hiérarchique du document de manière générique.
    Détecte automatiquement différents formats de titres.
    """
    logger.info("dans la fonction get_document_structure")
    structure = []
    
    # Patterns génériques pour différents formats de titres
    patterns = [
        # Chapitres traditionnels
        {
            'name': 'chapter',
            'pattern': re.compile(r'^(?:Chapitre|CHAPITRE|Chapter|CHAPTER)\s+([IVXLCDM]+|\d+)[.:]\s*(.+?)$', re.MULTILINE | re.IGNORECASE),
            'level': 1,
            'title_group': 2
        },
        # Sections numérotées principales (1. TITRE)
        {
            'name': 'main_section',
            'pattern': re.compile(r'^(\d+)\.\s+([A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝ][A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝ\s\-\']+)(?:\s*$|\s*:)', re.MULTILINE),
            'level': 1,
            'title_group': 2
        },
        # Sous-sections numérotées (1.1. TITRE, 2.3. TITRE)
        {
            'name': 'subsection',
            'pattern': re.compile(r'^(\d+\.\d+)\.?\s+(.+?)$', re.MULTILINE),
            'level': 2,
            'title_group': 2
        },
        # Sous-sections avec lettres (a) TITRE, A) TITRE)
        {
            'name': 'letter_section',
            'pattern': re.compile(r'^([a-zA-Z])\)\s+(.+?)$', re.MULTILINE),
            'level': 2,
            'title_group': 2
        },
        # Sous-sous-sections (1.1.1. TITRE)
        {
            'name': 'subsubsection',
            'pattern': re.compile(r'^(\d+\.\d+\.\d+)\.?\s+(.+?)$', re.MULTILINE),
            'level': 3,
            'title_group': 2
        },
        # Titres en majuscules sur une ligne (heuristique)
        {
            'name': 'uppercase_title',
            'pattern': re.compile(r'^([A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝ][A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝ\s\-\']{8,})$', re.MULTILINE),
            'level': 1,
            'title_group': 1
        }
    ]

    all_matches = []
    
    # Extraction de tous les titres potentiels
    for doc in documents:
        text = doc.page_content
        page_num = doc.metadata.get('page', 0)
        
        for pattern_info in patterns:
            matches = pattern_info['pattern'].finditer(text)
            for match in matches:
                title = match.group(pattern_info['title_group']).strip()
                
                # Filtrage des faux positifs
                if is_valid_title(title, pattern_info['name']):
                    all_matches.append({
                        'type': pattern_info['name'],
                        'title': title,
                        'page': page_num,
                        'level': pattern_info['level'],
                        'position': match.start(),
                        'match': match.group(0)
                    })
    
    # Tri par page puis par position dans le texte
    all_matches.sort(key=lambda x: (x['page'], x['position']))
    
    # Construction de la structure hiérarchique
    structure = build_hierarchical_structure(all_matches)
    
    logger.info(f"Structure extraite: {structure} éléments principaux")
    return structure

def is_valid_title(title, pattern_type):
    """
    Filtre les faux positifs selon le type de pattern.
    """
    # Filtres génériques
    if len(title.strip()) < 3:
        return False
    
    # Éviter les lignes avec trop de chiffres ou caractères spéciaux
    if len(re.findall(r'[\d\.\-_/\\]', title)) > len(title) * 0.6:
        return False
    
    # Filtres spécifiques par type
    if pattern_type == 'uppercase_title':
        # Éviter les URLs, emails, codes
        if any(x in title.lower() for x in ['http', 'www', '@', '.com', '.fr', '.pdf']):
            return False
        # Éviter les lignes avec beaucoup de chiffres
        if len(re.findall(r'\d', title)) > 5:
            return False
    
    if pattern_type == 'main_section':
        # Les sections principales doivent avoir une certaine longueur
        if len(title) < 8:
            return False
    
    return True

def build_hierarchical_structure(matches):
    """
    Construit une structure hiérarchique à partir des matches triés.
    """
    structure = []
    current_level1 = None
    current_level2 = None
    
    for match in matches:
        if match['level'] == 1:
            # Nouveau chapitre/section principale
            current_level1 = {
                'type': match['type'],
                'title': match['title'],
                'start_page': match['page'],
                'sections': []
            }
            structure.append(current_level1)
            current_level2 = None
            
        elif match['level'] == 2 and current_level1:
            # Sous-section
            current_level2 = {
                'type': match['type'],
                'title': match['title'],
                'page': match['page'],
                'subsections': []
            }
            current_level1['sections'].append(current_level2)
            
        elif match['level'] == 3 and current_level2:
            # Sous-sous-section
            current_level2['subsections'].append({
                'type': match['type'],
                'title': match['title'],
                'page': match['page']
            })
    logger.info(f"Structure hiérarchique construite avec {structure} sections principales.")
    return structure

def print_structure(structure, indent=0):
    """
    Affiche la structure du document de manière lisible avec indentation.
    """
    for item in structure:
        prefix = "  " * indent
        print(f"{prefix}{item['type'].upper()}: {item['title']} (page {item['start_page']})")
        
        for section in item.get('sections', []):
            section_prefix = "  " * (indent + 1)
            print(f"{section_prefix}- {section['type']}: {section['title']} (page {section['page']})")
            
            for subsection in section.get('subsections', []):
                subsection_prefix = "  " * (indent + 2)
                print(f"{subsection_prefix}• {subsection['type']}: {subsection['title']} (page {subsection['page']})")

def get_structure_summary(structure):
    """
    Retourne un résumé de la structure détectée.
    """
    total_sections = len(structure)
    total_subsections = sum(len(item.get('sections', [])) for item in structure)
    total_subsubsections = sum(
        len(section.get('subsections', []))
        for item in structure
        for section in item.get('sections', [])
    )
    
    return {
        'total_main_sections': total_sections,
        'total_subsections': total_subsections,
        'total_subsubsections': total_subsubsections,
        'detected_types': list(set(item['type'] for item in structure))
    }




# # Parsing des chapitres/sections
# import re
# import logging
# logger = logging.getLogger("pdf_cli")
# logging.basicConfig(level=logging.INFO)
# def get_document_structure(documents):
#     """
#     Extrait la structure hiérarchique du document (chapitres, sections).
#     """
#     logger.info("dans la fonction get_document_structure")
#     structure = []
#     current_chapter = None

#     chapter_pattern = re.compile(r'^(?:Chapitre|CHAPITRE)\s+\d+[.:]\s*(.*?)$', re.MULTILINE)
#     section_pattern = re.compile(r'^\d+\.\d+\.\s+(.*?)$', re.MULTILINE)

#     for doc in documents:
#         text = doc.page_content
#         page_num = doc.metadata.get('page')

#         chapter_matches = list(chapter_pattern.finditer(text))
        
#         if chapter_matches:
#             logger.info(f"Chapitres trouvés en page {page_num}: {[m.group(1) for m in chapter_matches]}")

#         for match in chapter_matches:
#             chapter_title = match.group(1).strip()
#             current_chapter = {
#                 "type": "chapter",
#                 "title": chapter_title,
#                 "start_page": page_num,
#                 "sections": []
#             }
#             structure.append(current_chapter)

#         if current_chapter:
#             section_matches = section_pattern.finditer(text)
#             for match in section_matches:
#                 section_title = match.group(1).strip()
#                 current_chapter["sections"].append({
#                     "type": "section",
#                     "title": section_title,
#                     "page": page_num
#                 })

#     return structure

