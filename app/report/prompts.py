# Prompts LLM pour analyse et synthèse
from langchain.prompts import ChatPromptTemplate

STRUCTURE_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un analyste de document expert. Voici des extraits d'un document technique.
    Extrais la structure de ce document (chapitres, sections principales).

    EXTRAITS:
    {document_samples}

    Réponds uniquement avec une liste hiérarchique au format JSON des chapitres et sections.
    """
)

SECTION_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un expert en analyse documentaire. Analyse la section "{section_title}" 
    basée sur les extraits suivants du document. 

    CONTEXTE PRÉCÉDENT:
    {previous_context}

    EXTRAITS DE LA SECTION:
    {section_content}

    Rédige un résumé analytique de cette section qui:
    1. Synthétise les informations clés
    2. Identifie les points importants
    3. Lie cette section au contexte global du document

    Ton résumé doit être concis, informatif et bien structuré.
    """
)

KEY_POINTS_PROMPT = ChatPromptTemplate.from_template(
    """Basé sur cette analyse de section:
    {section_analysis}

    Extrais exactement 3 points clés qui devraient être inclus dans un rapport global.
    Réponds uniquement avec les points, un par ligne, sans numérotation ni puces.
    """
)

FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un expert en rédaction de rapports exécutifs. Crée un rapport complet et professionnel
    basé sur l'analyse d'un document technique. Utilise les résumés de sections et les points clés fournis.

    TITRE DU DOCUMENT: {document_title}

    STRUCTURE DU DOCUMENT:
    {document_structure}

    POINTS CLÉS:
    {key_findings}

    RÉSUMÉS DES SECTIONS:
    {section_summaries}

    Crée un rapport professionnel qui:
    1. Commence par un résumé exécutif concis
    2. Présente les points clés et les conclusions importantes
    3. Fournit une analyse structurée suivant l'organisation du document original
    4. Se termine par des recommandations ou conclusions générales

    Le rapport doit être bien formaté avec des titres, sous-titres et paragraphes appropriés.
    """
)
