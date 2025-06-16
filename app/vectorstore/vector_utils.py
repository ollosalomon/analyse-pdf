# Recherche vectorielle avancée
# Tu peux ajouter ici des fonctions de recherche plus spécialisées

def format_search_results(results):
    """
    Formatage simple des résultats vectoriels.
    """
    formatted = []
    for doc in results:
        metadata = doc.metadata
        text = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        formatted.append(f"[Page {metadata.get('page', '?')}] {text}")
    return "\n".join(formatted)
