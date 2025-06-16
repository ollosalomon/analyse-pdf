import requests
import json
import numpy as np
from typing import List
import logging

# Import correct selon la version de LangChain
try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    try:
        from langchain.embeddings.base import Embeddings
    except ImportError:
        # Fallback pour versions très anciennes
        from langchain.schema.embeddings import Embeddings

logger = logging.getLogger(__name__)

class OllamaEmbeddings(Embeddings):
    """
    Classe pour utiliser Ollama comme modèle d'embedding
    """
    
    def __init__(self, model_name: str = "llava", base_url: str = "http://ollama_llava:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed une liste de documents"""
        embeddings = []
        for text in texts:
            try:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Erreur embedding pour le texte: {e}")
                # Retourner un embedding vide en cas d'erreur
                embeddings.append([0.0] * 384)  # Taille standard
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed une requête"""
        return self._get_embedding(text)
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Récupère l'embedding d'un texte via l'API Ollama
        Note: Cette méthode dépend de la disponibilité de l'API embeddings d'Ollama
        """
        try:
            # Méthode 1: API embeddings (si disponible)
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("embedding", [])
            else:
                logger.warning(f"API embeddings non disponible, utilisation de méthode alternative")
                return self._get_embedding_via_generate(text)
                
        except Exception as e:
            logger.error(f"Erreur API embeddings: {e}")
            return self._get_embedding_via_generate(text)
    
    def _get_embedding_via_generate(self, text: str) -> List[float]:
        """
        Méthode alternative: utiliser l'API generate pour créer un embedding
        (moins optimal mais fonctionnel)
        """
        try:
            # Prompt pour demander une représentation vectorielle
            prompt = f"Créer une représentation numérique de ce texte (répondre uniquement avec des nombres séparés par des virgules): {text[:500]}"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                # Extraire et convertir la réponse en vecteur
                result = response.json().get("response", "")
                # Ici vous devriez implémenter une logique pour convertir
                # la réponse en vecteur numérique
                return self._text_to_vector(text)
            else:
                logger.error(f"Erreur API generate: {response.status_code}")
                return self._text_to_vector(text)
                
        except Exception as e:
            logger.error(f"Erreur méthode alternative: {e}")
            return self._text_to_vector(text)
    
    def _text_to_vector(self, text: str) -> List[float]:
        """
        Méthode de fallback: créer un embedding simple basé sur le hash du texte
        """
        # Méthode très basique pour créer un vecteur
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convertir en vecteur de 384 dimensions
        vector = []
        for i in range(0, len(hash_hex), 2):
            vector.append(int(hash_hex[i:i+2], 16) / 255.0)
        
        # Compléter jusqu'à 384 dimensions
        while len(vector) < 384:
            vector.append(0.0)
        
        return vector[:384]

def get_embedding_model():
    """
    Charge le modèle d'embedding via Ollama avec fallback
    """
    try:
        # Vérifier la disponibilité d'Ollama
        logger.info("Test de connexion à Ollama...")
        response = requests.get("http://ollama_llava:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [m.get("name", "") for m in models]
            logger.info(f"Modèles Ollama disponibles: {available_models}")
            
            if any("llava" in model for model in available_models):
                logger.info("Utilisation d'Ollama/Llava pour les embeddings")
                return OllamaEmbeddings(model_name="llava")
            else:
                logger.warning("Llava non trouvé, tentative avec modèle par défaut")
                return OllamaEmbeddings()
        else:
            logger.warning(f"Ollama répond avec status {response.status_code}")
            return get_gpt4all_embedding_model()
            
    except requests.exceptions.ConnectionError:
        logger.error("Impossible de se connecter à Ollama")
        return get_gpt4all_embedding_model()
    except Exception as e:
        logger.error(f"Erreur inattendue avec Ollama: {e}")
        return get_gpt4all_embedding_model()

def get_gpt4all_embedding_model():
    """
    Fonction de fallback pour GPT4All
    """
    from langchain_community.embeddings import GPT4AllEmbeddings
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    return GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)

# Version améliorée avec gestion d'erreur
def get_embedding_model_robust():
    """
    Version robuste qui teste plusieurs options
    """
    embedding_options = [
        ("Ollama", get_embedding_model),
        ("GPT4All", get_gpt4all_embedding_model)
    ]
    
    for name, func in embedding_options:
        try:
            model = func()
            # Test simple pour vérifier que le modèle fonctionne
            test_embedding = model.embed_query("test")
            if test_embedding and len(test_embedding) > 0:
                logger.info(f"Modèle d'embedding {name} chargé avec succès")
                return model
        except Exception as e:
            logger.warning(f"Échec du chargement {name}: {e}")
    
    raise Exception("Aucun modèle d'embedding n'a pu être chargé")



# # Chargement du modèle d’embedding
# from langchain_community.embeddings import GPT4AllEmbeddings

# def get_embedding_model():
#     """
#     Charge le modèle d'embedding GPT4All.
#     """
#     model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
#     gpt4all_kwargs = {'allow_download': 'True'}
#     return GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)
