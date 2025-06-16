import requests
import base64
import logging
import time
import json
import os
import pytesseract
from PIL import Image
logger = logging.getLogger("llava_image")

def check_ollama_connection():
    """
    Vérifie si Ollama est accessible et si le modèle LLaVA est disponible.
    """
    try:
        # Test de connexion basique
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        llava_available = any("llava" in model.get("name", "").lower() for model in models)
        
        if not llava_available:
            logger.error("[LLaVA] Le modèle LLaVA n'est pas installé. Installez-le avec : ollama pull llava")
            return False
            
        logger.info("[LLaVA] Connexion Ollama OK et modèle LLaVA disponible")
        return True
        
    except requests.exceptions.ConnectionError:
        logger.error("[LLaVA] Impossible de se connecter à Ollama sur localhost:11434")
        logger.error("Vérifiez que Ollama est lancé avec : ollama serve")
        return False
    except Exception as e:
        logger.error(f"[LLaVA] Erreur de vérification : {e}")
        return False

def analyze_image(image_path, max_retries=3):
    """
    Analyse une image avec le modèle LLaVA via Ollama.
    Avec gestion d'erreurs améliorée et retry automatique.
    """
    # Vérification préliminaire
    if not check_ollama_connection():
        return "Erreur : Ollama ou LLaVA non disponible."
    
    try:
        # Lire et encoder l'image en base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Préparer la requête à Ollama
        payload = {
            "model": "llava",
            "prompt": "Décris cette image en détail en te concentrant sur les éléments importants pour un rapport technique.",
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Pour des réponses plus consistantes
                "top_p": 0.9
            }
        }
        
        # Tentatives avec retry
        for attempt in range(max_retries):
            try:
                logger.info(f"[LLaVA] Analyse de l'image (tentative {attempt + 1}/{max_retries})")
                
                response = requests.post(
                    "http://localhost:11434/api/generate", 
                    json=payload,
                    timeout=60  # Timeout plus long pour l'analyse d'image
                )
                
                response.raise_for_status()
                data = response.json()
                
                # Vérifier si la réponse contient une erreur
                if "error" in data:
                    logger.error(f"[LLaVA] Erreur du modèle : {data['error']}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return f"Erreur du modèle : {data['error']}"
                
                result = data.get("response", "").strip()
                if result:
                    logger.info(f"[LLaVA] Analyse réussie pour {image_path}")
                    return result
                else:
                    logger.warning("[LLaVA] Réponse vide du modèle")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"[LLaVA] Timeout lors de l'analyse (tentative {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
            except requests.exceptions.ConnectionError:
                logger.error("[LLaVA] Perte de connexion à Ollama")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                    
            except Exception as e:
                logger.error(f"[LLaVA] Erreur lors de l'analyse (tentative {attempt + 1}) : {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        return "Erreur : Impossible d'analyser l'image après plusieurs tentatives."
        
    except FileNotFoundError:
        logger.error(f"[LLaVA] Fichier image introuvable : {image_path}")
        return "Erreur : Fichier image introuvable."
    except Exception as e:
        logger.error(f"[LLaVA] Erreur critique : {e}")
        return f"Erreur critique lors de l'analyse : {str(e)}"



def ocr_extract_text(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='eng+fra')  # Ajoute 'fra' si tu veux aussi le français
        logger.info(f"texte OCR extrait: {text.strip()}")
        return text.strip()
    except Exception as e:
        logger.error(f"[OCR] Erreur OCR pour {image_path} : {e}")
        return ""


def test_llava_connection():
    """
    Fonction de test pour vérifier la connexion LLaVA avec une image de test.
    """
    test_payload = {
        "model": "llava",
        "prompt": "Hello, can you see this?",
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=test_payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        logger.info(f"[LLaVA] Test de connexion réussi : {data.get('response', 'Pas de réponse')}")
        return True
    except Exception as e:
        logger.error(f"[LLaVA] Test de connexion échoué : {e}")
        return False

# Fonction d'extraction d'images améliorée
def extract_images_from_page(page):
    """
    Extrait les images d'une page PyMuPDF, les sauvegarde,
    les analyse avec LLaVA et OCR, puis retourne les métadonnées.
    """
    image_list = []

    # Vérification préliminaire de LLaVA
    llava_available = check_ollama_connection()

    images = page.get_images(full=True)

    for img_index, img in enumerate(images):
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)

            if not base_image:
                continue

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{page.number + 1}_img_{img_index}.{image_ext}"
            image_path = os.path.join("output/extracted_images", image_filename)

            # Créer le dossier si nécessaire
            os.makedirs("output/extracted_images", exist_ok=True)

            # Sauvegarder l'image
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            # Analyse LLaVA (si disponible)
            description = analyze_image(image_path) if llava_available else "N/A"

            # Extraction OCR
            ocr_text = ocr_extract_text(image_path)

            image_list.append({
                "path": image_path,
                "description": description,
                "ocr_text": ocr_text,
                "page": page.number + 1,
                "index": img_index
            })

            logger.info(f"Image extraite et analysée : {image_filename}")

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de l'image {img_index} : {e}")
            continue
    logger.info(f"image_list__image_list: {image_list}")
    return image_list
