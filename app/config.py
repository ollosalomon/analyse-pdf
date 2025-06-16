# Configuration globale du projet
import os
from dotenv import load_dotenv

# Charger les variables d'environnement à partir d'un fichier .env
load_dotenv()

# Clés API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Répertoires de travail
DATA_PATH = os.getenv("DATA_PATH", "data_sources")
CHROMA_PATH = os.getenv("CHROMA_PATH", "output/vector_db")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# Modèle d'embedding utilisé
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2.gguf2.f16.gguf")
