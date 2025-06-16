# Documentation du projet



# 📘 RAG PDF Analyzer

Ce projet permet de traiter des fichiers PDF volumineux (jusqu’à 2000 pages), d’en extraire le contenu textuel et visuel, de les vectoriser, puis de générer automatiquement un rapport analytique structuré à l’aide d’un LLM (comme Gemini).

## 🧱 Fonctionnalités

- Extraction par lots (batch) de texte et d’images depuis un PDF
- Indexation vectorielle (ChromaDB + embeddings GPT4All)
- Analyse de structure du document (chapitres/sections)
- Génération de rapport synthétique via Gemini (Google Generative AI)
- Interface Streamlit conviviale
- Ligne de commande alternative

## 🚀 Démarrage rapide

```bash
# 1. Cloner et configurer l’environnement
cp .env.example .env  # ou éditer .env
pip install -r requirements.txt

# 2. Lancer l’interface web
streamlit run app/interface/streamlit_app.py

# 3. Ou utiliser le CLI
python app/main.py --help
```


rag_pdf_analyzer/
│
├── app/                        # Code principal de l'application
│   ├── __init__.py             # Rend le package importable
│   ├── config.py               # Chargement des variables d'environnement (.env)
│   ├── main.py                 # Point d'entrée global (redirige vers CLI ou web)
│
│   ├── ingestion/              # Étape d'extraction et de prétraitement
│   │   ├── __init__.py
│   │   ├── pdf_loader.py       # Découpe du PDF par lots (batchs de pages)
│   │   ├── image_extractor.py  # Extraction et description des images de chaque page
│   │   ├── chunker.py          # Découpe des textes en chunks hiérarchiques (chapitres/sections)
│   │   └── structure_parser.py # Détection des chapitres/sections via regex
│
│   ├── vectorstore/            # Indexation et recherche vectorielle
│   │   ├── __init__.py
│   │   ├── embeddings.py       # Chargement du modèle d'embeddings (GPT4All ou autre)
│   │   ├── chroma_db.py        # Gestion de la base Chroma + métadonnées
│   │   └── vector_utils.py     # Fonctions utilitaires de recherche (filtrée, par chapitre, etc.)
│
│   ├── report/                 # Génération de rapports LLM
│   │   ├── __init__.py
│   │   ├── generator.py        # Générateur avec mémoire de contexte (structure + résumés)
│   │   └── prompts.py          # Prompts LLM structurés (extraction de plan, résumé, synthèse finale)
│
│   └── interface/              # Interfaces utilisateur (CLI et Web)
│       ├── __init__.py
│       ├── streamlit_app.py    # Application Streamlit pour visualisation et rapport
│       └── cli.py              # Interface ligne de commande avec argparse
│
├── output/                     # Données générées automatiquement
│   ├── extracted_text/         # Fichiers texte extraits du PDF
│   ├── extracted_images/       # Images extraites et annotées par Gemini
│   ├── vector_db/              # Chunks vectorisés persistés via Chroma
│   └── metadata/               # Statistiques de traitement, état d’avancement, logs
│
├── tests/                      # Tests unitaires automatisés
│   ├── __init__.py
│   ├── test_loader.py          # Test de lecture PDF et batch
│   ├── test_chunking.py        # Test de la structure hiérarchique et des chunks
│   └── test_generation.py      # Test du générateur de rapport
│
├── .env                        # Variables sensibles (clé API Gemini, chemins, etc.)
├── requirements.txt            # Toutes les dépendances Python requises
├── README.md                   # Ce fichier de documentation
└── setup.py                    # Rend le projet installable en tant que module Python
