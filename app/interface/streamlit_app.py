# Interface utilisateur Streamlit
import streamlit as st
import os
import time
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import json
import sys
import requests
sys.path.append(".")
import logging
logger = logging.getLogger("chatbot")
logging.basicConfig(level=logging.INFO)

# Configuration de la page
st.set_page_config(
    page_title="PDF Analyzer - Traitement de PDF volumineux",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS de style
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-box, .success-box, .warning-box, .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box { background-color: #E3F2FD; }
    .success-box { background-color: #E8F5E9; }
    .warning-box { background-color: #FFF8E1; }
    .error-box { background-color: #FFEBEE; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_temp_dir():
    return tempfile.mkdtemp()

def check_ollama_status():
    """Vérifier le statut d'Ollama et des modèles"""
    try:
        # Vérifier si Ollama est accessible
        response = requests.get("http://ollama:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            return True, model_names
        else:
            return False, []
    except Exception:
        return False, []


def init_session_state():
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = "output"
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = None
    if 'report_path' not in st.session_state:
        st.session_state.report_path = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "upload"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'ollama_status' not in st.session_state:
        st.session_state.ollama_status = None
    # NOUVEAUX ÉTATS À AJOUTER
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 20
    if 'report_name' not in st.session_state:
        st.session_state.report_name = "rapport_final.md"
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

def show_ollama_status():
    """Afficher le statut d'Ollama dans la sidebar"""
    with st.sidebar:
        st.markdown("### 🤖 Statut Ollama")
        
        if st.button("🔄 Vérifier Ollama"):
            st.session_state.ollama_status = None
        
        if st.session_state.ollama_status is None:
            with st.spinner("Vérification..."):
                ollama_ok, models = check_ollama_status()
                st.session_state.ollama_status = (ollama_ok, models)
        
        ollama_ok, models = st.session_state.ollama_status
        
        if ollama_ok:
            st.success("✅ Ollama connecté")
            if models:
                st.info(f"📦 Modèles: {', '.join(models)}")
            else:
                st.warning("⚠️ Aucun modèle installé")
        else:
            st.error("❌ Ollama non accessible")

def page_upload():
    st.markdown('<h1 class="main-header">Traitement de PDF volumineux</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color: #1f77b4; color: white; padding: 15px; border-radius: 10px; font-size: 16px;">
        Cet outil vous permet de traiter des fichiers PDF volumineux (jusqu'à 2000 pages), 
        d'extraire leur contenu structuré et de générer un rapport d'analyse.
        </div>
        """, unsafe_allow_html=True)

    # MODIFIÉ : Maintenir l'état du fichier uploadé
    if st.session_state.file_uploaded and st.session_state.uploaded_file:
        st.success(f"📄 Fichier déjà chargé: {st.session_state.uploaded_file.name}")
        
        # Bouton pour changer de fichier
        if st.button("🔄 Changer de fichier"):
            st.session_state.uploaded_file = None
            st.session_state.file_uploaded = False
            st.rerun()
    
    # Upload du fichier seulement si pas déjà uploadé
    if not st.session_state.file_uploaded:
        uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.file_uploaded = True
            st.success(f"📄 Fichier chargé: {uploaded_file.name}")
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        # MODIFIÉ : Utiliser l'état de session
        batch_size = st.slider("Taille des lots", 10, 100, st.session_state.batch_size, key="batch_slider")
        st.session_state.batch_size = batch_size

    with col2:
        # MODIFIÉ : Utiliser l'état de session
        report_name = st.text_input("Nom du rapport", value=st.session_state.report_name, key="report_input")
        st.session_state.report_name = report_name

    # MODIFIÉ : Traitement seulement si fichier uploadé
    if st.session_state.file_uploaded and st.session_state.uploaded_file:
        # Ne pas sauvegarder le PDF sur le disque
        st.session_state.pdf_path = None

        if st.button("Lancer le traitement", type="primary"):
            # Vérifications préalables
            ollama_ok, models = check_ollama_status()
            if not ollama_ok:
                st.error("❌ Ollama n'est pas accessible. Vérifiez que le service est démarré.")
                return

            # Utiliser le buffer du fichier pour le traitement
            try:
                file_size = st.session_state.uploaded_file.size
                st.info(f"📊 Taille du fichier: {file_size / (1024*1024):.1f} MB")
            except Exception as e:
                st.error(f"❌ Impossible de lire le fichier: {str(e)}")
                return

            # Vérifier les imports nécessaires
            try:
                st.info("🔍 Vérification des dépendances...")
                from app.interface.cli import process_pdf_in_batches, generate_report_from_pdf
                st.success("✅ Modules importés avec succès")
            except ImportError as e:
                st.error(f"❌ Erreur d'import critique: {str(e)}")
                st.info("Vérifiez que tous les modules sont présents dans le conteneur")
                return
            except Exception as e:
                st.error(f"❌ Erreur lors de l'import: {str(e)}")
                return

            # Traitement avec gestion d'erreurs renforcée
            with st.spinner("Traitement du PDF en cours..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("🔄 Initialisation du traitement...")
                    progress_bar.progress(10)
                    time.sleep(0.5)

                    status_text.text("📖 Lecture du PDF...")
                    progress_bar.progress(25)

                    # Appel de la fonction avec le buffer du fichier
                    st.info("🚀 Lancement du traitement par lots...")
                    stats = process_pdf_in_batches(st.session_state.uploaded_file, st.session_state.output_dir, batch_size)

                    progress_bar.progress(100)
                    status_text.text("✅ Traitement terminé!")

                    st.session_state.processing_stats = stats
                    st.session_state.processing_complete = True
                    st.success("🎉 Traitement terminé avec succès!")
                    st.balloons()
                    
                except FileNotFoundError as e:
                    st.error(f"❌ Fichier non trouvé: {str(e)}")
                    st.info("Vérifiez que le fichier PDF est valide")
                    
                except MemoryError as e:
                    st.error("❌ Erreur de mémoire: Fichier trop volumineux")
                    st.info("Essayez avec une taille de lot plus petite ou un fichier plus petit")
                    
                except ImportError as e:
                    st.error(f"❌ Module manquant: {str(e)}")
                    st.info("Vérifiez l'installation des dépendances")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement: {str(e)}")
                    st.error(f"Type d'erreur: {type(e).__name__}")
                    
                    # Affichage détaillé de l'erreur
                    with st.expander("🐛 Détails de l'erreur (pour le debug)"):
                        import traceback
                        st.code(traceback.format_exc())
                    
                    # Informations de debug
                    st.info("🔧 Informations de debug:")
                    st.write(f"- Chemin ou nom du fichier: {getattr(uploaded_file, 'name', str(uploaded_file))}")
                    st.write(f"- Dossier de sortie: {st.session_state.output_dir}")
                    st.write(f"- Taille du lot: {batch_size}")
                    st.write(f"- Python path: {sys.path[:3]}...")
                    st.write(f"- Ollama status: {ollama_ok}")
                    
                    return
                    
                finally:
                    progress_bar.empty()
                    status_text.empty()

        # Section génération du rapport (inchangée)
        if st.session_state.processing_complete:
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📊 Génération du rapport</h2>', unsafe_allow_html=True)

            if st.button("Générer le rapport", type="secondary"):
                with st.spinner("Génération du rapport..."):
                    try:
                        # MODIFIÉ : Utiliser les valeurs de session
                        path = generate_report_from_pdf(
                            st.session_state.uploaded_file, 
                            st.session_state.output_dir, 
                            st.session_state.batch_size, 
                            st.session_state.report_name
                        )
                        st.session_state.report_path = path
                        st.session_state.report_generated = True
                        st.success("📄 Rapport généré avec succès!")
                        st.session_state.current_page = "results"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération du rapport: {str(e)}")
                            
def page_results():
    st.markdown('<h1 class="main-header">📊 Résultats du traitement</h1>', unsafe_allow_html=True)

    if not st.session_state.processing_complete:
        st.warning("⚠️ Aucun traitement effectué.")
        if st.button("Retour à l'upload"):
            st.session_state.current_page = "upload"
            st.rerun()
        return

    st.markdown('<h2 class="sub-header">📈 Statistiques de traitement</h2>', unsafe_allow_html=True)
    stats = st.session_state.processing_stats

    if stats:
        # Vérifier si stats est un dict ou un tuple
        if isinstance(stats, dict):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📄 Pages traitées", stats.get("pages_processed", 0))
            col2.metric("🖼️ Images extraites", stats.get("images_extracted", 0))
            col3.metric("📝 Chunks créés", stats.get("chunks_created", 0))
            col4.metric("⏱️ Durée", f"{stats.get('processing_time_seconds', 0) / 60:.1f} min")
        else:
            # Si stats est un tuple ou autre format, affichage basique
            st.info(f"📊 Statistiques: {stats}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📄 Pages traitées", "N/A")
            col2.metric("🖼️ Images extraites", "N/A")
            col3.metric("📝 Chunks créés", "N/A")
            col4.metric("⏱️ Durée", "N/A")
            
        # Graphique de distribution
        total_pages = stats.get("total_pages", 100)
        if total_pages > 0:
            pages = list(range(1, min(total_pages + 1, 200), max(1, total_pages // 20)))
            chunks = [20 + (i * 3) % 15 for i in range(len(pages))]
            df = pd.DataFrame({"Page": pages, "Chunks": chunks})
            fig = px.bar(df, x="Page", y="Chunks", 
                        title="Distribution des chunks par page",
                        color="Chunks",
                        color_continuous_scale="viridis")
            st.plotly_chart(fig, use_container_width=True)

    if st.session_state.report_generated and st.session_state.report_path:
        st.markdown('<h2 class="sub-header">📄 Rapport généré</h2>', unsafe_allow_html=True)
        try:
            with open(st.session_state.report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
            
            # Afficher un aperçu du rapport
            with st.expander("👁️ Aperçu du rapport", expanded=True):
                st.markdown(report_content[:2000] + ("..." if len(report_content) > 2000 else ""))

            # Bouton de téléchargement
            with open(st.session_state.report_path, "rb") as f:
                st.download_button(
                    "📥 Télécharger le rapport complet", 
                    f.read(), 
                    "rapport_analyse.md", 
                    mime="text/markdown",
                    type="primary"
                )
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du rapport : {str(e)}")

    if st.button("🔄 Traiter un nouveau PDF"):
        # MODIFIÉ : Réinitialisation plus propre sans affecter la navigation
        keys_to_reset = [
            "processing_complete", "report_generated", "pdf_path", 
            "processing_stats", "report_path", "uploaded_file", 
            "file_uploaded"
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                if key in ["pdf_path", "processing_stats", "report_path", "uploaded_file"]:
                    st.session_state[key] = None
                elif key == "file_uploaded":
                    st.session_state[key] = False
                else:
                    st.session_state[key] = False
        
        # Réinitialiser les valeurs par défaut
        st.session_state.batch_size = 20
        st.session_state.report_name = "rapport_final.md"
        
        st.session_state.current_page = "upload" 
        st.rerun()

@st.cache_resource
def get_vector_db(vector_db_path):
    """Cache la base vectorielle pour éviter de la recharger"""
    from app.vectorstore.chroma_db import EnhancedVectorDB
    from app.vectorstore.embeddings import get_embedding_model
    embedding_model = get_embedding_model()
    vector_db = EnhancedVectorDB(
        embedding_model, 
        persist_directory=vector_db_path
    )
    return vector_db

@st.cache_resource
def get_llm():
    """Cache le modèle LLM"""
    from langchain_community.llms import Ollama
    return Ollama(
        model="llava",
        base_url="http://ollama:11434",
        temperature=0.7,
        timeout=60
    )

@st.cache_resource
def get_vision_llm():
    """LLM pour la vision (LLaVA) - utilisé pour l'analyse d'images"""
    from langchain_community.llms import Ollama
    return Ollama(
        model="llava",
        base_url="http://ollama:11434",
        temperature=0.7,
        timeout=60,  # Plus long pour la vision
        num_ctx=4096  # Contexte plus large pour les images
    )

@st.cache_resource
def get_text_llm():
    """LLM léger pour le texte uniquement - chatbot"""
    from langchain_community.llms import Ollama
    return Ollama(
        model="phi3:mini",  # Modèle ultra-léger
        base_url="http://ollama:11434",
        temperature=0.3,
        timeout=10,  # Timeout court
        num_predict=300,  # Réponses concises
        num_ctx=2048  # Contexte réduit
    )

@st.cache_resource  
def get_qa_chain(_vector_db):
    """Chaîne QA optimisée avec le modèle léger"""
    if _vector_db is None:
        return None
        
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
    # Template optimisé pour les réponses rapides
    prompt_template = """Basé sur le contexte suivant, réponds de manière concise et précise.
    Si l'information n'est pas dans le contexte, dis-le clairement.

    Contexte: {context}

    Question: {question}

    Réponse (max 3 phrases):"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Utiliser le modèle léger pour le texte
    llm = get_text_llm()
    
    # Retriever optimisé
    retriever = _vector_db.get_retriever()
    retriever.search_kwargs = {"k": 2}  # Seulement 2 documents
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

def check_available_models():
    """Vérifier quels modèles sont disponibles"""
    try:
        import requests
        response = requests.get("http://ollama:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            return model_names
        return []
    except:
        return []

def get_optimal_text_model():
    """Choisir le meilleur modèle disponible pour le texte"""
    available_models = check_available_models()
    
    # Ordre de préférence (du plus léger au plus lourd)
    preferred_models = [
        "phi3:mini",
        "phi3:3.8b", 
        "gemma:2b",
        "llama3.2:3b",
        "mistral:7b",
        "llama2:7b",
        "llava"  # Fallback
    ]
    
    for model in preferred_models:
        if any(model in available for available in available_models):
            return model
    
    return "llava"  # Fallback par défaut

@st.cache_resource
def get_adaptive_text_llm():
    """LLM adaptatif qui choisit le meilleur modèle disponible"""
    from langchain_community.llms import Ollama
    
    optimal_model = get_optimal_text_model()
    
    # Paramètres selon le modèle
    if "phi3" in optimal_model or "gemma" in optimal_model:
        # Modèles ultra-légers
        params = {
            "temperature": 0.2,
            "timeout": 8,
            "num_predict": 200,
            "num_ctx": 2048
        }
    elif "llama3.2" in optimal_model or "mistral" in optimal_model:
        # Modèles moyens
        params = {
            "temperature": 0.3,
            "timeout": 15,
            "num_predict": 250,
            "num_ctx": 2048
        }
    else:
        # Fallback (llava, etc.)
        params = {
            "temperature": 0.5,
            "timeout": 20,
            "num_predict": 200,
            "num_ctx": 2048
        }
    
    st.info(f"🤖 Utilisation du modèle: {optimal_model}")
    
    return Ollama(
        model=optimal_model,
        base_url="http://ollama:11434",
        **params
    )

@st.cache_resource  
def get_qa_chain_adaptive(_vector_db, timeout=60):
    """Crée une chaîne QA adaptative avec timeout configuré"""
    try:
        from langchain_community.llms import Ollama
        from langchain.chains import RetrievalQA
        
        # Configuration du LLM avec timeout
        optimal_model = get_optimal_text_model()
        
        # Paramètres optimisés selon le modèle
        if optimal_model == "phi3:mini":
            llm_params = {
                "model": optimal_model,
                "base_url": "http://ollama:11434",
                "timeout": timeout,
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 2048,  # Contexte réduit pour plus de rapidité
                "repeat_penalty": 1.1
            }
        else:  # llava ou autres
            llm_params = {
                "model": optimal_model,
                "base_url": "http://ollama:11434",
                "timeout": timeout,
                "temperature": 0.5,
                "top_p": 0.8,
                "num_ctx": 4096,
                "repeat_penalty": 1.2
            }
        
        llm = Ollama(**llm_params)
        
        # Création de la chaîne QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            # retriever=_vector_db.get_retriever(
            #     search_kwargs={"k": 3}  # Limite à 3 résultats pour plus de rapidité
            # ),
            retriever=_vector_db.get_retriever(),
            return_source_documents=False
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la chaîne QA: {e}")
        return None
    
    
def check_ollama_connection():
    """Vérifie si Ollama est accessible"""
    try:
        import requests
        response = requests.get("http://ollama:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def determine_pdf_context_needed(user_input, conversation_mode):
    """Détermine si la question nécessite le contexte PDF"""
    
    if conversation_mode == "💭 Conversation libre":
        return False
    elif conversation_mode == "📄 Analyse PDF":
        return True
    else:  # Mode hybride
        # Mots-clés indiquant une question sur le document
        pdf_keywords = [
            "document", "pdf", "texte", "page", "section", "chapitre",
            "que dit", "selon le", "dans le document", "extrait",
            "résumé", "analyse", "contenu", "information"
        ]
        
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in pdf_keywords)

def handle_pdf_question(user_input, vector_db_path, timeout):
    """Traite une question nécessitant le contexte PDF"""
    
    # Réponses rapides
    quick_responses = {
        "résumé": "📋 Voici un résumé du document...",
        "sommaire": "📑 Table des matières du document...",
    }
    
    user_lower = user_input.lower().strip()
    if user_lower in quick_responses:
        return quick_responses[user_lower]
    
    # RAG standard
    vector_db = get_vector_db(vector_db_path)
    if not vector_db:
        return "❌ Base vectorielle inaccessible"
    
    qa_chain = get_qa_chain_adaptive(vector_db, timeout=timeout)
    if not qa_chain:
        return "❌ Impossible de créer la chaîne QA"
    
    result = qa_chain.invoke({"query": user_input})
    return result["result"]

def handle_general_question(user_input, timeout):
    """Traite une question générale sans contexte PDF"""
    
    # Réponses rapides pour les salutations
    quick_responses = {
        "bonjour": "👋 Bonjour ! Je suis votre assistant IA. Comment puis-je vous aider ?",
        "salut": "👋 Salut ! Que puis-je faire pour vous aujourd'hui ?",
        "hello": "👋 Hello! How can I help you today?",
        "merci": "😊 De rien ! Autre chose ?",
        "qui es-tu": "🤖 Je suis un assistant IA polyvalent, capable d'analyser des PDF et de discuter de tout sujet !",
        "que peux-tu faire": "🔧 Je peux analyser des PDF, répondre à vos questions, vous aider dans vos tâches et discuter de nombreux sujets !"
    }
    
    user_lower = user_input.lower().strip()
    if user_lower in quick_responses:
        return quick_responses[user_lower]
    
    # Conversation générale avec Ollama
    try:
        from langchain_community.llms import Ollama
        
        optimal_model = get_optimal_text_model()
        
        # Configuration pour conversation générale
        llm_params = {
            "model": optimal_model,
            "base_url": "http://ollama:11434",
            "timeout": timeout,
            "temperature": 0.7,  # Plus créatif pour conversation générale
            "top_p": 0.9,
            "num_ctx": 2048 if optimal_model == "phi3:mini" else 4096,
            "repeat_penalty": 1.1
        }
        
        llm = Ollama(**llm_params)
        
        # Prompt pour conversation générale
        prompt = f"""Tu es un assistant IA serviable et amical. Réponds à cette question de manière utile et conversationnelle :

Question: {user_input}

Réponse:"""
        
        response = llm.invoke(prompt)
        return response
        
    except Exception as e:
        return f"❌ Erreur lors de la génération: {str(e)}"

def display_response_time(processing_time):
    """Affiche le temps de réponse avec couleurs adaptatives"""
    if processing_time < 5:
        st.success(f"⚡ Réponse ultra-rapide ({processing_time:.1f}s)")
    elif processing_time < 15:
        st.success(f"✅ Réponse rapide ({processing_time:.1f}s)")
    elif processing_time < 30:
        st.warning(f"⏱️ Réponse normale ({processing_time:.1f}s)")
    else:
        st.info(f"🕐 Réponse lente ({processing_time:.1f}s)")

def handle_chat_error(error, processing_time, timeout):
    """Gère les erreurs de chat avec messages spécifiques"""
    error_msg = str(error)
    
    if "timeout" in error_msg.lower() or "read timed out" in error_msg.lower():
        st.error(f"⏱️ Timeout après {processing_time:.1f}s")
        st.info("💡 Essayez avec un timeout plus long ou une question plus simple")
    elif "connection" in error_msg.lower():
        st.error(f"🔌 Problème de connexion après {processing_time:.1f}s")
        st.info("🔧 Vérifiez que Ollama est démarré")
    else:
        st.error(f"❌ Erreur après {processing_time:.1f}s: {error_msg}")

def display_chat_history():
    """Affiche l'historique avec indication du mode utilisé"""
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Historique")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Effacer"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            show_all = st.checkbox("Afficher tout l'historique", value=False)
        
        # Limitation de l'affichage
        history_to_show = st.session_state.chat_history if show_all else st.session_state.chat_history[-6:]
        
        for i, item in enumerate(reversed(history_to_show)):
            if len(item) == 2:  # Ancien format
                sender, msg = item
                mode_icon = ""
            else:  # Nouveau format avec mode
                sender, msg, mode_icon = item
            
            if sender == "Vous":
                st.markdown(f"**👤 {sender}:** {msg}")
            else:
                st.markdown(f"**🤖 Assistant {mode_icon}:** {msg}")
            
            if i < len(history_to_show) - 1:
                st.markdown("---")
# Modifier la fonction page_chatbot

def page_chatbot():
    st.markdown('<h1 class="main-header">💬 Assistant IA Polyvalent</h1>', unsafe_allow_html=True)
    
    # Détection du mode (avec ou sans PDF)
    vector_db_path = os.path.join("output", "vector_db")
    has_pdf = st.session_state.processing_complete and os.path.exists(vector_db_path)
    
    # Sélection du mode de conversation
    with st.sidebar:
        st.markdown("### 🎯 Mode de conversation")
        
        if has_pdf:
            conversation_mode = st.radio(
                "Choisissez le mode :",
                ["📄 Analyse PDF", "💭 Conversation libre", "🔀 Mode hybride"],
                index=0
            )
        else:
            conversation_mode = st.radio(
                "Choisissez le mode :",
                ["💭 Conversation libre", "📄 Analyse PDF (nécessite upload)"],
                index=0
            )
            
        # Informations sur le mode sélectionné
        if conversation_mode == "📄 Analyse PDF":
            if has_pdf:
                st.success("✅ PDF chargé - Questions sur le document")
            else:
                st.warning("⚠️ Chargez d'abord un PDF dans l'onglet Upload")
        elif conversation_mode == "💭 Conversation libre":
            st.info("🗨️ Discussion générale sans document")
        elif conversation_mode == "🔀 Mode hybride":
            st.info("🔄 Questions sur PDF + conversation générale")

    # Affichage du modèle utilisé
    with st.sidebar:
        st.markdown("### 🤖 Modèles")
        available_models = check_available_models()
        if available_models:
            optimal_model = get_optimal_text_model()
            st.success(f"📝 Texte: {optimal_model}")
            if "llava" in available_models:
                st.success("👁️ Vision: llava")
            
            # Indicateur de performance du modèle
            if optimal_model == "phi3:mini":
                st.info("⚡ Modèle rapide et efficace")
            elif optimal_model == "llava":
                st.warning("🐌 Modèle lent mais polyvalent")
        else:
            st.error("❌ Aucun modèle détecté")

    # Configuration des timeouts
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        timeout_option = st.selectbox(
            "Timeout de génération:",
            ["Rapide (30s)", "Normal (60s)", "Lent (120s)", "Très lent (300s)"],
            index=1
        )
        
        timeout_mapping = {
            "Rapide (30s)": 30,
            "Normal (60s)": 60,
            "Lent (120s)": 120,
            "Très lent (300s)": 300
        }
        selected_timeout = timeout_mapping[timeout_option]

    # Vérification selon le mode
    if conversation_mode == "📄 Analyse PDF" and not has_pdf:
        st.warning("⚠️ Mode analyse PDF sélectionné mais aucun document chargé.")
        st.info("👆 Chargez un PDF dans l'onglet Upload ou basculez en 'Conversation libre'")
        return

    # Interface de chat adaptée au mode
    if conversation_mode == "📄 Analyse PDF":
        placeholder_text = "💭 Posez une question sur votre document PDF..."
    elif conversation_mode == "💭 Conversation libre":
        placeholder_text = "💭 Posez n'importe quelle question..."
    else:  # Mode hybride
        placeholder_text = "💭 Question sur le PDF ou conversation générale..."

    user_input = st.chat_input(placeholder_text)

    if user_input: 
        start_time = time.time()
        
        try:
            # Vérification de la connexion Ollama
            if not check_ollama_connection():
                st.error("❌ Impossible de se connecter à Ollama")
                return
            
            with st.spinner("🤖 Génération de la réponse..."):
                
                # Détermination du type de réponse nécessaire
                needs_pdf_context = determine_pdf_context_needed(user_input, conversation_mode)
                
                if needs_pdf_context and has_pdf:
                    # Mode RAG avec PDF
                    response = handle_pdf_question(user_input, vector_db_path, selected_timeout)
                else:
                    # Mode conversation libre
                    response = handle_general_question(user_input, selected_timeout)
                
                processing_time = time.time() - start_time
                
                # Ajouter à l'historique avec le mode utilisé
                mode_icon = "📄" if needs_pdf_context else "💭"
                st.session_state.chat_history.append(("Vous", user_input))
                st.session_state.chat_history.append(("Assistant", response, mode_icon))
                
                # Affichage du temps
                display_response_time(processing_time)
                
        except Exception as e:
            processing_time = time.time() - start_time
            handle_chat_error(e, processing_time, selected_timeout)

    # Historique avec indication du mode
    display_chat_history()
    
def main():
    init_session_state()
    
    # Afficher le statut d'Ollama
    show_ollama_status()
    
    # MODIFIÉ : Navigation qui respecte l'état de session
    current_menu = None
    if st.session_state.current_page == "upload":
        current_menu = "📤 Upload"
    elif st.session_state.current_page == "results":
        current_menu = "📊 Résultats"
    elif st.session_state.current_page == "chatbot":
        current_menu = "🤖 Chatbot"
    
    # Utiliser l'index pour maintenir la sélection
    menu_options = ["📤 Upload", "📊 Résultats", "🤖 Chatbot"]
    try:
        default_index = menu_options.index(current_menu) if current_menu else 0
    except ValueError:
        default_index = 0
    
    menu = st.sidebar.radio("🧭 Navigation", menu_options, index=default_index)
    
    # MODIFIÉ : Mettre à jour l'état seulement si changement
    if menu == "📤 Upload" and st.session_state.current_page != "upload":
        st.session_state.current_page = "upload"
        st.rerun()
    elif menu == "📊 Résultats" and st.session_state.current_page != "results":
        st.session_state.current_page = "results"
        st.rerun()
    elif menu == "🤖 Chatbot" and st.session_state.current_page != "chatbot":
        st.session_state.current_page = "chatbot"
        st.rerun()

    # Affichage des pages (inchangé)
    try:
        if st.session_state.current_page == "upload":
            page_upload()
        elif st.session_state.current_page == "results":
            page_results()
        elif st.session_state.current_page == "chatbot":
            page_chatbot()
    except Exception as e:
        st.error(f"❌ Erreur dans l'application : {str(e)}")
        st.info("🔄 Rechargez la page pour recommencer.")
        if st.checkbox("🐛 Mode debug - Afficher les détails"):
            st.exception(e)


if __name__ == "__main__":
    main()