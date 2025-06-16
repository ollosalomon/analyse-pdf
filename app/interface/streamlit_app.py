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
        st.session_state.output_dir = "output"  # <-- Dossier persistant
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

    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])
    col1, col2 = st.columns(2)

    with col1:
        batch_size = st.slider("Taille des lots", 10, 100, 20)

    with col2:
        report_name = st.text_input("Nom du rapport", value="rapport_final.md")

    if uploaded_file:
        # Ne pas sauvegarder le PDF sur le disque
        st.session_state.pdf_path = None  # Pas de chemin local

        st.success(f"📄 Fichier chargé: {uploaded_file.name}")

        if st.button("Lancer le traitement", type="primary"):
            # Vérifications préalables
            ollama_ok, models = check_ollama_status()
            if not ollama_ok:
                st.error("❌ Ollama n'est pas accessible. Vérifiez que le service est démarré.")
                return

            # Utiliser le buffer du fichier pour le traitement
            try:
                file_size = uploaded_file.size
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
                    stats = process_pdf_in_batches(uploaded_file, st.session_state.output_dir, batch_size)

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
                        path = generate_report_from_pdf(temp_file, st.session_state.output_dir, batch_size, report_name)
                        st.session_state.report_path = path
                        st.session_state.report_generated = True
                        st.success("📄 Rapport généré avec succès!")
                        st.session_state.current_page = "results"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération du rapport: {str(e)}")
                        with st.expander("🐛 Détails de l'erreur"):
                            import traceback
                            st.code(traceback.format_exc())
                            
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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Pages traitées", stats.get("pages_processed", 0))
        col2.metric("🖼️ Images extraites", stats.get("images_extracted", 0))
        col3.metric("📝 Chunks créés", stats.get("chunks_created", 0))
        col4.metric("⏱️ Durée", f"{stats.get('processing_time_seconds', 0) / 60:.1f} min")

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
        # Réinitialisation propre
        keys_to_reset = ["processing_complete", "report_generated", "pdf_path", 
                        "processing_stats", "report_path"]
        for key in keys_to_reset:
            if key in st.session_state:
                st.session_state[key] = None if key in ["pdf_path", "processing_stats", "report_path"] else False
        st.session_state.current_page = "upload"
        st.rerun()

# def page_chatbot():
#     st.markdown('<h1 class="main-header">💬 Assistant d\'analyse PDF</h1>', unsafe_allow_html=True)
    
#     # Vérifier Ollama
#     ollama_ok, models = check_ollama_status()
#     if not ollama_ok:
#         st.error("❌ Ollama n'est pas accessible. Vérifiez que le service est démarré.")
#         return
    
#     if "llava" not in " ".join(models).lower():
#         st.warning("⚠️ Le modèle LLaVA n'est pas installé. Installez-le avec: `ollama pull llava`")
#         return

#     st.markdown("Posez vos questions sur les documents traités. Le modèle LLaVA répondra en se basant sur la base vectorielle construite.")

#     # Vérifier si un PDF a été traité
#     if not st.session_state.processing_complete:
#         st.warning("⚠️ Veuillez d'abord traiter un PDF avant d'utiliser le chatbot.")
#         return

#     # Interface de chat
#     user_input = st.text_input("💭 Votre question :", key="chat_input", placeholder="Posez votre question sur le document...")

#     if user_input and st.button("📤 Envoyer", type="primary"):
#         try:
#             # Import des modules nécessaires
#             try:
#                 from app.vectorstore.chroma_db import EnhancedVectorDB
#                 from app.vectorstore.embeddings import get_embedding_model
#                 from langchain.chains import RetrievalQA
#                 from langchain_community.llms import Ollama
#             except ImportError as e:
#                 st.error(f"❌ Erreur d'import : {str(e)}")
#                 st.info("Vérifiez que tous les modules requis sont installés dans requirements.txt")
#                 return

#             # Vérifier la base vectorielle
#             vector_db_path = os.path.join(st.session_state.output_dir, "vector_db")
#             if not os.path.exists(vector_db_path):
#                 st.error("❌ Base vectorielle non trouvée. Traitez d'abord un PDF.")
#                 return

#             with st.spinner("🤖 Génération de la réponse..."):
#                 # Charger la base vectorielle
#                 embedding_model = get_embedding_model()
#                 vector_db = EnhancedVectorDB(embedding_model, persist_directory=vector_db_path)
#                 retriever = vector_db.get_retriever()

#                 # Configurer Ollama
#                 llm = Ollama(
#                     model="llava", 
#                     base_url="http://ollama:11434",
#                     temperature=0.7
#                 )

#                 # Créer la chaîne de QA
#                 qa_chain = RetrievalQA.from_chain_type(
#                     llm=llm, 
#                     retriever=retriever,
#                     return_source_documents=True
#                 )
                
#                 # Générer la réponse
#                 result = qa_chain({"query": user_input})
#                 response = result["result"]
#                 sources = result.get("source_documents", [])

#                 # Stocker dans l'historique
#                 st.session_state.chat_history.append(("Vous", user_input))
#                 st.session_state.chat_history.append(("LLaVA", response))
                
#                 # Afficher la réponse
#                 st.success("✅ Réponse générée!")
                
#                 # Redirection pour actualiser l'interface
#                 st.rerun()
                
#         except Exception as e:
#             st.error(f"❌ Erreur dans le chatbot : {str(e)}")
#             if st.checkbox("Afficher les détails de l'erreur"):
#                 st.exception(e)

#     # Affichage de l'historique
#     if st.session_state.chat_history:
#         st.markdown("### 💬 Historique des conversations")
#         for i, (sender, msg) in enumerate(reversed(st.session_state.chat_history[-10:])):  # Limiter à 10 derniers messages
#             if sender == "Vous":
#                 st.markdown(f"**👤 {sender}** : {msg}")
#             else:
#                 st.markdown(f"**🤖 {sender}** : {msg}")
            
#             if i < len(st.session_state.chat_history) - 1:
#                 st.markdown("---")

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
def get_qa_chain(_vector_db):  # Notez le underscore ici
    """Cache la chaîne QA"""
    from langchain.chains import RetrievalQA
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_vector_db.get_retriever(),
        return_source_documents=True
    )
# Modifier la fonction page_chatbot
def page_chatbot():
    st.markdown('<h1 class="main-header">💬 Assistant d\'analyse PDF</h1>', unsafe_allow_html=True)
    
    # Vérifications initiales inchangées...
    
    vector_db_path = os.path.join("output", "vector_db")
    if not os.path.exists(vector_db_path):
        st.warning("⚠️ Aucune base vectorielle trouvée...")
        return

    user_input = st.text_input("💭 Votre question :", key="chat_input")

    if user_input and st.button("📤 Envoyer", type="primary"):
        try:
            with st.spinner("🤖 Génération de la réponse..."):
                # Utiliser les versions cachées
                vector_db = get_vector_db(vector_db_path)
                qa_chain = get_qa_chain(vector_db)  # Plus d'erreur ici
                
                result = qa_chain.invoke({"query": user_input})
                response = result["result"]
                
                st.session_state.chat_history.append(("Vous", user_input))
                st.session_state.chat_history.append(("LLaVA", response))
                st.success("✅ Réponse générée!")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
            if st.checkbox("Détails"):
                st.exception(e)
    # Affichage de l'historique
    if st.session_state.chat_history:
        st.markdown("### 💬 Historique des conversations")
        for i, (sender, msg) in enumerate(reversed(st.session_state.chat_history[-10:])):  # Limiter à 10 derniers messages
            if sender == "Vous":
                st.markdown(f"**👤 {sender}** : {msg}")
            else:
                st.markdown(f"**🤖 {sender}** : {msg}")
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")

            
            
def main():
    init_session_state()
    
    # Afficher le statut d'Ollama
    show_ollama_status()
    
    # Navigation
    menu = st.sidebar.radio("🧭 Navigation", ["📤 Upload", "📊 Résultats", "🤖 Chatbot"])
    
    if menu == "📤 Upload":
        st.session_state.current_page = "upload"
    elif menu == "📊 Résultats":
        st.session_state.current_page = "results"
    elif menu == "🤖 Chatbot":
        st.session_state.current_page = "chatbot"

    # Affichage des pages
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