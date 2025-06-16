# Stage 1: Dependencies
FROM python:3.11-slim as deps

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Installation des dépendances Python
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Application
FROM deps as app

# Copie du code application
COPY . .

# Créer le répertoire output
RUN mkdir -p /app/output

# Expose le port utilisé par Streamlit
EXPOSE 8501

# Healthcheck pour vérifier que Streamlit est prêt
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande pour lancer Streamlit
CMD ["streamlit", "run", "app/interface/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]