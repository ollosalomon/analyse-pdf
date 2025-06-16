#!/bin/bash

# init-ollama.sh - Script pour initialiser Ollama avec les modèles requis

echo "🚀 Initialisation d'Ollama..."

# Attendre qu'Ollama soit prêt
echo "⏳ Attente du démarrage d'Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "Ollama pas encore prêt, attente..."
    sleep 5
done

echo "✅ Ollama est démarré!"

# Télécharger le modèle LLaVA
echo "📥 Téléchargement du modèle LLaVA..."
docker exec ollama_llava ollama pull llava:latest

# Télécharger le modèle d'embedding
echo "📥 Téléchargement du modèle d'embedding..."
docker exec ollama_llava ollama pull nomic-embed-text:latest

# Vérifier les modèles installés
echo "🔍 Vérification des modèles installés:"
docker exec ollama_llava ollama list

echo "🎉 Initialisation terminée!"
echo "💡 Vous pouvez maintenant utiliser l'application sur http://localhost:8501"