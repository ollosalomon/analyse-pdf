# Wrapper autour de ChromaDB
import os
import json
from langchain_community.vectorstores import Chroma
import logging
import uuid

logger = logging.getLogger("chroma_db")
logging.basicConfig(level=logging.INFO)

class EnhancedVectorDB:
    def __init__(self, embedding_model, persist_directory="./output/vector_db"):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.db = None
        self.metadata_index = {}
    
    def initialize_db(self):
        """
        Initialise ou charge la base de données vectorielle.
        """
        if os.path.exists(self.persist_directory):
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
            metadata_path = os.path.join(self.persist_directory, "metadata_index.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata_index = json.load(f)
        else:
            self.db = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            os.makedirs(self.persist_directory, exist_ok=True)
    
    def add_documents(self, documents, batch_size=100):
        """
        Ajoute des documents par lots à la base vectorielle.
        """
        if self.db is None:
            self.initialize_db()
        
        total_docs = len(documents)
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:min(i + batch_size, total_docs)]
            
            for doc in batch_docs:
                doc_id = doc.metadata.get('id', str(hash(doc.page_content)))
                
                if 'chapter' in doc.metadata:
                    chapter = doc.metadata['chapter']
                    if chapter not in self.metadata_index:
                        self.metadata_index[chapter] = []
                    self.metadata_index[chapter].append(doc_id)
                
                if 'page' in doc.metadata:
                    page = doc.metadata['page']
                    if 'pages' not in self.metadata_index:
                        self.metadata_index['pages'] = {}
                    if page not in self.metadata_index['pages']:
                        self.metadata_index['pages'][page] = []
                    self.metadata_index['pages'][page].append(doc_id)
            
            ids = [doc.metadata.get('id', f"{uuid.uuid4()}") for doc in batch_docs]
            self.db.add_documents(documents=batch_docs, ids=ids)
            self.db.persist()
            logger.info(f"{len(batch_docs)} documents ajoutés et persistés dans la base vectorielle.")
            
            metadata_path = os.path.join(self.persist_directory, "metadata_index.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_index, f)
    
    def hybrid_search(self, query, filters=None, k=5):
        """
        Recherche hybride avec filtres de métadonnées.
        """
        if self.db is None:
            self.initialize_db()
        
        where_clause = {}
        if filters:
            for key, value in filters.items():
                where_clause[key] = value
        
        return self.db.similarity_search(
            query=query,
            k=k,
            filter=where_clause if where_clause else None
        )
    
    def search_by_page_range(self, query, start_page, end_page, k=5):
        return self.hybrid_search(query, filters={"page": {"$gte": start_page, "$lte": end_page}}, k=k)
    
    def search_by_chapter(self, query, chapter_title, k=5):
        return self.hybrid_search(query, filters={"chapter": chapter_title}, k=k)
    
    def get_retriever(self, search_type="similarity", search_kwargs=None):
        """
        Retourne un retriever LangChain pour utilisation avec les chaînes QA.
        """
        if self.db is None:
            self.initialize_db()
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def get_vectorstore(self):
        """
        Retourne l'instance Chroma directement si nécessaire.
        """
        if self.db is None:
            self.initialize_db()
        return self.db




# # Wrapper autour de ChromaDB
# import os
# import json
# from langchain_community.vectorstores import Chroma
# import logging
# import uuid

# logger = logging.getLogger("chroma_db")
# logging.basicConfig(level=logging.INFO)

# class EnhancedVectorDB:
#     def __init__(self, embedding_model, persist_directory="./output/vector_db"):
#         self.embedding_model = embedding_model
#         self.persist_directory = persist_directory
#         self.db = None
#         self.metadata_index = {}

#     def initialize_db(self):
#         """
#         Initialise ou charge la base de données vectorielle.
#         """
#         if os.path.exists(self.persist_directory):
#             self.db = Chroma(
#                 persist_directory=self.persist_directory,
#                 embedding_function=self.embedding_model
#             )
#             metadata_path = os.path.join(self.persist_directory, "metadata_index.json")
#             if os.path.exists(metadata_path):
#                 with open(metadata_path, 'r') as f:
#                     self.metadata_index = json.load(f)
#         else:
#             self.db = Chroma(
#                 embedding_function=self.embedding_model,
#                 persist_directory=self.persist_directory
#             )
#             os.makedirs(self.persist_directory, exist_ok=True)

#     def add_documents(self, documents, batch_size=100):
#         """
#         Ajoute des documents par lots à la base vectorielle.
#         """
#         if self.db is None:
#             self.initialize_db()

#         total_docs = len(documents)

#         for i in range(0, total_docs, batch_size):
#             batch_docs = documents[i:min(i + batch_size, total_docs)]

#             for doc in batch_docs:
#                 doc_id = doc.metadata.get('id', str(hash(doc.page_content)))

#                 if 'chapter' in doc.metadata:
#                     chapter = doc.metadata['chapter']
#                     if chapter not in self.metadata_index:
#                         self.metadata_index[chapter] = []
#                     self.metadata_index[chapter].append(doc_id)

#                 if 'page' in doc.metadata:
#                     page = doc.metadata['page']
#                     if 'pages' not in self.metadata_index:
#                         self.metadata_index['pages'] = {}
#                     if page not in self.metadata_index['pages']:
#                         self.metadata_index['pages'][page] = []
#                     self.metadata_index['pages'][page].append(doc_id)

#             ids = [doc.metadata.get('id', f"{uuid.uuid4()}") for doc in batch_docs]

#             self.db.add_documents(documents=batch_docs, ids=ids)
#             self.db.persist()
#             logger.info(f"{len(batch_docs)} documents ajoutés et persistés dans la base vectorielle.")

#             metadata_path = os.path.join(self.persist_directory, "metadata_index.json")
#             with open(metadata_path, 'w') as f:
#                 json.dump(self.metadata_index, f)

#     def hybrid_search(self, query, filters=None, k=5):
#         """
#         Recherche hybride avec filtres de métadonnées.
#         """
#         if self.db is None:
#             self.initialize_db()

#         where_clause = {}
#         if filters:
#             for key, value in filters.items():
#                 where_clause[key] = value

#         return self.db.similarity_search(
#             query=query,
#             k=k,
#             filter=where_clause if where_clause else None
#         )

#     def search_by_page_range(self, query, start_page, end_page, k=5):
#         return self.hybrid_search(query, filters={"page": {"$gte": start_page, "$lte": end_page}}, k=k)

#     def search_by_chapter(self, query, chapter_title, k=5):
#         return self.hybrid_search(query, filters={"chapter": chapter_title}, k=k)
