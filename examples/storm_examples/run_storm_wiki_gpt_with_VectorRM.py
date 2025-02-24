"""
This STORM Wiki pipeline powered by GPT-3.5/4 and local retrieval model that uses Qdrant.
You need to set up the following environment variables to run this script:
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_API_TYPE: OpenAI API type (e.g., 'openai' or 'azure')
    - QDRANT_API_KEY: Qdrant API key (needed ONLY if online vector store was used)

You will also need an existing Qdrant vector store either saved in a folder locally offline or in a server online.
If not, then you would need a CSV file with documents, and the script is going to create the vector store for you.
The CSV should be in the following format:
content  | title  |  url  |  description
I am a document. | Document 1 | docu-n-112 | A self-explanatory document.
I am another document. | Document 2 | docu-l-13 | Another self-explanatory document.

Notice that the URL will be a unique identifier for the document so ensure different documents have different urls.

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Raw search results from search engine
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_outline.txt           # Outline refined with collected information
        url_to_info.json                # Sources that are used in the final article
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""

import os
import hashlib
import pandas as pd
import chromadb
from argparse import ArgumentParser
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

def check_existing_document(collection, document_id):
    """Vérifie si un document existe déjà dans la collection sans générer de doublon."""
    try:
        existing_data = collection.get(ids=[document_id])
        print(f"🔍 Vérification de l'existence du document {document_id}: {existing_data}")
        if existing_data and "ids" in existing_data and document_id in existing_data["ids"]:
            return document_id  # Retourne l'ID si trouvé
    except Exception as e:
        print(f"❌ Erreur lors de la vérification du document : {e}")
    return None

def check_similar_embeddings(collection, vector, threshold=0.95):
    """Vérifie si un embedding similaire existe déjà dans la collection."""
    try:
        results = collection.query(query_embeddings=[vector], n_results=1)
        if results and "documents" in results and len(results["documents"]) > 0:
            similarity = results["distances"][0]
            if similarity < (1 - threshold):  # Plus la similarité est élevée, moins il y a de doublons
                return False
            else:
                print(f"⚠️ Embedding similaire trouvé avec une similarité de {similarity}")
                return True
    except Exception as e:
        print(f"❌ Erreur lors de la vérification des embeddings similaires : {e}")
    return False

def update_existing_document(collection, document_id, document_content, vector, source):
    """Met à jour un document existant."""
    try:
        collection.update(
            ids=[document_id],
            embeddings=[vector],
            documents=[document_content],
            metadatas=[{"source": source}]
        )
        print(f"🔄 Document mis à jour (ID: {document_id})")
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour du document : {e}")

def add_or_update_document(collection, document_content, embedder, source):
    """Ajoute ou met à jour un document dans la collection sans doublon."""
    document_id = hashlib.md5(document_content.encode()).hexdigest()
    
    # Vérification de l'existence du document par ID
    existing_id = check_existing_document(collection, document_id)
    
    try:
        # Calcul du vecteur pour l'embedding
        vector = embedder.embed_documents([document_content])[0]
        
        # Vérification des embeddings similaires si le document n'existe pas par ID
        if not existing_id:
            if check_similar_embeddings(collection, vector):
                print(f"⚠️ Document similaire trouvé avec un embedding proche. Aucun doublon ajouté.")
                return  # Ne pas ajouter si une similarité est détectée
        
        # Mise à jour ou ajout du document
        if existing_id:
            print(f"🔄 Document déjà existant avec ID {document_id}. Mise à jour...")
            update_existing_document(collection, existing_id, document_content, vector, source)
        else:
            collection.add(
                ids=[document_id],
                embeddings=[vector],
                documents=[document_content],
                metadatas=[{"source": source}]
            )
            print(f"✅ Document ajouté (ID: {document_id})")
    except Exception as e:
        print(f"❌ Erreur lors de l'ajout/mise à jour du document : {e}")

def search_documents(query, collection_name, search_top_k, embedder, client):
    """Effectue une recherche sur les documents indexés."""
    try:
        collection = client.get_or_create_collection(collection_name)
        query_embedding = embedder.embed_documents([query])[0]
        result = collection.query(query_embeddings=[query_embedding], n_results=search_top_k)
        
        documents_with_metadata = []
        if result and isinstance(result, dict) and "documents" in result and "metadatas" in result:
            for doc, metadata in zip(result["documents"], result["metadatas"]):
                if doc:
                    documents_with_metadata.append({
                        "document": doc[0] if isinstance(doc, list) else doc,
                        "metadata": metadata
                    })
        return documents_with_metadata
    except Exception as e:
        print(f"❌ Erreur lors de la recherche de documents : {e}")
        return []

def parse_arguments():
    """Gestion des arguments CLI."""
    parser = ArgumentParser()
    parser.add_argument("--csv-file-path", type=str, default=None, help="Chemin du fichier CSV contenant les documents.")
    parser.add_argument("--collection-name", type=str, default="my_documents", help="Nom de la collection dans ChromaDB.")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Modèle d'embedder à utiliser.")
    parser.add_argument("--search-top-k", type=int, default=3, help="Nombre de documents à récupérer pour la recherche.")
    parser.add_argument("--clear-existing", action='store_true', help="Supprimer les embeddings existants avant d'ajouter de nouveaux documents.")
    return parser.parse_args()

def main(args):
    """Point d'entrée principal du script."""
    llm = OllamaLLM(model="llama2")
    client = chromadb.PersistentClient(path="./chroma_db")
    embedder = HuggingFaceEmbeddings(model_name=args.embedding_model)
    
    if args.clear_existing:
        client.delete_collection(args.collection_name)
        print("✅ Ancienne collection supprimée.")
    
    if args.csv_file_path:
        try:
            df = pd.read_csv(args.csv_file_path, dtype=str).fillna("unknown")
            if not all(col in df.columns for col in ["content", "title", "url", "description"]):
                print("⚠️ Le fichier CSV doit contenir les colonnes: content, title, url, description")
                return
            
            collection = client.get_or_create_collection(args.collection_name)
            for _, row in df.iterrows():
                document_content = row["content"].strip()
                if document_content:
                    add_or_update_document(collection, document_content, embedder, row["url"])
        except Exception as e:
            print(f"❌ Erreur lors du chargement du CSV : {e}")
    else:
        print("⚠️ Aucun fichier CSV spécifié. Aucun document ajouté.")
    
    topic = input("📝 Entrez un sujet de recherche : ")
    result = search_documents(topic, args.collection_name, args.search_top_k, embedder, client)
    
    print(f"\n🔍 Documents trouvés pour '{topic}':")
    for idx, doc in enumerate(result):
        print(f"\n📄 Document {idx+1}:\n{doc['document']}")
        metadata = doc['metadata'] if isinstance(doc['metadata'], dict) else {'source': 'inconnue'}
        print(f"🔗 Source: {metadata.get('source', 'inconnue')}")

if __name__ == "__main__":
    main(parse_arguments())

    args = parse_arguments()
    main(args)




