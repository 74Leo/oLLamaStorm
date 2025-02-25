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
from argparse import ArgumentParser
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

def initialize_qdrant(collection_name, vector_size):
    client = QdrantClient(path="./qdrant_db")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    return client

def add_document(client, collection_name, document_content, embedder, source):
    """Ajoute chaque document individuellement dans Qdrant avec son embedding propre."""
    try:
        document_id = int(hashlib.md5(document_content.encode()).hexdigest(), 16) % (10 ** 9)
        vector = embedder.embed_documents([document_content])[0]
        point = PointStruct(id=document_id, vector=vector, payload={"source": source, "content": document_content})
        client.upsert(collection_name=collection_name, points=[point])
        print(f"✅ Document ajouté (ID: {document_id}) depuis {source}")
    except Exception as e:
        print(f"❌ Erreur lors de l'ajout du document : {e}")

def load_documents_from_csv(directory, client, collection_name, embedder):
    """Charge les fichiers CSV et stocke chaque document séparément."""
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            try:
                # Mettez à jour ici pour utiliser le séparateur '|'
                df = pd.read_csv(file_path, dtype=str, header=0, sep='|').fillna("unknown")
                print(f"Noms des colonnes dans {filename} : {df.columns.tolist()}")
                for _, row in df.iterrows():
                    content = str(row["content"]).strip()
                    title = str(row["title"]).strip()
                    url = str(row["url"]).strip()
                    description = str(row["description"]).strip()

                    combined_content = f"{title}\n{description}\n{content}"
                    add_document(client, collection_name, combined_content, embedder, url)
            except Exception as e:
                print(f"❌ Erreur lors de la lecture du fichier {file_path} : {e}")




def search_best_document(query, collection_name, embedder, client, similarity_threshold=0.3):
    """Cherche le document le plus pertinent par rapport à la requête."""
    query_embedding = embedder.embed_documents([query])[0]
    search_results = client.search(collection_name=collection_name, query_vector=query_embedding, limit=3)

    for result in search_results:
        if result.score >= similarity_threshold:
            return result.payload["content"], result.payload["source"]

    return None, None

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--csv-dir", type=str, required=True, help="Dossier contenant les fichiers CSV.")
    parser.add_argument("--collection-name", type=str, default="qdrant_documents", help="Nom de la collection Qdrant.")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Modèle d'embedder.")
    return parser.parse_args()

def main(args):
    llm = OllamaLLM(model="llama2")
    embedder = HuggingFaceEmbeddings(model_name=args.embedding_model)

    try:
        example_text = "Ceci est un exemple."
        example_embedding = embedder.embed_documents([example_text])[0]
        vector_size = len(example_embedding)
        print("Dimension des embeddings :", vector_size)
    except Exception as e:
        print(f"Erreur lors de l'obtention de la dimension des embeddings : {e}")
        return

    client = initialize_qdrant(args.collection_name, vector_size)

    load_documents_from_csv(args.csv_dir, client, args.collection_name, embedder)

    user_query = input("Entrez votre requête : ")
    relevant_text, source = search_best_document(user_query, args.collection_name, embedder, client)

    if relevant_text:
        print(f"📄 **Document pertinent trouvé dans {source} :**")
        print(relevant_text)
    else:
        print("❌ Aucun document pertinent trouvé.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
