import streamlit as st
import litellm
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
import pandas as pd

# Initialisation de l'interface Streamlit
st.title("🦙 Ollama Chat avec LiteLLM & Qdrant")

# Configuration de Qdrant et de l'embedder
collection_name = "my_documents"
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# Vérification ou création de la collection
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    st.success(f"✅ Collection '{collection_name}' créée avec succès.")
else:
    st.warning(f"⚠️ La collection '{collection_name}' existe déjà. Utilisation de la collection existante.")

# Fonction pour charger les documents depuis des fichiers CSV
def load_documents(csv_paths):
    documents = []
    doc_id = 1  # ID unique
    
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path).fillna("")
            for _, row in df.iterrows():
                document_text = row.get("content", "")
                source = row.get("url", "Unknown")
                
                embedding = embedder.embed_query(document_text)
                if len(embedding) != 384:
                    st.error(f"❌ Dimension d'embedding incorrecte pour {doc_id}, ignoré.")
                    continue
                
                documents.append(PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={"content": document_text, "source": source}
                ))
                doc_id += 1
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement : {str(e)}")
    return documents

csv_paths = [
    "C:/Users/Dev1/Documents/Storm/storm/examples/storm_examples/documents.csv",
    "C:/Users/Dev1/Documents/Storm/storm/examples/storm_examples/fruits.csv",
    "C:/Users/Dev1/Documents/Storm/storm/examples/storm_examples/sport.csv"
]

documents = load_documents(csv_paths)
if documents:
    try:
        client.upsert(collection_name=collection_name, points=documents)
        st.success(f"✅ {len(documents)} documents insérés avec succès.")
    except Exception as e:
        st.error(f"❌ Erreur lors de l'insertion des documents : {str(e)}")

# Fonction de recherche
def search_documents(query, collection_name, search_top_k=1, similarity_threshold=1):
    query_embedding = embedder.embed_query(query)
    search_results = client.search(collection_name=collection_name, query_vector=query_embedding, limit=search_top_k)
    
    return [
        {
            "content": r.payload["content"],
            "source": r.payload.get("source", "Unknown"),
            "distance": r.score
        }
        for r in search_results if r.score >= similarity_threshold
    ]

# Interaction utilisateur
user_input = st.text_input("🗨️ You: ", "", key="user_input")

if st.button("Send"):
    if user_input:
        search_results = search_documents(user_input, collection_name, similarity_threshold=0.5)
        
        if search_results:
            context = "\n\n".join([doc["content"] for doc in search_results])
            prompt = f"Réponds à la question suivante en utilisant uniquement ces informations:\n\n{context}\n\nQuestion: {user_input}"
        else:
            prompt = user_input
        
        response = litellm.completion(
            model="ollama/llama2",
            messages=[{"role": "user", "content": prompt}]
        )
        ollama_answer = response['choices'][0]['message']['content']
        
        st.write(f"🤖 **Assistant:** {ollama_answer}")
        
        if search_results:
            st.subheader("📄 Documents utilisés :")
            for doc in search_results:
                st.write(f"---\n📖 **Extrait :** {doc['content']}\n🔗 **Source :** {doc['source']}")
        else:
            st.write("❌ Aucun document pertinent trouvé.")