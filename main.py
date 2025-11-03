import os
from dotenv import load_dotenv
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import google.generativeai as genai

# === Load environment variables ===
load_dotenv()

# === 1. Set up Chroma Cloud client ===
client = chromadb.CloudClient(
    api_key=os.getenv('CHROMA_API_KEY'),
    tenant=os.getenv('CHROMA_TENANT_ID'),
    database=os.getenv('CHROMA_DATABASE')
)

# === 2. Read your data.txt ===
file_path = 'data.txt'
with open(file_path, encoding='utf-8') as f:
    text = f.read().strip()

# Split into paragraphs or lines
documents = [d for d in text.split('\n\n') if d.strip()]
ids = [f'doc_{i}' for i in range(len(documents))]

# === 3. Configure Gemini API ===
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# === 4. Define Gemini embedding function ===
class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str = "models/text-embedding-004"):
        self.model_name = model_name

    def __call__(self, inputs: Documents) -> Embeddings:
        embeddings = []
        for text in inputs:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings

# === 5. Initialize embedding function ===
embed_fn = GeminiEmbeddingFunction()

# === 6. Create or get a Chroma collection ===
collection_name = "my_data_collection"

# Create fresh collection
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embed_fn
)

# === 7. Add documents to collection ===
collection.add(
    ids=ids,
    documents=documents,
    metadatas=[{"source": "data.txt"} for _ in documents]
)

print(f"✅ Added {len(documents)} documents to collection '{collection_name}'")

# === 8. Query example ===
query_text = "When Pratik Kohli close his machine?"

query_embedding = genai.embed_content(
    model="models/text-embedding-004",
    content=query_text,
    task_type="retrieval_query"
)

results = collection.query(
    query_embeddings=[query_embedding["embedding"]],
    n_results=3
)

print("\n🔍 Query results:")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"\n{i+1}. Distance: {distance:.4f}")
    print(f"   {doc[:200]}...")
