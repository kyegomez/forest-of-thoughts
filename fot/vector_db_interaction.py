import chromadb
import uuid

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="agent-thoughts")

def add_document(document: str):
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        documents=[document]
    )
    
    return doc_id

def query_documents(query: str, n_docs: int = 1):
    docs = collection.query(
        query_texts=[query],
        n_results=n_docs
    )["documents"]
    
    return docs[0]