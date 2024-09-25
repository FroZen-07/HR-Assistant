from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFaceHub

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name, "id": filename.file_id, "type": filename.type, "size": filename.size, "unique_id": unique_id},
        ))
    return docs

def create_embeddings_load_data():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def create_vector_store(embeddings, docs):
    vector_store = Chroma.from_documents(docs, embeddings)
    return vector_store

def similar_docs(query, k, vector_store, unique_id):
    relevant_docs = vector_store.similarity_search_with_relevance_scores(
        query, 
        k=k, 
        filter={"unique_id": unique_id}
    )
    return relevant_docs  # This now returns a list of (document, score) tuples

def get_summary(current_doc):
    llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature": 1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])
    return summary