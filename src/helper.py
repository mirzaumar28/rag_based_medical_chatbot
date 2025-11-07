from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )
    docs = loader.load()
    return docs



def filter_to_req_docs(docs: List[Document]) -> List[Document]:
    req_docs : List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        req_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source" : src}
            )
        )
    return req_docs
    

def text_split(req_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
        length_function = len
    )
    text = text_splitter.split_documents(req_docs)
    return text

def download_embedding_model():
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embeddings