from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_req_docs, text_split, download_embedding_model
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HUGGINGFACEHUB_ACCESS_TOKEN = os.environ.get('HUGGINGFACEHUB_ACCESS_TOKEN')

os.environ[PINECONE_API_KEY] = PINECONE_API_KEY
os.environ[HUGGINGFACEHUB_ACCESS_TOKEN] = HUGGINGFACEHUB_ACCESS_TOKEN

extracted_data = load_pdf_files(data = 'data/')

filter_data = filter_to_req_docs(extracted_data)

text_chunk = text_split(filter_data)

embedding = download_embedding_model()

pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key = PINECONE_API_KEY)    # authenticating the pinecone apikey


index_name = "medicalchatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = 'cosine',
        spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunk,
    index_name = index_name, 
    embedding = embedding
)

