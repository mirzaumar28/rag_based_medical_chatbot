from flask import Flask, render_template, jsonify, request
from src.helper import download_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGEFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"] = HUGGINGEFACEHUB_ACCESS_TOKEN


embeddings = download_embedding_model()

index_name = "medicalchatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

retriver = docsearch.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)

llm = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-V3.2-Exp',
    task = 'text-generation'
)
model = ChatHuggingFace(llm = llm)

prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriver, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input" : msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080, debug = True)