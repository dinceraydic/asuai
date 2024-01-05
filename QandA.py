import os

import openai
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")
persist_vector_directory = "./vector-store/son_vector_store"

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

loader = TextLoader("./data/yeni_sss.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chank_overlap=200,
)

docs = text_splitter.split_documents(documents)

# create vector db
vectordb = Chroma.from_documents(
    documents=docs,
    embeddings=OpenAIEmbeddings(),
    persist_directory=persist_vector_directory,
)
vectordb.persist()

# # setup qa chain
# chain = load_qa_chain(llm, verbose=True)
# query = "hangi programlar ücretsiz?"
