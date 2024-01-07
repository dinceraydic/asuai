import os

import openai
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
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

loader = TextLoader("./data/yeni_sss.txt", encoding="UTF-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

docs = text_splitter.split_documents(documents)

# create vector db
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory=persist_vector_directory,
)
vectordb.persist()

# retrievalQQ to get info
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(
        search_kwargs={"k": 3},
    ),
    return_source_documents=True,
)

result = qa_chain("hangi programlar ücretsiz")

print(result["result"])

# # setup qa chain
# chain = load_qa_chain(llm, verbose=True)
# query = "hangi programlar ücretsiz?"
