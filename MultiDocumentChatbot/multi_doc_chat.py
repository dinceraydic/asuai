import os

import langchain
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma
from streamlit_chat import message

from load_docs import load_docs

load_dotenv("../.env")
api_key = os.environ.get("OPENAI_API_KEY")
persist_vector_directory = "./vector-store/son_vector_store_v4"

## Caching
langchain.llm = SQLiteCache("./cache/langchain.db")

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# load files
documents = load_docs()

# create a chat history
chat_history = []

# text_splitter = CharacterTextSplitter(
#     chunk_size=1200,
#     chunk_overlap=10,
# )
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=10,
)

docs = text_splitter.split_documents(documents)

if os.path.exists(persist_vector_directory):
    # create vector db
    vectordb = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_vector_directory,
    )
    vectordb.persist()
else:
    # create vector db
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_vector_directory,
    )
    vectordb.persist()

# ConversationalRetrievalChain to get info
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True,
    verbose=False,
)

# === Streamlit front-end ===
st.title("ASÜ AI")
st.header("Merak ettiklerinizi sorabilirsiniz...🤖")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_query():
    input_text = st.chat_input("Sorunuzu sorun...")
    return input_text


# retrieve the user input
user_input = get_query()
if user_input:
    result = qa_chain(
        {
            "question": user_input,
            "chat_history": chat_history,
        }
    )
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result["answer"])

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
