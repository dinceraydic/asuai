import os

import streamlit as st
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")

# Configure langchain
# SQLiteCache(database_path=".langchain.db")
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200)
embedding_function = OpenAIEmbeddings()

# Load documents
loader = TextLoader("./data/metin1.txt", encoding="UTF-8")
documents = loader.load()

# Split and persist documents
docs = text_splitter.split_documents(documents)
db = Chroma.from_documents(
    docs, embedding_function, persist_directory="./vector-store/asu_ai_db_tr200_3"
)
db.persist()

db = Chroma.from_documents(
    docs, embedding_function, persist_directory="./some_new_mkultra"
)
db.persist()

# embedding function
embedding_function = OpenAIEmbeddings()

# Connect to OpenAI Model
model = ChatOpenAI(openai_api_key=api_key)


def answer_my_question(question, docs):
    template = "Bu soruyu cevapla:\n{question}\n Cevaplarını aşağıdaki bağlama göre ver:\n{document}\n"
    human_prompt = HumanMessagePromptTemplate.from_template(template)
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    result = model(
        chat_prompt.format_prompt(question=question, document=docs).to_messages()
    )
    return result.content


# Streamlit UI
st.title("Aksaray Üniversitesi AI Danışman")
st.subheader("Türkçe'den Türkçe'ye", divider="rainbow")
question_text_tr = st.text_area("Size nasıl yardımcı olabilirim?")

if st.button("Cevapla", type="primary"):
    doc_txt = db.similarity_search(str(question_text_tr))
    baglam = "".join(dc.page_content for dc in doc_txt[:3])
    answer_text = answer_my_question(question=question_text_tr, docs=str(baglam))
    st.markdown(str(answer_text))

    # debug
    st.write("-------------------------------------------------------------")
    st.markdown("""### Aşağısı cevapların geldiği kısmı kontrol amaçlı eklenmiştir.""")
    st.write(question_text_tr)
    st.write(baglam)
