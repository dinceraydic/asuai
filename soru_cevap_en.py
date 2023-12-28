# api key
import os

from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")

import langchain
from langchain.cache import SQLiteCache
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

import deepl
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

## Load document
loader = TextLoader("./data/FAQ-DeepL.txt")
documents = loader.load()

## Split text
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=250)
docs = text_splitter.split_documents(documents)

embedding_function = OpenAIEmbeddings()

db = Chroma.from_documents(
    docs, embedding_function, persist_directory="./vector-store/asu_ai_en_db"
)
db.persist()

auth_key = os.environ.get("DEEPL_API_KEY")
translator = deepl.Translator(auth_key)

## Connect OpenAI Model
model = ChatOpenAI(openai_api_key=api_key)


def answer_my_question(question, docs):
    question = translator.translate_text(question, target_lang="EN-US")

    ## Prompt - Format Question
    template = "Answer this question:\n{question}\n Here is some extra context:\n{document}\n If the answer isn't in the context return 'I cannot find the answer.'."
    human_prompt = HumanMessagePromptTemplate.from_template(template)

    ## Chat Prompt - Get Result Content
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

    ## Get Result
    result = model(
        chat_prompt.format_prompt(question=question, document=docs).to_messages()
    )

    return translator.translate_text(result.content, target_lang="TR")


import streamlit as st

st.title("ASÜ-AI")
st.subheader("TR ↔️ EN, EN ↔️ TR", divider="rainbow")
question_text_tr = st.text_area("Size nasıl yardımcı olabilirim?")

if st.button("Cevapla", type="primary"):
    question_text_en = translator.translate_text(
        str(question_text_tr), target_lang="EN-US"
    )
    doc_txt = db.similarity_search(str(question_text_en))
    baglam = ""
    for dc in doc_txt[:3]:
        baglam += dc.page_content
    answer_text = answer_my_question(question=question_text_tr, docs=str(baglam))
    st.markdown(str(answer_text))

    # debugging
    st.write("-------------------------------------------------------------")
    st.markdown("""### Aşağısı cevapların geldiği kısmı kontrol amaçlı eklenmiştir.""")
    st.write(question_text_en)
    st.write(baglam)
