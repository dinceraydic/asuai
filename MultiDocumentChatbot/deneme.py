# import os

# from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import TextLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import (
#     CharacterTextSplitter,
#     RecursiveCharacterTextSplitter,
# )
# from langchain.vectorstores import Chroma

# load_dotenv("../.env")
# api_key = os.environ.get("OPENAI_API_KEY")
# persist_vector_directory = "../vector-store/son_vector_store"
# # document_file_path = "../../LangChainTutorial_1/LangChainNotebooks-UNZIP-ME/00-Models-IO/01-Data-Connections/some_data/FDR_State_of_Union_1944.txt"
# # document_file_path = "./original_docs/SSS.txt"
# document_file_path = "./docs/yonetmelik.txt"


# llm_model = "gpt-3.5-turbo"
# llm = ChatOpenAI(temperature=0.0, model=llm_model)

# ## Load document
# loader = TextLoader(document_file_path, encoding="utf-8")
# documents = loader.load()

# ## Split text
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=500, chunk_overlap=10
# )
# docs = text_splitter.split_documents(documents)

# print(docs[0].page_content)
# print(len(docs))
