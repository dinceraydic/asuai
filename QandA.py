import os

from dotenv import load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

loader = TextLoader("./data/yeni_sss.txt")
documents = loader.load()


# # setup qa chain
# chain = load_qa_chain(llm, verbose=True)
# query = "hangi programlar ücretsiz?"
