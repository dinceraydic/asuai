docs = """When and under what conditions can I apply to your graduate programs? Graduate student admissions are made in the fall and spring semesters specified in the academic calendar. Minimum application requirements: 
To have the undergraduate degree required in the program application requirements To have at least 55 points from ALES in the field required in the program application conditions. For the PhD Program, getting at least 55 points from the Foreign Language Exam
Can I apply to more than one graduate program? You can apply to 1 Master's program without thesis and 1 Master's program with thesis or 1 Master's program without thesis and 1 PhD program. 
Will I pay a fee for graduate programs? Master's and PhD programs with thesis are free of charge, but if you cannot graduate within the normal period, you will pay a fee until your maximum period expires.
Non-thesis Master's programs are paid and the total fee of our institute's programs is 5500.00 TL. It is collected in two equal installments at the beginning of the semester. Distance Education Non-Thesis Master's Degree programs are paid and the total fee is 6500,00 TL. It is collected in two equal installments at the beginning of the semester.
In non-thesis master's programs, free of charge for the relatives of martyrs and veterans, and for the disabled; discounts are made according to disability rates. The fee for the employees of the institution with a protocol in non-thesis master's programs is 3400TL.
The total fee for Aksaray University staff is 2000TL. How long is the military service deferment (extension) period for graduate programs? Does the institute do this process?
It is 1.5 years for non-thesis Master's programs, 3 years for Master's programs with thesis and 6 years for PhD programs. Our institute makes the deferment procedures within 1 month from the date of registration."""

with open("./AsuAI/FAQ-DeepL.txt", mode="r", encoding="utf-8") as file:
    docs2 = file.read()

docs2_1 = docs2[:4000]

# api key
import os

from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")

import langchain
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

import deepl
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

auth_key = os.environ.get("DEEPL_API_KEY")
translator = deepl.Translator(auth_key)

## Connect OpenAI Model
model = ChatOpenAI(openai_api_key=api_key)


def answer_my_question(question, docs):
    question = translator.translate_text(question, target_lang="EN-US")
    print(question)

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

st.title("Aksaray Üniversitesi AI Danışman")
question_text = st.text_area("Size nasıl yardımcı olabilirim?")

if st.button("Cevapla", type="primary"):
    answer_text = answer_my_question(question=question_text, docs=docs2_1)
    st.markdown(answer_text)
