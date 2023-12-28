from langchain.document_loaders import TextLoader, WebBaseLoader

# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
loader = TextLoader("./AsuAI/FAQ-DeepL.txt")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

model_path = r"D:\_programlama\ML\RAG-1\models\chat\llama-2-7b-chat.Q3_K_M.gguf"

# # Test similarity search is working with our local embeddings.
# question = "What are the approaches to Task Decomposition?"
# docs = vectorstore.similarity_search(question)
# len(docs)


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_path,  # "/Users/rlm/Desktop/Code/llama.cpp/models/llama-2-13b-chat.ggufv3.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

# llm("tell me about ALES")

import os

import deepl
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

auth_key = os.environ.get("DEEPL_API_KEY")
translator = deepl.Translator(auth_key)


def answer_my_question(question):
    # question = translator.translate_text(question, target_lang="EN-US")
    docs = vectorstore.similarity_search(str(question))
    print("Similarity search: ", question, "\n")

    context = ""
    for dc in docs[:2]:
        context += dc.page_content

    ## Prompt - Format Question
    template = f"Answer this question:\n{question}\n Here is some extra context:\n{context}\n If the answer isn't in the context return 'I cannot find the answer.'."
    human_prompt = HumanMessagePromptTemplate.from_template(template)

    ## Chat Prompt - Get Result Content
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

    ## Get Result
    # print("template: ", template)
    result = llm(
        f"What should I do if I want to take the thesis defense exam? according to the following text\n\n{context}",  # template  # chat_prompt.format_prompt(question=question, document=docs).to_messages()
    )
    print("result: ", result)

    # return translator.translate_text(result, target_lang="TR")
    return result


question = "what is ALES"
answer = answer_my_question(question)
print(answer)
