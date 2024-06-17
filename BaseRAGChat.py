import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-"
openai.api_key = "sk-"

import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma

# 보고서 생성 토픽을 위한 System prompt
system_prompt = """당신은 사용자의 질문에 대한 응답을 생성하는 어시스턴트 인공지능 입니다. 사용자의 질문에 대한 답을 친절하고 상세하게 알려주세요."""

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

chat = ChatOpenAI(model="gpt-4o")

prompt = PromptTemplate(template=""" 

문장: 
{document}

질문: {query}
""", input_variables=["document", "query"])

text_splitter = SpacyTextSplitter(chunk_size=700, pipeline="ko_core_news_sm")

@cl.on_chat_start
async def on_chat_start():
    files = None 

    while files is None: 
        files = await cl.AskFileMessage(
            max_size_mb=100,
            max_files=4,
            content="PDF를 선택해 주세요",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()
    # file = files[0]
    # print(files)

    if not os.path.exists("tmp"): 
        os.mkdir("tmp") 
    for file in files:
        with open(f"tmp/{file.name}", "wb") as f: 
            f.write(file.content) 

    database = Chroma( #← 데이터베이스 초기화
        embedding_function=embeddings)

    # documents = []
    for file in files:
        documents = PyMuPDFLoader(f"tmp/{file.name}").load()
        splitted_documents = text_splitter.split_documents(documents)
        # documents.extend(document)
        database.add_documents(splitted_documents)

    cl.user_session.set(  
        "database",  
        database  
    )

    await cl.Message(content=f"`{file.name}` 로딩이 완료되었습니다. 질문을 입력하세요.").send() 

@cl.on_message
async def on_message(input_message):
    print("입력된 메시지: " + input_message)

    database = cl.user_session.get("database")

    documents = database.similarity_search(input_message)

    documents_string = ""

    for document in documents:
        documents_string += f"""
    ---------------------------
    {document.page_content}
    """
    
    result = chat([
        HumanMessage(content=prompt.format(document=documents_string,
                                           query=system_prompt + '\n\n' + input_message))
    ])
    await cl.Message(content=result.content).send()
