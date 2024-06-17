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
system_prompt = """#역할 및 목표: 전문적인 보고서(Professional Report) 작성을 돕는 서비스입니다.  사용자가 보고서 작성 관련 질문이나 조언을 요청할 때, 이 GPT는 보고서 작성 요건, 목차 구성, 목차 별 스토리 보드 구성, 문서 작성, 작성 내용 평가 및 수정 단계 별 조언과 사례를 제공합니다. 전문적인 보고서 작성 시 LLM의 활용을 최적화하기 위한 구체적인 방안을 제안합니다.

#타겟 오디언스: 전문적인 보고서 작성자.

#응답 지식 가이드라인:
##'Knoweldge'에 저장된 파일을 최우선순위로 참조하되 ChatGPT, 외부 검색 지식을 조합하여 답변을 제공합니다.
##'knowledge'에 저장된 답변을 최대한 가공하지 않고 응답 스타일만을 조정하여 제공합니다.
##부당한 가정을 하지 않고 정확한 정보 제공을 우선시합니다.

#제약사항:
##이 서비스는 한국어로 답변을 제공합니다.

#설명: GPT는 사용자의 질문이 불분명하거나 불완전한 경우, 사용자에게 명확히 설명해 줄 것을 질문을 통해 요청할 수 있습니다.

#개인화: GPT는 현장의 실제 경험을 반영하는 언어와 예시를 사용하여 노련한 컨설턴트의 스타일을 모방해야 합니다.

#응답 스타일 및 톤
##응답은 분석적이며 전문적인 어투를 사용한다. 또한 글의 일관성을 유지하고, 리듬과 템포를 조절하여 글의 흐름을 개선한다.
##공식적 언어 사용: 공식적인 문자 뒷부분은 존칭은 빼고 '다'로 마무리하며, 표준어를 사용한다.
##최대한 상세한 설명을 제공한다. 'Knowledge'에 표 형태로 저장된 지식은 표형태로 생략 없이 제공한다."""

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

chat = ChatOpenAI(model="gpt-4o")

prompt = PromptTemplate(template="""문장을 기반으로 질문에 답하세요. 

문장: 
{document}

질문: {query}
""", input_variables=["document", "query"])

text_splitter = SpacyTextSplitter(chunk_size=700, pipeline="ko_core_news_sm")

@cl.on_chat_start
async def on_chat_start():
    files = None #← 파일이 선택되어 있는지 확인하는 변수

    while files is None: #← 파일이 선택될 때까지 반복
        files = await cl.AskFileMessage(
            max_size_mb=100,
            max_files=4,
            content="PDF를 선택해 주세요",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()
    # file = files[0]
    # print(files)

    if not os.path.exists("tmp"): #← tmp 디렉터리가 존재하는지 확인
        os.mkdir("tmp") #← 존재하지 않으면 생성
    for file in files:
        with open(f"tmp/{file.name}", "wb") as f: #← PDF 파일을 저장
            f.write(file.content) #← 파일 내용을 작성

    database = Chroma( #← 데이터베이스 초기화
        embedding_function=embeddings,
        # 이번에는 persist_directory를 지정하지 않음으로써 데이터베이스 영속화를 하지 않음
    )

    # documents = []
    for file in files:
        documents = PyMuPDFLoader(f"tmp/{file.name}").load() #← 저장한 PDF 파일을 로드
        splitted_documents = text_splitter.split_documents(documents) #← 문서를 분할
        # documents.extend(document)
        database.add_documents(splitted_documents) #← 문서를 데이터베이스에 추가

    cl.user_session.set(  #← 데이터베이스를 세션에 저장
        "database",  #← 세션에 저장할 이름
        database  #← 세션에 저장할 값
    )

    await cl.Message(content=f"`{file.name}` 로딩이 완료되었습니다. 질문을 입력하세요.").send() #← 불러오기 완료를 알림

@cl.on_message
async def on_message(input_message):
    print("입력된 메시지: " + input_message)

    database = cl.user_session.get("database") #← 세션에서 데이터베이스를 가져옴

    documents = database.similarity_search(input_message)

    documents_string = ""

    for document in documents:
        documents_string += f"""
    ---------------------------
    {document.page_content}
    """
    
    result = chat([
        HumanMessage(content=prompt.format(document=documents_string,
                                           query=system_prompt + '\n\n' + input_message)) #← input_message로 변경
    ])
    await cl.Message(content=result.content).send()