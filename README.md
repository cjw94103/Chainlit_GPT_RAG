# 1. RAG란?
<p align="center"><img src="https://github.com/cjw94103/Chainlit_GPT_RAG/assets/45551860/383e121d-6e05-4fcc-93e2-017e66c66f72" width="35%" height="35%"></p>

RAG (Retrieval-Augmented Generation)은 LLM (Large Language Model)의 출력을 최적화하여 응답을 생성하기 전에 학습 데이터 소스 외부의 신뢰할 수 있는 지식 베이스를 참조하도록 하는 프로세스 입니다. LLM은 방대한 양의 데이터를 기반으로 수십~수백억개의 파라미터를 사용하여 질문에 대한 답변, 언어 번역, 문장 완성과 같은 작업에 대한 휼륭한 결과를 제공하지만 실시간으로 생성되는 학습 데이터의 최신성을 유지할 수 없는 문제가 있으며 또 다른 알려진 문제점은 아래와 같습니다.

- 답변이 없을 때 허위 정보를 제공합니다. (할루시네이션)
- 사용자가 구체적이고 최신의 응답을 기대하지만 대체로 일반적인 정보를 제공하는 경향이 있습니다.
- 신뢰할 수 없는 출처로부터 응답을 생성하는 경우가 있습니다.
- 다양한 분야의 용어 혼동으로 인해 응답이 정확하지 않을때가 있습니다.
  
RAG는 기존 LLM의 생성 능력과 외부 지식 베이스의 정보를 결합하여 보다 정확하고 사실에 기반한 답변을 제공할 수 있습니다. 또한 모델의 출력 결과에 대한 증거를 외부 지식 베이스로부터 제시할 수 있어 설명 가능성과 신뢰성을 향상시킬 수 있습니다. 본 구현에서는 GPT, Langchain을 이용하여 간단한 RAG 시스템을 Chainlit을 이용한 웹 기반의 챗봇으로 구현합니다. 구현은 간단하지만 시스템의 출력은 현업에서 사용할 수 있을만큼 강력합니다.

# 2. 구현 설명
RAG 챗봇은 GPT, Langchain을 이용하여 Chainlit을 이용한 웹 기반으로 구현합니다. 간략한 동작 과정은 아래와 같습니다.   

1. PDF를 최대 4개까지 업로드 합니다. 이때 각 PDF의 페이지수는 70~85 페이지로 맞춰주셔야 제대로 동작합니다.
2. PyMuPDFLoader를 이용하여 PDF를 읽어오고 Spacy의 "ko_core_news_sm" 모델로 텍스트를 문장 단위로 쪼갭니다.
3. openai의 "text-embedding-ada-002"를 이용하여 텍스트를 벡터로 변환한 후 Chroma 벡터데이터베이스에 저장합니다.
4. 사용자가 채팅창에 질문을 입력하면 질문과 가장 유사한 PDF 텍스트를 Chroma DB의 similarity_search를 메서드를 통해 가져온 후 참조 텍스트를 생성합니다.
5. 참조 텍스트 + 사용자의 질문을 PromptTemplate에 맞춰 정렬한 후 openai의 GPT4-o에게 전달합니다.
6. GPT4-o는 질문에 대한 응답을 전송합니다.

벡터데이터베이스를 위한 embedding 모델 및 GPT 모델은 사용자의 니즈에 따라 코드 레벨에서 변경하여 사용하시면 됩니다.

# 3. 코드 실행
먼저 requirements.txt에 명시된 라이브러리를 아래와 같은 명령어를 통하여 설치합니다.

```python
pip install requirements.txt
```

설치 완료 후 웹 기반의 챗봇을 실행하기 위하여 아래와 같은 Chainlit 명령어를 입력해주세요.
```python
chainlit run BaseRAGChat.py --port [port_num] --host [your_ip_adress]
```
# 4. 실행 이미지
<p align="center"><img src="https://github.com/cjw94103/Chainlit_GPT_RAG/assets/45551860/495a9e18-d855-4ed9-a82c-efcf8f377158" width="100%" height="100%"></p>

PDF를 업로드 하고 PDF에 대한 질문에 대한 답을 챗봇 형태로 실행해보실 수 있습니다.
