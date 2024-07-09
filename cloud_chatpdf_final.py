__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader

API_KEY = os.getenv('OPENAI_API_KEY')

openai_key = st.text_input('OPEN_AI_API_KEY',type="password")


#제목
st.title("ChatPDF")
st.write("---")


#파일업로드
uploaded_file = st.file_uploader("PDF파일 올려주세요.",type=["pdf"])
st.write("---")

# 1. Loader
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages
        


#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)


    # 2. Split
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)
    #print(texts[0])

    #3. Embedding
    from langchain_openai import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings(api_key=openai_key)

    #4. load it into Chroma
    from langchain_chroma import Chroma
    vectordb = Chroma.from_documents(texts, embeddings_model)
    
    #https://velog.io/@udonehn/RAG%EB%A5%BC-%EC%A0%81%EC%9A%A9%ED%95%9C-%EC%A7%88%EC%9D%98%EC%9D%91%EB%8B%B5-%EC%B1%97%EB%B4%87-LangChanin
    #Question I
    # from langchain_openai import ChatOpenAI
    # from langchain.retrievers.multi_query import MultiQueryRetriever
    # question = "아내가 먹고 싶어하는 음식은 무엇이야?"
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
    # #retriever_from_llm = MultiQueryRetriever.from_llm(  retriever=vectordb.as_retriever(), llm=llm)

    # # retriever_from_llm.get_relevant_documents(query=question)


    #Question II
    st.header("PDF 에게 질문해보세요.")
    question = st.text_input("질문을 입력하세요")
    if st.button("질문하기"):
        with st.spinner("Wait for it..."):
            from langchain_openai import ChatOpenAI
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough
            from langchain import hub

            llm = ChatOpenAI(api_key=openai_key,model_name="gpt-4o", temperature=0)


            # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
            from langchain.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        You are a helpful assistant. 
                        Answer questions using only the following context. 
                        If you don't know the answer just say you don't know, don't make it up:
                        \n\n
                        {context}",
                        """
                    ),
                    ("human", "{question}"),
                ]
            )

            qa_chain = (
                {
                    "context": vectordb.as_retriever(),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            result = qa_chain.invoke(question)
            print(result);
            st.write(result)