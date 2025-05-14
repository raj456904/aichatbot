from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def load_qa_chain():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
