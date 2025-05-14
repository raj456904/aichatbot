import streamlit as st
from qa_chain import load_qa_chain

st.title("📄 AI-Powered Q&A Chatbot")
st.write("Ask anything from the uploaded document")

qa_chain = load_qa_chain()

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})
        st.subheader("Answer")
        st.write(result["result"])
        
        with st.expander("Source Document Chunks"):
            for doc in result["source_documents"]:
                st.write(doc.page_content[:500])
