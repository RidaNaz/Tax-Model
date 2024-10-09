import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# Display the Logo and Title on the Same Line
col1, col2 = st.columns([0.2, 0.8])  # Adjust the column width as needed
with col1:
    st.image("Ridalogo.png", width=70)  # Replace "logo.png" with the path to your logo file

# Display the title with custom padding using HTML and CSS
with col2:
    st.markdown(
        """
        <style>
        .custom-title {
            padding-bottom: 25px;  /* Adjust the padding as needed */
        }
        </style>
        <h1 class="custom-title">TAX Query Model (2024-25)</h1>
        """, unsafe_allow_html=True
    )

st.markdown("<h5 style='text-align: center;'>by Rida Naz</h5>", unsafe_allow_html=True)

# Add a gap after the title using st.markdown with CSS styling
st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)  # You can adjust the `margin-bottom` value as needed

llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-11b-text-preview")

prompt=ChatPromptTemplate.from_template(
"""
You are knowledgeable and professional specializing in the Pakistani tax system. Answer the questions or tax rates based on the provided tax context and tables only. Please provide the most accurate and specific response based on the user question, referring directly to the information within the tax PDF.
<context>
{context}
<context>
Questions:{input}
"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./pdfs") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings



prompt1=st.text_input("Looking for specific tax rates? Let me know your query!")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
