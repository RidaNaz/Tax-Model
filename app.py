import os
import time
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

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Rida Naz - TAX Query Model",
    page_icon="favicon.ico",
    layout="centered",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .custom-title {
        padding-bottom: 10px;
        font-size: 2.5em;
        font-weight: bold;
        color: #2E86C1;
    }
    .custom-subtitle {
        font-size: 1.2em;
        color: #5D6D7E;
        text-align: center;
    }
    .stButton>button {
        background-color: #28a745; /* Green color for the button */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1em;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #218838; /* Darker green on hover */
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the Logo and Title on the Same Line
col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.image("favicon.ico", width=50)

with col2:
    st.markdown("<h1 class='custom-title'>TAX Query Model (2024-25)</h1>", unsafe_allow_html=True)

st.markdown("<h5 class='custom-subtitle'>by Rida Naz</h5>", unsafe_allow_html=True)

# Add a gap after the title
st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are knowledgeable and professional specializing in the Pakistani tax system. Answer the questions or tax rates based on the provided tax context and tables only. Please provide the most accurate and specific response based on the user question, referring directly to the information within the tax PDF.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector embeddings..."):
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFDirectoryLoader("./pdfs")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector embeddings created successfully!")
        
# Automatically generate embeddings when the app starts
vector_embedding()

# User input for tax query
prompt1 = st.text_input("Looking for specific tax rates? Let me know your query!", placeholder="Enter your tax query here...")

# Process the query
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please generate document embeddings first.")
    else:
        with st.spinner("Processing your query..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write(response['answer'])

            # Display relevant document chunks
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
                    
                    
# Reset button to clear session state
if st.button("Reset App"):
    st.session_state.clear()
    st.rerun()