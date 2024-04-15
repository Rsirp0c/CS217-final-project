# QA Learning Chat Bot
import streamlit as st
import os
import re
import numpy as np
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import PyPDFLoader
import unicodedata
import cohere
import tempfile
from langchain_community.vectorstores import Pinecone
from streamlit_pinecone import PineconeConnection
from pinecone import Pinecone as PineconeClient

# import the storing of static information in a separate file
from src.constant import *
from src.sidebar import *

def get_open_ai_chat_response(query):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vectorstore = Pinecone.from_existing_index(
        index_name=os.getenv('PINECONE_INDEX_NAME'), embedding=embeddings)
    
    retriever = vectorstore.as_retriever()

    # RAG prompt
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # RAG
    model = Ollama(
                    model="llava",  
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler])
                )

    chain = (
        RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke(query)

    return response

# Reading pdfs
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    return documents

# Process pdfs
def process_documents(documents):
    doc_text = ''
    for doc in documents:
        text = doc.page_content
        
        # preprocess
        text = clean_text(text)
        doc_text += text
    return doc_text

# Preprocess the text
def clean_text(text):
    # Replace newline characters with spaces
    text = text.replace('\n', ' ')
    # Remove unknown characters
    text = ''.join(c for c in text if unicodedata.category(c) != 'Co')
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text
    
        
    
def upSertEmbeds(processed_text):
    def embed(text):
        embeds = co.embed(
            texts=text,
            model='embed-english-v3.0',
            input_type='search_document',
            truncate='END'
        ).embeddings
        return embeds

    embeds = embed(processed_text)

    shape = np.array(embeds).shape

    batch_size = 128

    ids = [str(i) for i in range(shape[0])]
    # create list of metadata dictionaries
    meta = [{'text': text} for text in processed_text]

    # create list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeds, meta))

    for i in range(0, shape[0], batch_size):
        i_end = min(i+batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])


def main():
    st.set_page_config(page_title="RAG Chatbot App", page_icon=":sunglasses:", layout="wide")

    # ----------------- Sidebar & Environment setup -----------------
    sidebar_func()
    
    if st.session_state.api_keys['pinecone_api_key'] and st.session_state.api_keys['cohere_api_key'] and st.session_state.api_keys['pinecone_index_name'] and st.session_state.api_keys['pinecone_environment']:
        
        pinecone = PineconeClient(api_key=st.session_state.api_keys['pinecone_api_key'])
        environment = st.session_state.api_keys['pinecone_environment']
        index = pinecone.Index(st.session_state.api_keys['pinecone_index_name'])

        co = cohere.Client(st.session_state.api_keys['cohere_api_key'])
    
        # conn = st.connection(
        #     "pinecone", 
        #     type=PineconeConnection, 
        #     api_key = st.session_state.api_keys['pinecone_api_key'],
        #     environment = st.session_state.api_keys['pinecone_environment'], 
        #     index_name = st.session_state.api_keys['pinecone_index_name']
        # )

        # st.sidebar.success('Connected to :blue[Pinecone] and :blue[Cohere]')
    st.sidebar.info("The app is a simple demonstration of a QA Chatbot using the RAG model.")

    # ----------------- Headers -----------------

    st.header("COSI217 final project: :blue[RAG]")
    st.write("### Current dataset is: ", '`'+str(st.session_state.current_dataset)+'`')

    # ----------------- File Upload -----------------

    uploaded_file = st.file_uploader("File upload", type="pdf")

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
        documents = read_pdf(path)
        processed_text = process_documents(documents)
        # Display processed text
        st.subheader("Processed Text:")
        st.write(processed_text)
        text_to_embed = [processed_text]
        upSertEmbeds(text_to_embed)
    

    input = st.text_input(
        "Ask a question", key="input")

    submit = st.button("Ask")

    if submit:
        response = get_open_ai_chat_response(input)
        st.subheader("The Response is:")
        st.write(response)
    
    if st.session_state.environment_status == 'dev':
        for element in st.session_state:
                st.write(f"{element}: {st.session_state[element]}")

    

if __name__ == "__main__":
    main()
