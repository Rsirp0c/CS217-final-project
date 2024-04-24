# QA Learning Chat Bot
import streamlit as st
import os
import re
import numpy as np
# from openai import OpenAI
from langchain_openai import OpenAI
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from langchain_community.llms.llamafile import Llamafile
import tempfile

import cohere
from pinecone import Pinecone as PineconeClient

# import the storing of static information in a separate file
from src.constant import *
from src.sidebar import *
from src.functions import *

st.set_page_config(page_title="RAG Chatbot App", page_icon=":sunglasses:", layout="wide")

# ----------------- Sidebar & Environment setup -----------------

sidebar_func()

if st.session_state.api_keys['pinecone_api_key']:
    pinecone = PineconeClient(api_key=st.session_state.api_keys['pinecone_api_key'])
        
if  st.session_state.api_keys['cohere_api_key']:
    co = cohere.Client(st.session_state.api_keys['cohere_api_key'])

if st.session_state.current_dataset:
    index = pinecone.Index(st.session_state.current_dataset)

with st.sidebar:
    if st.button('Delete Current Dataset'):
        pinecone.delete_index(st.session_state.current_dataset)
        st.session_state.datasets.pop(st.session_state.current_dataset)
        st.session_state.current_dataset = None
        st.rerun()
    if st. button('Delete All Datasets'):
        for index in st.session_state.datasets:
            pinecone.delete_index(index)
        st.session_state.datasets = {}
        st.session_state.current_dataset = None
        st.rerun()
    st.info("The app is a simple demonstration of a QA Chatbot using the RAG model.")

# ----------------- Headers -----------------

st.header("COSI217 final project: :blue[RAG]")
st.write("### Current dataset is: ", '`'+str(st.session_state.current_dataset)+'`')

upload,_, respond = st.columns([1, 0.1, 1])

# ----------------- File Upload -----------------
with upload:
    "#### :blue[Uploading] & :blue[Chunking]"

    if st.session_state.current_dataset:
        dimensions = st.session_state.datasets[st.session_state.current_dataset][1]
        embed = dim2embed[dimensions]
        st.write("**Current Embedding model is: ", '`'+embed+'`**') 
    
    chunk = st.radio( "###### Choose chunking strategy 👇",
                    options = ["chunk1", "chunk2", "chunk3"],
                    # captions=["Chunk 1", "Chunk 2", "Chunk 3"]
                    )

    uploaded_file = st.file_uploader("Upload file", type="pdf")

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
        documents = read_pdf(path)
        processed_text = process_documents(documents)
        # # Display processed text
        # st.subheader("Processed Text:")
        # st.write(processed_text)
        text_to_embed = [processed_text]
        upSertEmbeds(text_to_embed, co, index, chunk)

# ----------------- Respond setting -----------------

with respond:
    "#### :blue[Calling] & :blue[Responding]"
    
    recall_number = st.number_input('###### Choose the number of recall',value=3, step=1)

    model = st.radio('###### Select the LLM model 👇', 
                         ['OpenAI', 'Cohere', 'TinyLlama'], 
                         help='Choose different LLM models to generate responses')
    if model == 'TinyLlama':
        st.warning('Follow the insturction [here](https://python.langchain.com/docs/integrations/llms/llamafile/) and download the TinyLlama model before use!', icon="⚠️")


"---"
# ----------------- Chat -----------------

st.write("### Chat here 👋")

if model == "OpenAI" and st.session_state.api_keys['openai_api_key']:
    if st.session_state.api_keys['openai_api_key']:
        client = OpenAI(api_key=st.session_state.api_keys['openai_api_key'])
    else:
        st.error("No OpenAI Api key")
elif model == "Cohere" and st.session_state.api_keys['cohere_api_key']:
    if st.session_state.api_keys['cohere_api_key']:
        client = ChatCohere(
                cohere_api_key=st.session_state.api_keys['cohere_api_key']
            )
    else:
        st.error("No Cohere Api key")
elif model == "TinylLlama":
    client = Llamafile()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    prompt = message = [HumanMessage(content=prompt)]

    with st.chat_message("assistant"):
        if uploaded_file or st.session_state.current_dataset: 
            # response = client.invoke(prompt).content
            response = get_response(prompt, client, recall_number)
        else:
            response = "Please upload a file to get started. Chat soon!😝"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    if st.session_state.environment_status == 'dev':
        for element in st.session_state:
                st.write(f"{element}: {st.session_state[element]}")

