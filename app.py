# QA Learning Chat Bot
import streamlit as st
import os
import re
import numpy as np
from openai import OpenAI
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
    index = st.session_state.current_dataset[1]

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

upload,_, respond = st.columns([1, 0.1, 1])

# ----------------- File Upload -----------------
with upload:
    "#### :blue[Uploading] & :blue[Chunking]"

    if st.session_state.current_dataset:
        embed = st.session_state.datasets[st.session_state.current_dataset][1]
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
                         ['model 1', 'model 2', 'model 3'], 
                         help='Choose different LLM models to generate responses')


# ----------------- Chat -----------------

st.write("### Chat here 👋")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

# input = st.text_input(
#     "Ask a question", key="input")

# submit = st.button("Ask")

# if submit:
#     response = get_open_ai_chat_response(input)
#     st.subheader("The Response is:")
#     st.write(response)

with st.sidebar:
    if st.session_state.environment_status == 'dev':
        for element in st.session_state:
                st.write(f"{element}: {st.session_state[element]}")

