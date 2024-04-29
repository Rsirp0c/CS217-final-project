# QA Learning Chat Bot
import streamlit as st
import os
import re
import numpy as np
import pandas as pd
# from openai import OpenAI
from langchain_openai import OpenAI
from langchain_cohere import ChatCohere
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

upload, _, respond = st.columns([1, 0.1, 1])

# ----------------- File Upload -----------------
with upload:
    "#### :blue[Uploading] & :blue[Chunking]"

    if st.session_state.current_dataset:
        dimensions = st.session_state.datasets[st.session_state.current_dataset][1]
        embed = dim2embed[dimensions]
        st.write("**Current Embedding model is: ", '`'+embed+'`**') 
    
    chunk = st.radio( "###### Choose chunking strategy 👇",
                    options = ["character text splitter", "recursive character text splitter", "spacy text splitter"],
                    help="Choose different chunking strategies to split the document into smaller parts"
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
        if chunk == "character text splitter":
            processed_text_chunks = character_text_splitter(processed_text)
        elif chunk == "recursive character text splitter":
            processed_text_chunks = recursive_character_text_splitter(processed_text)
        elif chunk == "spacy text splitter":
            processed_text_chunks = spacy_text_splitter(processed_text)
        upSertEmbeds(processed_text_chunks, index)

# ----------------- Respond setting -----------------

if 'top_k_chunks' not in st.session_state:
    st.session_state.top_k_chunks = None

with respond:
    "#### :blue[Calling] & :blue[Responding]"
    if st.session_state.current_dataset:
        metric = st.session_state.datasets[st.session_state.current_dataset][2]
        st.write("**Current distence metric is: ", '`'+metric+'`**') 
    # if st.session_state.current_dataset:
    #     max_k = index.describe_index_stats()['namespaces']['']['vector_count']
    #     # max_k = index.describe_index_stats()['total_vector_count']
    # else:
    # max_k = 5
    recall_number = st.number_input('###### Choose the number of retrieval',value=3, step=1, min_value=1)

    model = st.radio('###### Select the LLM model 👇', 
                         ['OpenAI', 'Cohere', 'TinyLlama'], 
                         help='Choose different LLM models to generate responses')
    if model == 'TinyLlama':
        st.warning('Follow the insturction [here](https://python.langchain.com/docs/integrations/llms/llamafile/) and download the TinyLlama model before use!', icon="⚠️")

if st.session_state.top_k_chunks:
    df = pd.DataFrame(
        {
        'id': [match['id'] for match in st.session_state.top_k_chunks['matches']],
        'score': [match['score'] for match in st.session_state.top_k_chunks['matches']],
        'text': [' '.join(match['metadata']['text'].split()[:10])+'...' for match in st.session_state.top_k_chunks['matches']]
        }
    )
    st.data_editor(
    df,
    column_config={
        "score": st.column_config.ProgressColumn(
            "similarity score",
            help="The similarity score between the query and the document.",
            min_value=0,
            max_value=1,
        ),
    },
    hide_index=True,
)
# ----------------- prompt engineer -----------------
"---"
"#### :blue[Prompt Generation] & :blue[Reranking](optional)"
'''
**Description:**\n
The normal retrival will use the **single** provided prompt to retrive the **top_k** documents.\n 
However, this feature:
1. Generate **a list of questions** based on the prompt you provided.
2. Use the generated questions to retrive `len(questions)*top_k` documents.
3. Rerank the documents using default `Cohere Rerank model` based on your initial prompt.
'''
if 'advanced_option' not in st.session_state:
    st.session_state.advanced_option = False
on = st.toggle('Activate feature')
if on:
    st.session_state.advanced_option = True
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
    # prompt = HumanMessage(content=prompt)
    # print(prompt)

    with st.chat_message("assistant"):
        if uploaded_file or st.session_state.current_dataset: 
            if st.session_state.advanced_option:
                queries = generate_queries(client, prompt, recall_number)
                if queries:
                    df = pd.DataFrame(queries, columns=['Generated Questions'])
                    st.data_editor(df, hide_index=True)
                text_chunks = []
                for query in queries:
                    text_chunk, top_k_chunks = retrieve_documents(query, recall_number)
                    text_chunks += text_chunk
                print("all docs retrieved")
                reranked_result = get_reranked_result(prompt, text_chunks, recall_number)
                response = get_response1(prompt, client, reranked_result)
            else:
                text_chunks, top_k_chunks = retrieve_documents(prompt, recall_number)
                response, top_k_chunks = get_response1(prompt, client, text_chunks)
            print(top_k_chunks)
            # response = client.invoke(prompt).content
            # response = get_response(prompt, client, recall_number)
        else:
            response = "Please upload a file to get started. Chat soon!😝"
        st.markdown(response)

        if top_k_chunks:
                df = pd.DataFrame(
                    {
                    'id': [match['id'] for match in top_k_chunks['matches']],
                    'score': [match['score'] for match in top_k_chunks['matches']],
                    'text': [' '.join(match['metadata']['text'].split()[:13])+'...' for match in top_k_chunks['matches']]
                    }
                )
                st.data_editor(df,
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "similarity score",
                        help="The similarity score between the query and the document.",
                        min_value=0,
                        max_value=1,
                    ),},
                hide_index=True,)
        
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    if st.session_state.environment_status == 'dev':
        for element in st.session_state:
                st.write(f"{element}: {st.session_state[element]}")

