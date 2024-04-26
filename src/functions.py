from langchain_community.vectorstores import Pinecone

from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.messages.human import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import unicodedata

import streamlit as st
import cohere
from pinecone import Pinecone as PineconeClient
from sentence_transformers import SentenceTransformer
import os
import re
import numpy as np

embed2dim = { # embeddings to dimensions
                'Cohere-embed-english-v3.0': 1024,
                'all-MiniLM-L6-v2': 384,
                'snowflake-arctic-embed-m': 768}

dim2embed = { # dimensions to embeddings
                1024: 'Cohere-embed-english-v3.0',
                384: 'all-MiniLM-L6-v2',
                768: 'snowflake-arctic-embed-m'}

def embed(docs):
    curr_embedding_dimension = st.session_state.datasets[st.session_state.current_dataset][1]
    if curr_embedding_dimension == 1024:
        co = cohere.Client(st.session_state.api_keys['cohere_api_key'])
        embeds = co.embed(texts=docs, model='embed-english-v3.0',input_type='search_document').embeddings
    elif curr_embedding_dimension == 384:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeds = model.encode(docs)
    elif curr_embedding_dimension == 768:
        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m", trust_remote_code=True)
        embeds = model.encode(docs)
    return embeds
    

def upSertEmbeds(processed_text, index):
    '''
    @param processed_text: [[text1], [text2], ...]
    @param index: Pinecone index
    '''
    embeds = embed(processed_text)
    shape = np.array(embeds).shape
    vectors = []

    for i in range(shape[0]):
        vector = {'id': str(i),
                  'values': embeds[i],
                  'metadata': {'text': processed_text[i]}
                 }
        vectors.append(vector)

    index.upsert(vectors)


def get_response(query, model, top_k_val):
    '''
    @param query: a string. The question to ask the model.
    @param model: a string. The model to use for the response.
    @param recall: an int. The number of documents to retrieve.
    @return: a string. The response from the model.
    '''
    query_vector = embed([query]).tolist()
    pc = PineconeClient(api_key=st.session_state.api_keys['pinecone_api_key'])
    index = pc.Index(st.session_state.current_dataset)

    top_k_chunks = index.query(
                        vector = query_vector,
                        top_k = top_k_val,
                        include_values = False,
                        include_metadata = True
                    )
    
    retrieved_chunks = [match['metadata'].get('text', 'Default text') for match in top_k_chunks['matches']]

    # RAG prompt
    template =  """
                Answer the question based only on the following context:
                {context}
                Question: {question}
                """
    
    prompt = PromptTemplate.from_template(template)

    # RAG
    chain = (
        RunnableParallel(
            {"context": retrieved_chunks, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke(query)

    return response

def get_response1(query, model, top_k_val):
    '''
    @param query: a string. The question to ask the model.
    @param model: a string. The model to use for the response.
    @param recall: an int. The number of documents to retrieve.
    @return: a string. The response from the model.
    '''
    query_vector = embed([query])
    if type(query_vector) != list:
        query_vector = embed([query]).tolist()
    pc = PineconeClient(api_key=st.session_state.api_keys['pinecone_api_key'])
    index = pc.Index(st.session_state.current_dataset)

    top_k_chunks = index.query(
                        vector = query_vector,
                        top_k = top_k_val,
                        include_values = False,
                        include_metadata = True
                    )
    
    text_chunks = [match['metadata'].get('text', 'Default text') for match in top_k_chunks['matches']]

    # RAG prompt
    prompt =  f"""
                Answer the question based only on the following context:
                {text_chunks}
                Question: {query}
                """
    query = HumanMessage(query)
    chain = ConversationChain(
        llm=model,
        memory=ConversationBufferMemory()
    )

    response = chain.invoke(prompt)['response']

    return response, top_k_chunks

# Reading pdfs
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    return documents


# Process pdfs
def process_documents(documents):
    '''
    @param documents: List of documents.
    @return: a long sting. Concatenated text from document.
    '''
    doc_text = ''
    for doc in documents:

        text = doc.page_content
        text = text.replace('\n', ' ')
        # Remove unknown characters
        text = ''.join(c for c in text if unicodedata.category(c) != 'Co')
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        doc_text += text

    return doc_text

def chunking(text):
    '''
    @param text: a long string
    @return: a list of strings. Each string is a chunk of the text.
    '''
    chunk_size = len(text) // 10
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks
