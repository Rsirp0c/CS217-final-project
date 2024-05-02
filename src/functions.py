from langchain_community.vectorstores import Pinecone

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import unicodedata

# import splitters library
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import CharacterTextSplitter


import streamlit as st
import cohere
from pinecone import Pinecone as PineconeClient
from sentence_transformers import SentenceTransformer
import os
import re
import json
import ast
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


# def get_response(query, model, top_k_val, filters):
#     '''
#     @param query: a string. The question to ask the model.
#     @param model: a string. The model to use for the response.
#     @param recall: an int. The number of documents to retrieve.
#     @return: a string. The response from the model.
#     '''
#     query_vector = embed([query]).tolist()
#     pc = PineconeClient(api_key=st.session_state.api_keys['pinecone_api_key'])
#     index = pc.Index(st.session_state.current_dataset)

#     top_k_chunks = index.query(
#                         vector = query_vector,
#                         top_k = top_k_val,
#                         filter= filters,
#                         include_values = False,
#                         include_metadata = True
#                     )
    
#     retrieved_chunks = [match['metadata'].get('text', 'Default text') for match in top_k_chunks['matches']]

#     # RAG prompt
#     template =  """
#                 Answer the question based only on the following context:
#                 {context}
#                 Question: {question}
#                 """
    
#     prompt = ChatPromptTemplate.from_template(template)

#     # RAG
#     chain = (
#         RunnableParallel(
#             {"context": retrieved_chunks, "question": RunnablePassthrough()})
#         | prompt
#         | model
#         | StrOutputParser()
#     )

#     response = chain.invoke(query)

#     return response

def retrieve_documents(query, top_k_val, filters):
    '''
    @param query: a string. The question to ask the model.
    @param top_k_val: an int. The number of documents to retrieve.
    @return: a list of strings. Each string is a chunk of the text.
    '''
    query_vector = embed([query])
    if type(query_vector) != list:
        query_vector = embed([query]).tolist()
    pc = PineconeClient(api_key=st.session_state.api_keys['pinecone_api_key'])
    index = pc.Index(st.session_state.current_dataset)

    top_k_chunks = index.query(
                        vector = query_vector,
                        top_k = top_k_val,
                        filter= filters,
                        include_values = False,
                        include_metadata = True
                    )
    text_chunks = [match['metadata'].get('text', 'Default text') for match in top_k_chunks['matches']]

    return text_chunks, top_k_chunks

def get_response(query, model, text_chunks):
    '''
    @param query: a string. The question to ask the model.
    @param model: a string. The model to use for the response.
    @param recall: an int. The number of documents to retrieve.
    @return: a string. The response from the model.
    '''

    # RAG prompt
    prompt =  f"""
                Answer the question based only on the following context:
                {text_chunks}
                Question: {query}
                """
    
    response = model.invoke(prompt).content

    return response

# Reading pdfs
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    return documents


# Process pdfs and return text
def process_documents(documents):
    '''
    @param documents: List of documents.
    @return: a long sting. Concatenated text from document.
    '''
    doc_text = ''
    for doc in documents:

        text = doc.page_content
        # Remove unknown characters
        text = ''.join(c for c in text if unicodedata.category(c) != 'Co')
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s,.]', '', text)
        # Convert to lowercase
        text = text.lower()
        doc_text += text

    return doc_text

def character_text_splitter(text):
    '''
    @param text: a long string
    @return: a list of strings. Each string is a chunk of the text.
    '''

    chunk_size = 500

    if len(text) > 2000:
        chunk_size = 800

    text_splitter = CharacterTextSplitter(   
        separator = " ", # split by space
        chunk_size = chunk_size, # split into chunks of 1000 characters
        chunk_overlap  = 20, # overlap by 200 characters
        length_function = len, # use len function to calculate length
    )
    texts = text_splitter.split_text(text)
    return texts

def recursive_character_text_splitter(text):
    '''
    @param text: a long string
    @return: a list of strings. Each string is a chunk of the text.
    '''

    chunk_size = 500

    if len(text) > 2000:
        chunk_size = 800 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # split into chunks of 100 characters
        chunk_overlap=20, # overlap by 20 characters
        length_function=len, 
        separators=["\n\n", "\n","(?<=\. )", " ", ""], # split by new line, space, and period
        is_separator_regex=True, # use regex for separators
    )
    texts = text_splitter.split_text(text)
    return texts
    
def spacy_text_splitter(text):
    '''
    @param text: a long string
    @return: a list of strings. Each string is a chunk of the text.
    '''

    chunk_size = 500

    if len(text) > 2000:
        chunk_size = 800

    text_splitter = SpacyTextSplitter(
        pipeline="en_core_web_sm",
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    texts = text_splitter.split_text(text)
    return texts

def str_to_json(s):
    try:
        # First, attempt to parse the string as JSON
        return json.loads(s)
    except json.JSONDecodeError:
        # If it fails, assume the string might be a Python literal
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            # Handle the case where parsing fails for both methods
            print("Error: Input string is neither valid JSON nor a valid Python literal.")
            return None

def generate_queries(model, prompt, num_queries):
    '''
    for generating questions based on a prompt
    '''
    query_generation_prompt = ChatPromptTemplate.from_template("Given the prompt: '{prompt}', generate {num_queries} questions that are better articulated. Return in the form of an list. For example: ['question 1', 'question 2', 'question 3']")
    query_generation_chain = query_generation_prompt | model
    return str_to_json(query_generation_chain.invoke({"prompt": prompt, "num_queries": num_queries}).content)

def get_reranked_result(query, docs, top_n):
    co = cohere.Client(st.session_state.api_keys['cohere_api_key'])
    if(len(docs) == 0):
        return {}
    rerank_results = co.rerank(model="rerank-english-v2.0", query=query, documents=docs, top_n=top_n, return_documents=True)
    results = {}
    for idx, r in enumerate(rerank_results.results):
        rank = idx + 1
        document_index = r.index
        document = r.document.text
        score =  f"{r.relevance_score:.2f}"
        results[rank] = {"document": document, "score": score, "index": document_index}
    return results