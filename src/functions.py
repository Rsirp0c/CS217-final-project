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
from langchain_community.vectorstores import Pinecone
from streamlit_pinecone import PineconeConnection
import streamlit as st
import os
import re
import numpy as np

dim2embed = { # mapping of dimensions to embeddings
                8: 'model1',
                8: 'model2',
                8: 'model3',}


def get_response(query, model):
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
    
        
    
def upSertEmbeds(processed_text,co, index, chunk):
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
