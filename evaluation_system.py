import toml
import numpy as np
import pandas as pd
# from openai import OpenAI
# from langchain_openai import OpenAI
# from langchain_cohere import ChatCohere
# from langchain_community.llms.llamafile import Llamafile
# import tempfile

import cohere
from langchain_cohere import ChatCohere
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec

from src.functions import *

# # API keys
# with open('.streamlit/secrets.toml', 'r') as f:
#     config = toml.load(f)

# cohere_api_key = config['COHERE_API_KEY']
# pinecone_api_key = config['PINECONE_API_KEY']
# openai_api_key = config['OPENAI_API_KEY']

COHERE_API_KEY = ""
PINECONE_API_KEY = ""

#create and config a pinecone client
def create_pinecone_client(name, model, metrics):
    '''
    @param name: name of the index
    @param model: snowflake/cohere/all-MiniLM
    @param metrics: euclidean/cosine/dotproduct
    @return: pinecone client
    '''
    pc = PineconeClient(api_key = PINECONE_API_KEY)
    pc.create_index(
        name=name,
        dimension=embed2dim[model],       
        metric=metrics,
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 
    return pc

def embed_tesing(docs, co, curr_embedding_dimension):
    if curr_embedding_dimension == 1024:
        embeds = co.embed(texts=docs, model='embed-english-v3.0',input_type='search_document').embeddings
    elif curr_embedding_dimension == 384:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeds = model.encode(docs)
    elif curr_embedding_dimension == 768:
        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m", trust_remote_code=True)
        embeds = model.encode(docs)
    return embeds

def upSertEmbeds_testing(processed_text, index, co, curr_embedding_dimension):
    '''
    @param processed_text: [[text1], [text2], ...]
    @param index: Pinecone index
    '''
    embeds = embed_tesing(processed_text, co, curr_embedding_dimension)
    shape = np.array(embeds).shape
    vectors = []

    for i in range(shape[0]):
        vector = {'id': str(i),
                  'values': embeds[i],
                  'metadata': {'text': processed_text[i]}
                 }
        vectors.append(vector)

    index.upsert(vectors)

def chunking_and_uploading(path, chunk, index, co, curr_embedding_dimension):
    '''
    @param path: path to the pdf file
    @param chunk: character text splitter/recursive character text splitter/spacy text splitter
    @param index: index name
    '''
    documents = read_pdf(path)
    processed_text = process_documents(documents)
    if chunk == "character text splitter":
        processed_text_chunks = character_text_splitter(processed_text)
    elif chunk == "recursive character text splitter":
        processed_text_chunks = recursive_character_text_splitter(processed_text)
    elif chunk == "spacy text splitter":
        processed_text_chunks = spacy_text_splitter(processed_text)
    upSertEmbeds_testing(processed_text_chunks, index, co, curr_embedding_dimension)



def retrieve_documents_without_filter(query, top_k_val, index, co, curr_embedding_dimension):
    '''
    @param query: a string. The question to ask the model.
    @param top_k_val: an int. The number of documents to retrieve.
    @return: a list of strings. Each string is a chunk of the text.
    '''
    query_vector = embed_tesing([query], co, curr_embedding_dimension)
    if type(query_vector) != list:
        query_vector = embed_tesing([query],co, curr_embedding_dimension).tolist()

    top_k_chunks = index.query(
                        vector = query_vector,
                        top_k = top_k_val,
                        include_values = False,
                        include_metadata = True
                    )
    text_chunks = [match['metadata'].get('text', 'Default text') for match in top_k_chunks['matches']]

    return text_chunks

def get_reranked_result_test(query, docs, top_n, co):
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


def prompt_response(prompt_extension: bool, top_k_val: int, prompt: str, client, index, co, curr_embedding_dimension):
    response = ""
    if prompt_extension:
        num_queries = min(max(top_k_val // 10, 5), 10)
        queries = generate_queries(client, prompt, num_queries)
        if queries:
            text_chunks = []
            for query in queries:
                retrive_num = top_k_val // num_queries + 3
                text_chunk = retrieve_documents_without_filter(query, retrive_num, index, co, curr_embedding_dimension)
                text_chunks += text_chunk
            print("all docs retrieved")
            reranked_result = get_reranked_result_test(prompt, text_chunks, top_k_val, co)
            reranked_chunks = [entry['document'] for entry in reranked_result.values()]
            response = get_response(prompt, client, reranked_chunks)
    else:
        text_chunks = retrieve_documents_without_filter(prompt, top_k_val, index, co, curr_embedding_dimension)
        response = get_response(prompt, client, text_chunks)

    return response
    
def hyperparameter_tuning(pdf_path,models,metrics,chunking,prompt_extensions,top_k,prompts) -> None:
    client = ChatCohere(cohere_api_key=COHERE_API_KEY)
    co = cohere.Client(COHERE_API_KEY)
    res_list = []
    for model in models:
        for metric in metrics:
            name = f"{model}-{metric}"
            print(f"Model: {model}, Metric: {metric}")
            pc = create_pinecone_client(name, model, metric)
            index = pc.Index(name)
            print("index created")
            curr_embedding_dimension = embed2dim[model]
            for chunk in chunking:
                chunking_and_uploading(pdf_path, chunk, index, client, curr_embedding_dimension)
                print("chunking done")
                for prompt_extension in prompt_extensions:
                    for k in top_k:
                        for prompt in prompts:
                            response = prompt_response(prompt_extension, k, prompt, client, index, co, curr_embedding_dimension)
                            res_list.append([model, metric, chunk, prompt_extension, k, prompt, response])
                            print("progress: ", len(res_list))
                index.delete(deleteAll=True)
            pc.delete_index(name)

        res_df = pd.DataFrame(res_list, columns=['Model', 'Metric', 'Chunking', 'Prompt Extension', 'Top K', 'Prompt', 'Response'])
        res_df.to_csv('hyperparameter_tuning_results.csv', index=False)


pdf_path = 'src/demo_hw7.pdf'
# models = ['snowflake-arctic-embed-m', 'all-MiniLM-L6-v2', 'cohere-embed-english-v3.0']
# metrics = ['cosine', 'euclidean', 'dotproduct']
# chunking = ['character text splitter', 'recursive character text splitter', 'spacy text splitter']
models = ['snowflake-arctic-embed-m']
metrics = ['cosine',"dotproduct"]
chunking = ['character text splitter', 'recursive character text splitter']
prompt_extensions = [True, False]
top_k = [5, 10]
prompt = ["what should I do for this assignment?",
          "How transformer works in this assignment?"]
          
'''
"What is the purpose of this assignment?",
          "How grading works for this assignment?",
          "What is the deadline for this assignment?",
'''          
hyperparameter_tuning(pdf_path,models,metrics,chunking,prompt_extensions,top_k,prompt)
