from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import qdrant_client

# Load the documents
# loader = TextLoader("pcre.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

url=""
api_key=""

# Create a Qdrant client (from qdrant)
client = qdrant_client.QdrantClient(
    url=url,
    api_key = api_key, # For Qdrant Cloud, None for local instance
)

# use the Qdrant client as a parameter to create a Qdrant object in langchain
doc_store = Qdrant(
    client=client, collection_name="my_documents", 
    embeddings=embeddings,
)

# Create a collection and upload the documents

# qdrant = Qdrant.from_documents(
#     docs,
#     embeddings,
#     url=url,
#     prefer_grpc=True,
#     api_key=api_key,
#     collection_name="my_documents",
# )

query = "who is the writer of the book"

# perform a similarity search
found_docs = doc_store.similarity_search(query)
print(found_docs[0].page_content, "\n")

# perform a similarity search with score
found_docs = doc_store.similarity_search_with_score(query)
found_docs = found_docs[:2]
for i, doc in enumerate(found_docs):
    print(f"{i + 1}.",doc[1], "\n", doc[0].page_content, "\n")


# perform a max marginal relevance search
found_docs = doc_store.max_marginal_relevance_search(query, k=2, fetch_k=10)
for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")

