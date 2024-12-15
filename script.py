import os
import requests
import json
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Step 1: Download JSON file
def download_json_file(url, save_path):
    if os.path.exists(save_path):
      print(f"{save_path} already exists. Skipping download.")
      return
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading JSON file"):
                file.write(chunk)
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

# Step 2: Load JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Step 3: Convert JSON data to Documents
def json_to_documents(data, key='name', text_key='oracle_text'):
    documents = []
    for item in data:
        if key in item and text_key in item:
            content = f"{item[key]}: {item[text_key]}"
            documents.append(Document(page_content=content))
    return documents

# Step 4: Chunk documents
def chunk_documents(documents, chunk_size=500, overlap=50):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_documents([doc]))
    return chunks


def setup_retriever(chunks):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en')

    # Create the vector store
    vector_store = InMemoryVectorStore(embeddings)

    # Add documents to the vector store with embedded vectors
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk.page_content)  # Embed the document text
        vector_store.aadd_documents(chunk, embedding=embedding)

    # Setup retriever
    retriever = MultiVectorRetriever(vector_store=vector_store, embeddings=embeddings)
    return retriever

# Step 6: Query Function
def test_query(retriever, query, top_k=5):
    """
    Test the retriever with a query and return the top_k results.
    """
    results = retriever.get_relevant_documents(query, k=top_k)
    print(f"\nQuery: {query}")
    print("Top Results:")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result.page_content}\n")

# Step 7: Main Script
def main():
    # URL of the JSON file
    url = "https://data.scryfall.io/default-cards/default-cards-20241215100720.json"
    save_path = "default-cards.json"

    # Step 1: Download JSON file
    print("Downloading JSON file...")
    download_json_file(url, save_path)

    # Step 2: Load JSON data
    print("Loading JSON data...")
    data = load_json(save_path)

    # Step 3: Convert JSON data to Documents
    print("Converting JSON data to Documents...")
    documents = json_to_documents(data)

    # Step 4: Chunk documents
    print("Chunking documents...")
    chunks = chunk_documents(documents)

    # Step 5: Setup Retriever
    print("Setting up retriever...")
    retriever = setup_retriever(chunks)

    print("Retriever setup complete!")
    return retriever

if __name__ == "__main__":
    retriever = main()

    # Allow user to input queries for testing
    while True:
        print("\nEnter a query to test the retriever (or type 'exit' to quit):")
        user_query = input("Query: ")
        if user_query.lower() == 'exit':
            print("Exiting...")
            break
        test_query(retriever, user_query)