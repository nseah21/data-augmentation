# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd


def embed_data(filepath):
    # Extract column names from CSV
    df = pd.read_csv(filepath)
    fieldnames = list(df.columns)

    print(fieldnames)
    return 

    # Load data using CSVLoader
    loader = CSVLoader(
        file_path=filepath,
        csv_args={"delimiter": ",", "quotechar": '"', "fieldnames": fieldnames},
    )
    docs = loader.load()

    # Split documents into smaller chunks to aid in indexing
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embed data into vector stores
    stores = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory="./chromadb"
    )

    return stores


if __name__ == "__main__":
    embed_data()
