from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import create_model
import pandas as pd
import chardet
import json


def embed_data(filepath):
    # Extract column names from CSV
    with open(filepath, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]

    df = pd.read_csv(filepath, encoding=encoding, nrows=0)
    fieldnames = list(df.columns)[:-1]

    fields = {field: (str, ...) for field in fieldnames}
    model = create_model("CSVModel", **fields)

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
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chromadeeznuts",
    )

    return model


if __name__ == "__main__":
    embed_data("./datasets/corona-sentiments.csv")
