from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
import json
import csv

from embed_data import embed_data
from dotenv import load_dotenv

load_dotenv()

MAX_ROWS = 200


def retrieve_and_augment_data(input: str, iter_count: int):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    embedding_function = OpenAIEmbeddings()
    stores = Chroma(
        persist_directory="./chromadb", embedding_function=embedding_function
    )

    model = embed_data("./datasets/stories.csv")

    template = """
        You are a creative AI assistant whose purpose is to perform data augmentation on a given set of data. 
        You will need to retrieve some data from the vector store, and use this information to generate a *list* of one or more entries of new data, with each entry having the following structure:
        
        {model}
        
        You are to come up with a new scam story description based on the existing scam data in the vector store, and then label it with an appropriate scam type.
        As much as possible, try to make the augmented data similar but distinct to existing entries in the vector store. 
        
        You should try to generate new data that belong to different types of scams, instead of being fixated on just one type of scam. 
        
        Context: {context}
        Description: {description}
    """
    parser = JsonOutputParser(pydantic_object=model)

    retriever = stores.as_retriever(
        search_type="mmr", search_kwargs={"fetch_k": 60, "k": 30}
    )
    prompt = ChatPromptTemplate.from_messages(["human", template])
    rag_chain = (
        {
            "context": retriever,
            "description": RunnablePassthrough(),
            "model": lambda _: json.dumps(
                {property: "str" for property in model.schema().get("properties")},
                indent=2,
            ),
        }
        | prompt
        | llm
        | parser
    )

    iter_count = min(iter_count, MAX_ROWS)

    results = []
    for _ in range(iter_count):
        results.append(rag_chain.invoke(input))

    append_to_csv("./datasets/stories.csv", data=results)

    return results


def append_to_csv(filepath, data):
    with open(filepath, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        for row in data:
            writer.writerow(row)


results = retrieve_and_augment_data(
    "Please augment data relating to impersonation scams.", 2
)
print(results)
