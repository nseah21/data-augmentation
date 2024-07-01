from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import json
import csv

from embed_data import embed_data
from dotenv import load_dotenv

load_dotenv()

MAX_ROWS = 200
FILENAME = "./datasets/corona-sentiments.csv"

def retrieve_and_augment_data(input: str, iter_count: int):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    embedding_function = OpenAIEmbeddings()
    stores = Chroma(
        persist_directory="./chromadeeznuts", embedding_function=embedding_function
    )

    model = embed_data(FILENAME)

    template = """
        You are a creative AI assistant whose purpose is to perform data augmentation on a given set of data. 
        You will need to retrieve some data from the vector store, and use this information to generate a *list* of one or more entries of new data, with each entry having the following structure:
        
        {model}
        
        As much as possible, try to make the augmented data similar to but distinct from existing entries in the vector store. 
        
        You should try to generate new data that belong to different categories, instead of being fixated on just one category. 
        
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

    append_to_csv(FILENAME, data=results)

    return results


def append_to_csv(filepath, data):
    with open(filepath, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        for row in data:
            writer.writerow(row)


results = retrieve_and_augment_data(
    "Please generate data with negative Corona sentiments. Try to emulate how people text on Twitter, e.g. spelling mistakes and abbreviations.", 4
)
print(results)
