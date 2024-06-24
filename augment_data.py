from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()


class AugmentedData(BaseModel):
    description: str = Field(
        description="A description of a scam experience provided by a scam victim"
    )
    type_of_scam: str = Field(
        description="The category or type of scam based on its description"
    )


def retrieve_and_augment_data(input: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    embedding_function = OpenAIEmbeddings()
    stores = Chroma(
        persist_directory="./chromadb", embedding_function=embedding_function
    )

    template = """
        You are a creative AI assistant whose purpose is to perform data augmentation on a given set of data. 
        You will need to retrieve some data from the vector store, and use this information to generate new data in the following in JSON format:
        {{
            "description": "The description of the generated scam",
            "type_of_scam": "The category of the generated scam"
        }}
        
        You are to come up with a new scam story description based on the existing scam data in the vector store, and then label it with an appropriate scam type.
        As much as possible, try to make the augmented data similar but distinct to existing entries in the vector store. 
        
        You should try to generate new data that belong to different types of scams, instead of being fixated on just one type of scam. 
        
        Context: {context}
        Description: {description}
    """
    parser = JsonOutputParser(pydantic_object=AugmentedData)

    retriever = stores.as_retriever(
        search_type="mmr", search_kwargs={"fetch_k": 60, "k": 30}
    )
    prompt = ChatPromptTemplate.from_messages(["human", template])
    rag_chain = (
        {"context": retriever, "description": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    return rag_chain.invoke(input)


result = retrieve_and_augment_data("Please generate 1 new entry of augmented data.")
print(result)
