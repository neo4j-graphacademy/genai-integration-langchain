import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
# tag::import_embedding_model[]
from langchain_openai import OpenAIEmbeddings
# end::import_embedding_model[]
# tag::import_neo4j[]
from langchain_neo4j import Neo4jGraph, Neo4jVector
# end::import_neo4j[]

# Initialize the LLM
model = init_chat_model("gpt-4o", model_provider="openai")

# Create a prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# tag::embedding_model[]
# Create the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
# end::embedding_model[]

# tag::graph[]
# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)
# end::graph[]

# tag::plot_vector[]
# Create Vector
plot_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)
# end::plot_vector[]

# Define functions for each step in the application

# tag::retrieve_docs[]
# Retrieve context 
def retrieve(state: State):
    # Use the vector to find relevant documents
    context = plot_vector.similarity_search(
        state["question"], 
        k=6,
        # filter={"revenue": {"$lte": 2000000}}
    )
    return {"context": context}

# Generate the answer based on the question and context
def generate(state: State):
    messages = prompt.invoke({"question": state["question"], "context": state["context"]})
    response = model.invoke(messages)
    return {"answer": response.content}

# Define application steps
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
app = workflow.compile()

# Run the application
question = "What is the movie with the pig who wants to be a sheep dog?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
# tag::print_context[]
print("Context:", response["context"])
# end::print_context[]



# tag::examples[]
# What is the movie with the pig who wants to be a sheep dog?
# What are 3 movies about aliens coming to earth?
# end::examples[]