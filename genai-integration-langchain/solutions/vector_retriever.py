import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
# tag::import_neo4jgraph[]
from langchain_neo4j import Neo4jGraph
# end::import_neo4jgraph[]
# tag::import_neo4jvector[]
from langchain_neo4j import Neo4jVector
# end::import_neo4jvector[]
# tag::import_embedding_model[]
from langchain_openai import OpenAIEmbeddings
# end::import_embedding_model[]

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
# tag::state[]
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
# end::state[]

# tag::graph[]
# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)
# end::graph[]

# tag::embedding_model[]
# Create the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
# end::embedding_model[]

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

# tag::retrieve[]
# Retrieve context 
def retrieve(state: State):
    # Use the vector to find relevant documents
    context = plot_vector.similarity_search(
        state["question"], 
        k=6
    )
    return {"context": context}
# tag::retrieve[]

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
question = "What are some movies about fast cars?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
# tag::print_context[]
print("Context:", response["context"])
# end::print_context[]



# tag::examples[]
# What are some movies about fast cars?
# What are 3 movies about aliens coming to earth?
# What is a typical budget for a romance movie?
# end::examples[]