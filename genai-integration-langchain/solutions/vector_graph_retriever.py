import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector

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

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Create the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# tag::retrieval_query[]
# Define the retrieval query
retrieval_query = """
MATCH (node)<-[r:RATED]-()
WITH node, score, avg(r.rating) AS userRating
RETURN 
    "Title: " + node.title + ", Plot: " + node.plot AS text, 
    score, 
    {
        title: node.title,
        genres: [ (node)-[:IN_GENRE]->(g) | g.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        userRating: userRating
    } AS metadata
ORDER BY userRating DESC
"""
# end::retrieval_query[]

# tag::plot_vector[]
# Create Vector
plot_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
    retrieval_query=retrieval_query,
)
# end::plot_vector[]

# Define functions for each step in the application

# Retrieve context 
def retrieve(state: State):
    # Use the vector to find relevant documents
    context = plot_vector.similarity_search(
        state["question"], 
        k=6,
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
question = "Who acts in movies about Love and Romance?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
print("Context:", response["context"])




""" tag::examples[]
* Who acts in movies about Love and Romance?
* What are top user rated movies about a house haunted by ghosts?
* What movies genres relate to movies about betrayal?
end::examples[] """