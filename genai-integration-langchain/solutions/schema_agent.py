import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
# tag::import_neo4jgraph[]
from langchain_neo4j import Neo4jGraph
# end::import_neo4jgraph[]

# tag::neo4jgraph[]
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)
# end::neo4jgraph[]

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
    context: dict
    answer: str

# Define functions for each step in the application

# tag::retrieve[]
# Retrieve context 
def retrieve(state: State):
    context = graph.query("call db.schema.visualization()")
    return {"context": context}
# end::retrieve[]

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
# tag::question[]
question = "How is the graph structured?"
# end::question[]
response = app.invoke({"question": question})
print("Answer:", response["answer"])



"""
tag::examples[]
How is the graph structured?
How are Movie nodes connected to Person nodes?
What relationships are in the graph?
What properties do Movie nodes have?
How would I find user ratings for a movie?
end::examples[]
"""