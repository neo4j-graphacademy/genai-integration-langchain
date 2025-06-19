import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph

# Initialize the LLM
model = init_chat_model(
    "gpt-4o", 
    model_provider="openai"
)

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Create the Cypher QA chain

# Invoke the chain

