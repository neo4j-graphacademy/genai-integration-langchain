import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Create the embedding model

# Create Vector

# Search for similar movie plots

# Parse the documents
