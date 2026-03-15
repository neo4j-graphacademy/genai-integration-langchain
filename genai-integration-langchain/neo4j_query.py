import os
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph

# Create Neo4jGraph instance
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)

# Run a query and print the result
result = graph.query("""
MATCH (m:Movie {title: "Mission: Impossible"})<-[a:ACTED_IN]-(p:Person)
RETURN p.name AS actor, a.role AS role
""")

print(result)
