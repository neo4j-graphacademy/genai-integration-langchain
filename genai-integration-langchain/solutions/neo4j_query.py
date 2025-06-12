import os
from dotenv import load_dotenv
load_dotenv()

# tag::import_neo4j[]
from langchain_neo4j import Neo4jGraph
# end::import_neo4j[]

# tag::neo4jgraph[]
# Create Neo4jGraph instance
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)
# end::neo4jgraph[]

# tag::query[]
# Run a query and print the result
result = graph.query("""
MATCH (m:Movie {title: "Mission: Impossible"})<-[a:ACTED_IN]-(p:Person)
RETURN p.name AS actor, a.role AS role
""")

print(result)
# end::query[]