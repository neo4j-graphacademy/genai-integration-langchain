import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph
# tag::import_neo4jvector[]
from langchain_neo4j import Neo4jVector
# end::import_neo4jvector[]
# tag::import_embedding_model[]
from langchain_openai import OpenAIEmbeddings
# end::import_embedding_model[]

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

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

# tag::search[]
# Search for similar movie plots
plot = "Love conquers all"
result = plot_vector.similarity_search(plot, k=3)
print(result)
# end::search[]

# tag::results[]
# Parse the documents
for doc in result:
    print(f"Title: {doc.metadata['title']}")
    print(f"Plot: {doc.page_content}\n")
# end::results[]



# tag::examples[]
# Toys come alive
# Love conquers all
# Aliens invade Earth
# end::examples[]