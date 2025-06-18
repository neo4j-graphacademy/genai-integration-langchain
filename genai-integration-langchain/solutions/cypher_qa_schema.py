import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain

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

# tag::cypher_qa_include[]
# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model, 
    include_types=["Movie", "ACTED_IN", "Person"],
    allow_dangerous_requests=True,
    verbose=True, 
)
# end::cypher_qa_include[]

# tag::cypher_qa_exclude[]
# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model, 
    exclude_types=["User", "RATED"],
    allow_dangerous_requests=True,
    verbose=True, 
)
# end::cypher_qa_exclude[]

# tag::invoke[]
# Invoke the chain
question = "Who acted in the movie Aliens?"
response = cypher_qa.invoke({"query": question})
print(response["result"])
# end::invoke[]



"""
tag::examples[]
Who acted in the movie Aliens?
Who directed the movie Superman?
What is the plot of the movie Toy Story?
How many movies are in the Sci-Fi genre?
end::examples[]
"""