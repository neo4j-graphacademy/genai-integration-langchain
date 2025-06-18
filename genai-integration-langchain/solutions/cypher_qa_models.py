import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4o", 
    model_provider="openai"
)

# tag::cypher_model[]
cypher_model = init_chat_model(
    "gpt-4o", 
    model_provider="openai",
    temperature=0.0
)
# end::cypher_model[]

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# tag::cypher_qa[]
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model, 
    cypher_llm=cypher_model,
    allow_dangerous_requests=True,
    verbose=True,
)
# end::cypher_qa[]

question = "Who acted in the movie Aliens?"
response = cypher_qa.invoke({"query": question})
print(response)
