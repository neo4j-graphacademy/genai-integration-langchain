import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain.chat_models import init_chat_model
# tag::import_prompt[]
from langchain_core.prompts.prompt import PromptTemplate
# end::import_prompt[]

model = init_chat_model(
    "gpt-4o", 
    model_provider="openai"
)

cypher_model = init_chat_model(
    "gpt-4o-mini", 
    model_provider="openai",
    temperature=0.0
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# tag::cypher_template[]
# Cypher template
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
# end::cypher_template[]

# tag::cypher_template_instructions[]
# Cypher template with additional instructions
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, for example "The 39 Steps" becomes "39 Steps, The".

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
# end::cypher_template_instructions[]

# tag::cypher_template_examples[]
# Cypher template with examples
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, for example "The 39 Steps" becomes "39 Steps, The".

Schema:
{schema}
Examples:
1. Question: Get user ratings?
   Cypher: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE u.name = "User name" RETURN r.rating AS userRating
2. Question: Get average rating for a movie?
   Cypher: MATCH (m:Movie)<-[r:RATED]-(u:User) WHERE m.title = 'Movie Title' RETURN avg(r.rating) AS userRating

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
# end::cypher_template_examples[]

# tag::cypher_template_example_genre[]
# Cypher template with examples
cypher_template2 = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, for example "The 39 Steps" becomes "39 Steps, The".

Schema:
{schema}
Examples:
1. Question: Get user ratings?
   Cypher: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE u.name = "User name" RETURN r.rating AS userRating
2. Question: Get average rating for a movie?
   Cypher: MATCH (m:Movie)<-[r:RATED]-(u:User) WHERE m.title = 'Movie Title' RETURN avg(r.rating) AS userRating
2. Question: Get movies for a genre?
   Cypher: MATCH ((m:Movie)-[:IN_GENRE]->(g:Genre) WHERE g.name = 'Genre Name' RETURN m.title AS movieTitle

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
# end::cypher_template_example_genre[]

# tag::cypher_prompt[]
cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], 
    template=cypher_template
)
# end::cypher_prompt[]

# tag::cypher_qa[]
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model, 
    cypher_llm=cypher_model,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True,
    verbose=True,
)
# end::cypher_qa[]

question = "How many Sci-Fi movies has Tom Hanks acted in?"
response = cypher_qa.invoke({"query": question})
print(response["result"])




# tag::examples_the[]
# Who directed the movie The Matrix?
# end::examples_the[]

# tag::examples_rating[]
# What is the highest rating for Goodfellas?
# What is the average user rating for the movie Toy Story?
# end::examples_rating[]

# tag::examples_genre[]
# What is the highest user rated movie in the Horror genre?
# end::examples_genre[]