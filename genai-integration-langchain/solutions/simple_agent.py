from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict

# tag::llm_prompt[]
# Initialize the LLM
model = init_chat_model("gpt-4o", model_provider="openai")

# Create a prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)
# end::llm_prompt[]

# tag::application_state[]
# Define state for application
class State(TypedDict):
    question: str
    context: dict
    answer: str
# end::application_state[]

# tag::application_functions[]
# Define functions for each step in the application

# Retrieve context 
def retrieve(state: State):
    context = {
        "London": "Cloudy, sunny skies later in the day.",
        "San Francisco": "Sunny skies, raining overnight.",
    }
    return {"context": context}

# Generate the answer based on the question and context
def generate(state: State):
    messages = prompt.invoke({"question": state["question"], "context": state["context"]})
    response = model.invoke(messages)
    return {"answer": response.content}
# end::application_functions[]

# tag::application_workflow[]
# Define application steps
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
app = workflow.compile()
# end::application_workflow[]

# tag::invoke[]
# Run the application
question = "What is the weather in San Francisco?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
# end::invoke[]