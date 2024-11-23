# First we initialize the model we want to use.
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

model = ChatOllama(model="llama3.2:latest", temperature=0, base_url="https://ollama.bealink.id")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that  help farmers treat their plants, you must  use tools output to answer."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def find_disease_info_function(input: str) -> str:
    """Find data about plant  disease"""
    return f"for {input} disease you must pesticide name oxidacomfature brand "


tools = [find_disease_info_function]

query = "blight"

agent = create_tool_calling_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

for step in agent_executor.stream({"input": query}):
    print(step)