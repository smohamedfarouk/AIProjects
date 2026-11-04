import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()  
MODEL = "openai:gpt-4o-mini"

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math_server.py")


async def main():
    client = MultiServerMCPClient(
        {
            "beginner_tools": {
                "command": sys.executable,   
                "args": [SERVER_PATH],       
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    print("Tools loaded from MCP:", [t.name for t in tools])

   
    llm = init_chat_model(MODEL)
    llm_with_tools = llm.bind_tools(tools)


    async def agent_node(state: MessagesState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

   
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)        
    graph.add_node("tools", ToolNode(tools))   

    graph.add_edge(START, "agent")
    
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")           

    app = graph.compile()


    question = "What is (3 + 5) multiplied by 12? Also, what time is it right now?"
    result = await app.ainvoke({"messages": [{"role": "user", "content": question}]})

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
