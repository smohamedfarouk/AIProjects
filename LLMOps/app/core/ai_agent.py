from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from app.common.customexceptions import CustomException
from app.config.settings import settings


def get_ai_message(model_id, query, allow_search, system_prompt):
    try:
        llm = ChatGroq(model=model_id, api_key=settings.GROQ_API_KEY)
        tools = [TavilySearch(max_results=2)] if allow_search else []
        agent = create_agent(llm, tools=tools, system_prompt=system_prompt or None)
        messages = [("user", message) for message in query]
        response = agent.invoke({"messages": messages})
        for message in reversed(response["messages"]):
            if isinstance(message, AIMessage):
                return message.text
        raise ValueError("Agent returned no AI response")
    except Exception as error:
        raise CustomException("Failed to get AI response") from error
