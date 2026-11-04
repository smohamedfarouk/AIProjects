import os
from dotenv import load_dotenv, find_dotenv
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langsmith import traceable


# API keys (OPENAI_API_KEY, LANGCHAIN_API_KEY) are loaded from .env
load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "hierarchical-agents"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


BLOCKED = ["weapons", "self-harm", "illegal drugs", "hacking", "violence"]

mod_prompt = ChatPromptTemplate.from_messages([
    ("system", "Safety classifier. Reply ONLY 'SAFE' or 'UNSAFE'. UNSAFE if hate, "
               "harassment, sexual, self-harm, or illegal/dangerous instructions."),
    ("user", "{text}")
])
moderator = mod_prompt | llm | StrOutputParser()

def is_safe(text: str) -> bool:
    if any(b in text.lower() for b in BLOCKED):
        return False
    return moderator.invoke({"text": text}).strip().upper().startswith("SAFE")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str          
    task: str          
    blocked: bool

WORKERS = ["researcher", "writer", "coder"]


def make_worker(role: str, system: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "{task}")
    ])
    chain = prompt | llm | StrOutputParser()

    @traceable(name=f"worker_{role}")
    def worker(state: AgentState) -> dict:
        result = chain.invoke({"task": state["task"]})
        # output guardrail per worker
        if not is_safe(result):
            result = f"[{role}] Output blocked by guardrail."
        return {"messages": [AIMessage(content=f"[{role}]\n{result}")]}
    return worker

researcher = make_worker(
    "researcher",
    "You are a researcher. Gather and summarize key facts for the task concisely."
)
writer = make_worker(
    "writer",
    "You are a writer. Produce clear, polished prose for the task."
)
coder = make_worker(
    "coder",
    "You are a coder. Write clean, correct code for the task with brief comments."
)


sup_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a supervisor managing workers: {workers}. "
     "Given the task and conversation so far, decide who acts next. "
     "When the task is fully complete, respond with 'FINISH'. "
     "Reply with ONLY one word: a worker name or FINISH."),
    ("user", "Task: {task}\n\nConversation so far:\n{history}")
])
supervisor_chain = sup_prompt | llm | StrOutputParser()

@traceable(name="supervisor")
def supervisor(state: AgentState) -> dict:
    history = "\n".join(m.content for m in state["messages"]) or "(none)"
    decision = supervisor_chain.invoke({
        "workers": ", ".join(WORKERS),
        "task": state["task"],
        "history": history,
    }).strip().lower()

    nxt = "FINISH"
    for w in WORKERS:
        if w in decision:
            nxt = w
            break
    if "finish" in decision:
        nxt = "FINISH"
    return {"next": nxt}


@traceable(name="input_guardrail")
def guard_entry(state: AgentState) -> dict:
    task = state["task"]
    if not is_safe(task):
        return {
            "blocked": True,
            "messages": [AIMessage(content="Request blocked by guardrail.")],
            "next": "FINISH",
        }
    return {"blocked": False}

def route_after_guard(state: AgentState) -> Literal["supervisor", "END"]:
    return "END" if state.get("blocked") else "supervisor"

def route_supervisor(state: AgentState) -> str:
    return END if state["next"] == "FINISH" else state["next"]


graph = StateGraph(AgentState)
graph.add_node("guard", guard_entry)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("coder", coder)

graph.add_edge(START, "guard")
graph.add_conditional_edges("guard", route_after_guard,
                            {"supervisor": "supervisor", "END": END})
graph.add_conditional_edges("supervisor", route_supervisor,
                            {"researcher": "researcher", "writer": "writer",
                             "coder": "coder", END: END})
# workers report back to supervisor (hierarchical loop)
for w in WORKERS:
    graph.add_edge(w, "supervisor")

app = graph.compile()


@traceable(name="hierarchical_agent_run")
def run(task: str):
    state = {"messages": [HumanMessage(content=task)], "task": task,
             "next": "", "blocked": False}
    final = app.invoke(state, config={"recursion_limit": 20})
    for m in final["messages"]:
        print(f"{m.type.upper()}: {m.content}\n")
    return final

if __name__ == "__main__":
    run("Research the benefits of intermittent fasting, then write a short summary "
        "and a Python function that calculates an eating-window schedule.")