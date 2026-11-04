import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# API keys (OPENAI_API_KEY, LANGCHAIN_API_KEY) are loaded from .env
load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "simple-chatbot-eval2"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


BLOCKED_TOPICS = ["weapons", "self-harm", "illegal drugs", "hacking", "violence"]

def input_guardrail(user_input: str) -> tuple[bool, str]:
    text = user_input.lower()
    for topic in BLOCKED_TOPICS:
        if topic in text:
            return False, f"Request blocked: relates to restricted topic '{topic}'."
    if len(user_input) > 2000:
        return False, "Request blocked: input too long."
    return True, ""

# LLM-based moderation guardrail
moderation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a safety classifier. Respond ONLY with 'SAFE' or 'UNSAFE'. "
     "Mark UNSAFE if the text contains hate, harassment, sexual content, "
     "self-harm, or instructions for illegal/dangerous acts."),
    ("user", "{text}")
])
moderator = moderation_prompt | llm | StrOutputParser()

def llm_guardrail(text: str) -> bool:
    return moderator.invoke({"text": text}).strip().upper().startswith("SAFE")

def output_guardrail(response: str) -> str:
    if not llm_guardrail(response):
        return "I'm sorry, I can't share that response."
    return response


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, polite assistant. Keep answers concise."),
    ("user", "{question}")
])
chatbot_chain = chat_prompt | llm | StrOutputParser()

@traceable(name="guarded_chatbot")
def chatbot(question: str) -> str:
    ok, msg = input_guardrail(question)
    if not ok:
        return msg
    if not llm_guardrail(question):
        return "I'm sorry, I can't help with that request."
    raw = chatbot_chain.invoke({"question": question})
    return output_guardrail(raw)


client = Client()

dataset_name = "chatbot-eval-dataset2"
examples = [
    {"inputs": {"question": "Who won IPL 2026?"},
     "outputs": {"answer": "RCB"}},
    {"inputs": {"question": "Explain photosynthesis in one line."},
     "outputs": {"answer": "Plants convert sunlight, water, and CO2 into glucose and oxygen."}},
    {"inputs": {"question": "What is 12 * 8?"},
     "outputs": {"answer": "96"}},
]

if not client.has_dataset(dataset_name=dataset_name):
    ds = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=ds.id,
    )

def target(inputs: dict) -> dict:
    return {"answer": chatbot(inputs["question"])}


def correctness(outputs: dict, reference_outputs: dict) -> dict:
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "Return only a number 1 (correct) or 0 (incorrect). "
                   "Compare the prediction to the reference for factual correctness."),
        ("user", "Prediction: {pred}\nReference: {ref}")
    ])
    judge = judge_prompt | llm | StrOutputParser()
    score = judge.invoke({"pred": outputs["answer"],
                          "ref": reference_outputs["answer"]}).strip()
    return {"key": "correctness", "score": 1 if score.startswith("1") else 0}

# Built-in evaluators
qa_evaluator = LangChainStringEvaluator(
    "qa",
    config={"llm": llm},
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"],
    },
)

results = evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, qa_evaluator],
    experiment_prefix="guarded-chatbot",
)

print(results)