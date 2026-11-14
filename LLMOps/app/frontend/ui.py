import streamlit as st
import requests

from app.config.settings import settings
st.title("FAROUK'S MULTI AI AGENT LLMops")
st.set_page_config(page_title="MULTI AI AGENT", layout="centered")

prompt_column, button_column = st.columns([3, 1], vertical_alignment="bottom")
system_prompt = prompt_column.text_area("Define your AI Agent", key="system_prompt")
model_name = button_column.selectbox("Select model", settings.ALLOWED_MODELS_NAMES, key="model_name")
allow_search = button_column.checkbox("Allow web search", key="allow_search")
query_column, ask_column = st.columns([3, 1], vertical_alignment="bottom")
query = query_column.text_area("Enter your query", key="user_message")
ask_agent = ask_column.button("Ask Agent", width="stretch")

API_URL = f"http://{settings.API_HOST}:{settings.API_PORT}/chat"

if ask_agent and query.strip():
    payload = {
        "model_name": model_name,
        "messages": [query],
        "allow_search": allow_search,
        "system_prompt": system_prompt,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        st.text_area("AI Response", value=response.json(), key="ai_response", height=900)
    except requests.RequestException as error:
        st.error(f"Unable to get AI response: {error}") 