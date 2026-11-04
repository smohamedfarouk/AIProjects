"""
Streamlit UI for the Text-to-SQL Agent (with guardrail + LangSmith visibility)
"""
import ast
import streamlit as st
import pandas as pd
from backend import TextToSQLAgent

# ---------- Page Config ----------
st.set_page_config(page_title="Text-to-SQL Agent", page_icon="🗄️", layout="wide")

# ---------- Custom Styling ----------
st.markdown("""
<style>
    .stApp { background-color: #fafafa; }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;
    }
    .guard-ok   { color: #16a34a; font-weight: 600; }
    .guard-warn { color: #d97706; font-weight: 600; }
    .guard-bad  { color: #dc2626; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="main-header">
    <h1>🗄️ Text-to-SQL Agent</h1>
    <p>Ask questions in plain English. Guardrails + LangSmith monitoring built in.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Initialize Agent ----------
@st.cache_resource
def load_agent():
    return TextToSQLAgent()

try:
    agent = load_agent()
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")
    st.stop()

# ---------- Chat History ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🔌 Connection Status")
    st.success("✅ Connected to MySQL")

    st.markdown("### 🛡️ Active Guardrails")
    st.markdown("""
- Input screening (injection / destructive intent)
- PII detection & masking (in + out)
- Read-only SQL enforcement
- Sensitive-column blocking
- Hallucination check on answers
""")

    st.markdown("### 📊 Monitoring")
    st.caption("Traces & evals stream to LangSmith project `text-to-sql-agent`.")

    st.markdown("### 📋 Available Tables")
    for table in agent.get_table_names():
        st.markdown(f"- `{table}`")

    with st.expander("📐 View Schema"):
        st.code(agent.get_schema_info(), language="sql")

    st.markdown("### 💡 Sample Questions")
    samples = [
        "How many students are enrolled in GenAI?",
        "Which student got the highest placement package?",
        "List all students from Bangalore",
        "What is the average package by course?",
        "Show total revenue from each course",
    ]
    for q in samples:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Helper: parse raw result into DataFrame ----------
def try_parse_to_df(raw_result: str):
    try:
        data = ast.literal_eval(raw_result)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple):
            return pd.DataFrame(data)
    except Exception:
        pass
    return None

# ---------- Helper: render guardrail badges ----------
def render_guardrails(content: dict):
    cols = st.columns(3)
    blocked = content.get("blocked", False)
    pii = content.get("pii_flags", [])
    halluc = content.get("hallucination_flag", False)

    with cols[0]:
        if blocked:
            st.markdown("<span class='guard-bad'>⛔ Input Blocked</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='guard-ok'>✅ Input Passed</span>", unsafe_allow_html=True)
    with cols[1]:
        if pii:
            st.markdown(f"<span class='guard-warn'>🔒 PII Masked: {', '.join(pii)}</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<span class='guard-ok'>🔒 No PII</span>", unsafe_allow_html=True)
    with cols[2]:
        if halluc:
            st.markdown("<span class='guard-warn'>⚠️ Possible Hallucination</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<span class='guard-ok'>🎯 Grounded</span>", unsafe_allow_html=True)

# ---------- Render one assistant response ----------
def render_assistant(content: dict):
    st.markdown(content.get("answer", ""))
    render_guardrails(content)

    if content.get("sql_query"):
        with st.expander("🔍 View SQL Query"):
            st.code(content["sql_query"], language="sql")

    raw = content.get("raw_result", "")
    df = try_parse_to_df(raw)
    if df is not None and not df.empty:
        with st.expander("📊 View Result Data"):
            st.dataframe(df, use_container_width=True)
    elif raw:
        with st.expander("📄 Raw Result"):
            st.text(raw)

# ---------- Display Chat History ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg["content"])

# ---------- Handle Input ----------
question = None
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

user_input = st.chat_input("Ask a question about your database...")
if user_input:
    question = user_input

# ---------- Process Question ----------
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Running guardrails, generating SQL, querying..."):
            try:
                result = agent.query(question)
                render_assistant(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"answer": error_msg, "sql_query": "", "raw_result": "",
                                "pii_flags": [], "hallucination_flag": False, "blocked": True},
                })

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>"
    "Built with LangGraph + Streamlit + MySQL • Guardrails + LangSmith Evals"
    "</p>",
    unsafe_allow_html=True
)