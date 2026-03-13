import streamlit as st
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

/* Background */
.stApp {
    background-color: #0e0e10;
    color: #e8e4d9;
}

/* Header */
.research-header {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #e8e4d9;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.research-sub {
    font-size: 0.8rem;
    color: #5a5a62;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Input */
.stTextInput > div > div > input {
    background: #18181c !important;
    border: 1px solid #2e2e36 !important;
    border-radius: 4px !important;
    color: #e8e4d9 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #c8b97a !important;
    box-shadow: 0 0 0 1px #c8b97a22 !important;
}
.stTextInput label {
    color: #5a5a62 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* Button */
.stButton > button {
    background: #c8b97a !important;
    color: #0e0e10 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Cards */
.result-card {
    background: #18181c;
    border: 1px solid #2e2e36;
    border-radius: 6px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c8b97a;
    margin-bottom: 0.5rem;
}
.card-value {
    font-size: 0.95rem;
    color: #e8e4d9;
    line-height: 1.6;
}
.tag {
    display: inline-block;
    background: #24242a;
    border: 1px solid #2e2e36;
    border-radius: 3px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    color: #9a9aaa;
    margin: 0.2rem 0.2rem 0.2rem 0;
}
.source-link {
    color: #7a9ec8;
    font-size: 0.8rem;
    word-break: break-all;
    display: block;
    margin: 0.3rem 0;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #2e2e36;
    margin: 2rem 0;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #c8b97a !important;
}

/* Status */
.stAlert {
    background: #18181c !important;
    border: 1px solid #2e2e36 !important;
    color: #e8e4d9 !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="research-header">Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="research-sub">Powered by Gemini · DuckDuckGo · Wikipedia</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Agent setup ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    class AgentResponse(BaseModel):
        topic: str
        summary: str
        sources: list[str]
        tools_used: list[str]

    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=google_api_key)

    parser = PydanticOutputParser(pydantic_object=AgentResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools.
Wrap the output in this format and provide no other text.

{format_instructions}
"""),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    tools = [search_tool, wiki_tool, save_tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return executor, parser, AgentResponse

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("RESEARCH QUERY", placeholder="e.g. What caused the 2008 financial crisis?", label_visibility="visible")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Research →")

# ── Run agent ─────────────────────────────────────────────────────────────────
if run and query.strip():
    try:
        executor, parser, AgentResponse = get_agent()
        with st.spinner("Researching..."):
            raw = executor.invoke({"query": query, "chat_history": []})
            result = parser.parse(raw.get("output"))
            st.session_state.history.insert(0, result)
    except Exception as e:
        st.error(f"Error: {e}")

elif run and not query.strip():
    st.warning("Please enter a query.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    for i, r in enumerate(st.session_state.history):
        if i == 0:
            st.markdown("---")
            st.markdown(f'<div class="result-card"><div class="card-label">Topic</div><div class="card-value" style="font-family:\'DM Serif Display\',serif;font-size:1.4rem;">{r.topic}</div></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="result-card"><div class="card-label">Summary</div><div class="card-value">{r.summary}</div></div>', unsafe_allow_html=True)

            sources_html = "".join(f'<span class="source-link">↗ {s}</span>' for s in r.sources)
            st.markdown(f'<div class="result-card"><div class="card-label">Sources</div>{sources_html}</div>', unsafe_allow_html=True)

            tools_html = "".join(f'<span class="tag">{t}</span>' for t in r.tools_used)
            st.markdown(f'<div class="result-card"><div class="card-label">Tools Used</div><div style="margin-top:0.3rem">{tools_html}</div></div>', unsafe_allow_html=True)

        else:
            with st.expander(f"#{i+1} — {r.topic}", expanded=False):
                st.markdown(f'<div class="card-label">Summary</div><div class="card-value">{r.summary}</div>', unsafe_allow_html=True)
                for s in r.sources:
                    st.markdown(f'<span class="source-link">↗ {s}</span>', unsafe_allow_html=True)
                tools_html = "".join(f'<span class="tag">{t}</span>' for t in r.tools_used)
                st.markdown(f'<div class="card-label" style="margin-top:0.75rem">Tools Used</div>{tools_html}', unsafe_allow_html=True)

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
        