"""
Streamlit frontend for the LangChain AI Chat backend.

Runs standalone — no FastAPI layer. Talks directly to LangChain (via the
`chain` in llm_and_chains.py) and SQLite (via db_opr.py).

Run:
    streamlit run streamlit_app.py

Requires a GROQ_API_KEY (and optionally GOOGLE_API_KEY) available as an
environment variable / Streamlit secret. See llm_and_chains.py.
"""

import time

import streamlit as st

from db_opr import (
    init_db,
    save_chat,
    save_message,
    load_history,
    evaluate,
    new_session_id,
    get_sessions,
    get_session_history,
    delete_session_history,
)
from llm_and_chains import model, prompt as chat_prompt
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="💬",
    layout="wide"
    )


# =========================================================
# DATABASE INIT (runs once per app process)
# =========================================================

@st.cache_resource
def _init_db_once():
    init_db()
    return True


_init_db_once()


# =========================================================
# SESSION STATE
# =========================================================

def init_state():
    defaults = {
        "current_session": None,
        "messages": [],   # list of {"role": ..., "content": ...} currently displayed
        "sessions": [],   # list of {"session_id": ..., "title": ...} from get_sessions()
        "groq_api_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def get_chain():
    """Build the chat chain using the user-supplied Groq API key."""
    llm = ChatGroq(
        model=model,
        temperature=0.7,
        max_retries=3,
        api_key=st.session_state.groq_api_key,
    )
    return chat_prompt | llm | StrOutputParser()


# =========================================================
# BACKEND HELPERS (formerly HTTP calls, now direct calls)
# =========================================================

def fetch_sessions():
    """Refresh the sidebar chat list."""
    try:
        rows = get_sessions()
        st.session_state.sessions = [
            {"session_id": row["session_id"], "title": row["title"]} for row in rows
        ]
    except Exception as e:
        st.sidebar.error(f"Failed to load sessions: {e}")


def fetch_history(session_id):
    """Load messages for a past chat."""
    try:
        rows = get_session_history(session_id)
        st.session_state.messages = [
            {"role": row["role"], "content": row["content"]} for row in rows
        ]
        st.session_state.current_session = session_id
    except Exception as e:
        st.sidebar.error(f"Failed to load history: {e}")


def create_new_chat():
    # Start a fresh session.
    try:
        sid = new_session_id()
        st.session_state.current_session = sid
        st.session_state.messages = []
        fetch_sessions()
    except Exception as e:
        st.sidebar.error(f"Failed to create new chat: {e}")


def delete_chat(session_id):
    """Remove a chat and its logs."""
    try:
        delete_session_history(session_id)
        if st.session_state.current_session == session_id:
            st.session_state.current_session = None
            st.session_state.messages = []
        fetch_sessions()
    except Exception as e:
        st.sidebar.error(f"Failed to delete chat: {e}")


def stream_chat_response(session_id, prompt):
    """
    Stream the LLM reply token-by-token (mirrors the old /chat endpoint's
    generator), then save the exchange, run eval, and store token/score info
    on st.session_state for the caller to use after streaming completes.
    """
    history = load_history(session_id)
    start = time.time()
    reply = ""
    chain = get_chain()

    for chunk in chain.stream({"input": prompt, "history": history}):
        reply += chunk
        yield chunk

    duration = (time.time() - start) * 1000

    save_message(session_id, "user", prompt)
    save_message(session_id, "assistant", reply)

    savechat = {
        "user_message": prompt,
        "ai_response": reply,
        "model": model,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    chat_id = save_chat(session_id, savechat)
    overall_score = evaluate(chat_id, prompt, reply, session_id)

    st.session_state["_last_chat_id"] = chat_id
    st.session_state["_last_score"] = overall_score
    st.session_state["_last_duration_ms"] = round(duration, 2)


# =========================================================
# SIDEBAR — NEW CHAT
# =========================================================

with st.sidebar:
    st.header("Configuration")

    groq_api_key_input = st.text_input(
        "Groq API Key", value=st.session_state.groq_api_key, type="password"
    )
    st.session_state.groq_api_key = groq_api_key_input
    st.caption(f"Model: `{model}`")

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()

    st.divider()

    # =====================================================
    # SIDEBAR — CHATS LIST
    # =====================================================
    st.subheader("Chats")

    if not st.session_state.sessions:
        fetch_sessions()

    if not st.session_state.sessions:
        st.caption("No chats yet.")
    else:
        for chat in st.session_state.sessions:
            sid = chat.get("session_id")
            title = chat.get("title") or f"Session {sid}"
            if len(title) > 30:
                title = title[:30] + "..."

            col_title, col_delete = st.columns([5, 1])
            with col_title:
                is_active = st.session_state.current_session == sid
                if st.button(
                    ("📌 " if is_active else "") + title,
                    key=f"chat_{sid}",
                    use_container_width=True,
                ):
                    fetch_history(sid)
                    st.rerun()
            with col_delete:
                if st.button("🗑️", key=f"delete_{sid}"):
                    delete_chat(sid)
                    st.rerun()


# =========================================================
# MAIN PAGE — TITLE
# =========================================================

st.markdown("<h1 style='text-align: center;'>AI Chat Assistant</h1>", unsafe_allow_html=True)
st.divider()

if not st.session_state.groq_api_key:
    st.warning("⚠️ Please enter your Groq API key!")
    st.stop()

if st.session_state.current_session is None:
    st.info("Click **New Chat** in the sidebar to start a conversation, or pick one from the list.")
    st.stop()


# =========================================================
# MAIN PAGE — CONVERSATION DISPLAY
# =========================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================================
# MAIN PAGE — CHAT INPUT / CHAT FLOW
# =========================================================

prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""
        try:
            for chunk in stream_chat_response(st.session_state.current_session, prompt):
                full_reply += chunk
                placeholder.markdown(full_reply + "▌")
            placeholder.markdown(full_reply)
        except Exception as e:
            full_reply = f"⚠️ Error generating response: {e}"
            placeholder.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})

    # Refresh the sidebar so a brand-new chat picks up its title
    # (first user message) once it's been saved.
    fetch_sessions()
