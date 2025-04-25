import streamlit as st
import pira_agent as pa            # your logic module

st.set_page_config("Jiva-ai", "🤖", layout="centered") 
st.markdown(""" <style> /* only inside chat bubbles produced by st.chat_message */ .chat-message p {margin:0 0 1em; line-height:1.55;} .chat-message strong {color:#ffd166;} /* nice amber for numbers */ .chat-message em {font-style:normal;} /* disable italics app-wide */ </style> """, unsafe_allow_html=True)

# ── initialise session state ─────────────────────────────────────────
if "history"   not in st.session_state:
    st.session_state.history   = [
        {"role":"assistant",
         "content":"Hi 👋 I’m your Perishable-Inventory Routing Assistant."}]
if "last_plan" not in st.session_state: st.session_state.last_plan = None
if "last_user" not in st.session_state: st.session_state.last_user = ""

# ── helper to render & store a bubble ───────────────────────────────
def add(role:str, txt:str):
    st.session_state.history.append({"role":role,"content":txt})
    with st.chat_message(role): st.markdown(txt, unsafe_allow_html=True)

# ── render chat so far ──────────────────────────────────────────────
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

# ── read user input (returns str or None) ───────────────────────────
raw = st.chat_input("Type your supply-chain question…", key="chat_box")

if raw:                                        # user pressed Enter
    # 1️⃣ clear the input *before* next rerun
    st.session_state.pop("chat_box", None)

    user_msg = raw.strip()
    # 2️⃣ skip exact consecutive duplicates
    if user_msg == st.session_state.last_user:
        st.stop()

    add("user", user_msg)

    # placeholder bubble while thinking
    with st.chat_message("assistant"):
        wait = st.empty()
        wait.markdown("⏳ Crunching…")

    # ── 3️⃣ orchestrate ────────────────────────────────────────────
    route = pa.orchestrator_llm(user_msg)      # ← LLM router
    if route["kind"] == "chat":
        # fallback to regex router if LLM didn’t find knobs
        route = pa.orchestrator_regex(user_msg) if hasattr(pa,"orchestrator_regex") else route

    # ── 4️⃣ answer branches ───────────────────────────────────────
    if route["kind"] == "chat":
        answer = pa.llm_smalltalk(route.get("text", user_msg))

    else:  # scenario
        try:
            p = pa.safeguard(route["params"])
            if (not p.get("demand_multiplier") and
                not p.get("supplier_cap_delta") and
                p.get("outbound_mode","auto") == "auto"):
                # follow-up question on existing plan
                if st.session_state.last_plan:
                    answer = pa.llm(st.session_state.last_plan, user_msg)
                else:
                    answer = pa.llm_smalltalk(user_msg)
            else:
                plan   = pa.sagebard(p)
                st.session_state.last_plan = plan
                answer = pa.llm(plan, user_msg)
        except Exception as e:
            answer = f"❌ **Error**: {e}"

    # 5️⃣ display final assistant message
    # existing call
                    # ← new
    wait.markdown(answer, unsafe_allow_html=True)
    st.session_state.history.append({"role":"assistant","content":answer})
    st.session_state.last_user = user_msg     # remember last
