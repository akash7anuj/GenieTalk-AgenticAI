import os
import io
from typing import List, Dict, Any

import streamlit as st
import google.generativeai as genai

# =========================
# BASIC CONFIG
# =========================
st.set_page_config(
    page_title="GenieTalk Agentic AI",
    page_icon="🧞‍♂️",
    layout="wide"
)

# =========================
# HELPER: LOAD GEMINI
# =========================
def init_gemini(api_key: str, model_name: str = "models/gemini-flash-latest"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model

# =========================
# FILE / DOC HANDLING
# =========================
def read_txt_file(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def read_pdf_file(file) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return "PyPDF2 is not installed. Please install it with `pip install PyPDF2`."

    reader = PdfReader(file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def get_uploaded_text(files: List[Any]) -> str:
    """
    Merge all uploaded files (PDF + TXT) into single text context.
    """
    if not files:
        return ""

    full_text = []
    for f in files:
        if f.name.lower().endswith(".txt"):
            full_text.append(read_txt_file(f))
        elif f.name.lower().endswith(".pdf"):
            full_text.append(read_pdf_file(f))
        else:
            full_text.append(f"Unsupported file type: {f.name}")
    return "\n\n".join(full_text)

# =========================
# AGENT "TOOLS" (SKILLS)
# =========================
def tool_general_chat(model, user_input: str, language: str, role: str, history: List[Dict]) -> str:
    # ✅ Convert system prompt into a USER message (Gemini-compatible)
    system_prompt = f"""
You are GenieTalk, an AI agentic assistant.

Role: {role}

You must:
- Be helpful and concise.
- Adapt your tone to the role (e.g., more emotional for Emotional Support, technical for Coding Help).
- Always respond in this language: {language}.
"""

    contents = []

    # ✅ System prompt goes as USER
    contents.append({
        "role": "user",
        "parts": [system_prompt]
    })

    # ✅ Previous chat history
    for msg in history:
        contents.append({
            "role": "user",
            "parts": [msg["user"]]
        })
        contents.append({
            "role": "model",
            "parts": [msg["assistant"]]
        })

    # ✅ Current user input
    contents.append({
        "role": "user",
        "parts": [user_input]
    })

    # ✅ Generate response
    response = model.generate_content(contents)
    return response.text.strip()


def tool_document_qa(model, question: str, doc_text: str, language: str) -> str:
    if not doc_text.strip():
        return f"(I could not find any document text. Please upload a PDF or TXT first.)"

    prompt = f"""
You are a document QA assistant.

You receive:
- A question from the user
- Full extracted text from one or more documents

1. First, briefly state what you understood about the question.
2. Then answer using ONLY the document text when possible.
3. If something is not in the document, clearly say it.

Answer in this language: {language}.

User question:
{question}

Document text:
\"\"\"{doc_text[:25000]}\"\"\"  # (truncated if very long)
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def tool_translate(model, text: str, target_language: str) -> str:
    prompt = f"""
You are a professional translator.

Translate the following text into: {target_language}.

Keep the original meaning, tone, and style.

Text:
\"\"\"{text}\"\"\"
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def tool_resume_review(model, resume_text: str, language: str) -> str:
    if not resume_text.strip():
        return "Please upload your resume as PDF/TXT or paste it so I can review it."

    prompt = f"""
You are a resume and career advisor.

You receive a resume text and must:
1. Summarize the candidate's profile.
2. Point out 5–10 very specific improvements (content + formatting).
3. Suggest 3 tailored role titles the candidate can target.
4. Suggest 3–5 strong bullet points they can add.

Answer in this language: {language}.

Resume text:
\"\"\"{resume_text[:20000]}\"\"\"
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def tool_coding_help(model, question: str, language: str) -> str:
    prompt = f"""
You are a senior software engineer and coding mentor.

User question:
{question}

You must:
- Explain step by step, but not too long.
- Show minimal but correct code examples.
- Add short comments.
- Answer in language: {language}.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def tool_emotional_support(model, message: str, language: str) -> str:
    prompt = f"""
You are a supportive, empathetic friend.

User message:
{message}

You must:
- Validate their feelings.
- Avoid giving medical or clinical diagnosis.
- Offer simple, kind suggestions and coping strategies.
- Encourage them to reach out to trusted people or professionals if needed.
- Answer in language: {language}.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# =========================
# AGENTIC MODE: PLAN + ACT
# =========================
def agentic_plan_and_execute(
    model,
    goal: str,
    role: str,
    language: str,
    doc_text: str,
    chat_history: List[Dict]
) -> Dict[str, Any]:
    """
    Agentic behavior:
    1. Understand user's high-level goal.
    2. Create a mini-plan (3–6 steps).
    3. Decide which "skills/tools" to use per step.
    4. Execute (within a single model call, but logically multi-step).
    """

    available_tools_description = """
You have these internal skills/tools:

1. general_chat: For broad reasoning, explanation, brainstorming.
2. coding_help: For code, debugging, writing functions, etc.
3. resume_review: For CV/resume critique and job guidance.
4. emotional_support: For empathy and motivation (not medical).
5. document_qa: For answering questions about uploaded PDFs/TXTs.
6. translate: For translating text into user's target language.

You cannot actually call external APIs here; you must "simulate" tool use by clearly reasoning.
"""

    history_text = ""
    if chat_history:
        history_text = "\nPrevious conversation:\n"
        for turn in chat_history[-6:]:  # last few turns
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    doc_hint = ""
    if doc_text.strip():
        doc_hint = f"\n\nThe user also provided document text. You can treat it as context when needed:\n\"\"\"{doc_text[:8000]}\"\"\""

    prompt = f"""
You are GenieTalk, an AGENTIC AI assistant.

User goal:
\"\"\"{goal}\"\"\"

Role: {role}
Target answer language: {language}

{available_tools_description}

Your job:
1. Briefly restate the goal.
2. Create a numbered plan (3–6 steps).
3. For each step, say which tool/skill you are conceptually using (from the list).
4. Then actually do the reasoning/work for those steps.
5. Finally, give a clean FINAL ANSWER section that the user can read directly.

Use clear headings:
- Goal
- Plan
- Step-by-step Thinking
- Final Answer

Be practical and focused. If documents are relevant, you may use them conceptually.

{history_text}
{doc_hint}
"""
    response = model.generate_content(prompt)
    text = response.text.strip()

    return {
        "agentic_explanation": text,
    }

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {user, assistant}
if "agent_runs" not in st.session_state:
    st.session_state.agent_runs = []  # store agentic plans/executions

# =========================
# SIDEBAR UI
# =========================
with st.sidebar:
    st.title("🧞‍♂️ GenieTalk Agentic AI")

    st.markdown("### 🔑 API Key")
    api_key = st.text_input(
        "Enter your Gemini API key",
        type="password",
        help="Get it from Google AI Studio."
    )

    st.markdown("### 🎭 Mode")
    main_mode = st.radio(
        "Select interaction mode",
        ["Chat", "Agentic Task / Goal"],
        help="Chat = normal assistant; Agentic Task = goal-based planning agent."
    )

    # st.markdown("### 🧠 Role / Persona")
    role = st.selectbox(
        "Choose agent role",
        [
            "General Assistant",
            "Coding Help",
            "Resume Review",
            "Emotional Support",
            "Document QA",
            "Translator"
        ]
    )

    st.markdown("### 🌐 Reply Language")
    reply_language = st.selectbox(
        "Language for replies",
        ["English", "Hindi", "Bengali", "Spanish", "French", "German", "Tamil", "Telugu", "Other"],
        index=0
    )
    if reply_language == "Other":
        reply_language = st.text_input("Type target language name", value="English")

    st.markdown("### 📂 Upload Documents (optional)")
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT for document QA or resume review",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    doc_text_context = get_uploaded_text(uploaded_files) if uploaded_files else ""

    st.markdown("### 💾 Chat Controls")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.agent_runs = []
        st.success("Chat history cleared!")

    st.markdown("---")
    st.markdown("### 📤 Export Chat")
    if st.button("Download Chat as .txt"):
        all_text = []
        for m in st.session_state.messages:
            all_text.append(f"User: {m['user']}\nAssistant: {m['assistant']}\n")
        export_str = "\n".join(all_text)
        st.download_button(
            "Click to download",
            data=export_str,
            file_name="genieTalk_chatlog.txt",
            mime="text/plain"
        )

    st.markdown("---")
    st.markdown("### 🎙️ Voice (optional)")
    st.info(
        "You can plug your previous voice input / text-to-speech code here.\n"
        "For example using `streamlit-mic-recorder` and `gTTS`.\n"
        "This section is just a placeholder in this version."
    )

# =========================
# MAIN AREA
# =========================
st.title("🧞‍♂️ GenieTalk — Agentic AI Chatbot")

st.markdown(
    """
**What makes this Agentic?**

- Understands your **goals**, not just single questions  
- Creates a **plan with steps**  
- Selects internal **skills/tools** (coding help, doc QA, translation, resume review, emotional support)  
- Uses your **documents** as context for decisions and answers  
"""
)

# Display previous chat
for m in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(m["user"])
    with st.chat_message("assistant"):
        st.markdown(m["assistant"])

# =========================
# USER INPUT
# =========================
user_input = st.chat_input("Type your message or describe a goal...")

if user_input and not api_key:
    st.warning("Please enter your Gemini API key in the sidebar first.")
    st.stop()

if user_input and api_key:
    model = init_gemini(api_key)

    if main_mode == "Chat":
        # ===== NORMAL CHAT MODE =====
        with st.chat_message("user"):
            st.markdown(user_input)

        # Route based on role
        if role == "General Assistant":
            assistant_reply = tool_general_chat(model, user_input, reply_language, role, st.session_state.messages)
        elif role == "Coding Help":
            assistant_reply = tool_coding_help(model, user_input, reply_language)
        elif role == "Resume Review":
            # If there is a document, use it as resume text
            resume_text = doc_text_context if doc_text_context else user_input
            assistant_reply = tool_resume_review(model, resume_text, reply_language)
        elif role == "Emotional Support":
            assistant_reply = tool_emotional_support(model, user_input, reply_language)
        elif role == "Document QA":
            assistant_reply = tool_document_qa(model, user_input, doc_text_context, reply_language)
        elif role == "Translator":
            assistant_reply = tool_translate(model, user_input, reply_language)
        else:
            assistant_reply = tool_general_chat(model, user_input, reply_language, role, st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

        st.session_state.messages.append(
            {"user": user_input, "assistant": assistant_reply}
        )

    else:
        # ===== AGENTIC TASK / GOAL MODE =====
        with st.chat_message("user"):
            st.markdown(f"**Goal / Task:** {user_input}")

        agent_run = agentic_plan_and_execute(
            model=model,
            goal=user_input,
            role=role,
            language=reply_language,
            doc_text=doc_text_context,
            chat_history=st.session_state.messages
        )

        agent_text = agent_run["agentic_explanation"]

        with st.chat_message("assistant"):
            st.markdown(agent_text)

        st.session_state.messages.append(
            {"user": user_input, "assistant": agent_text}
        )
        st.session_state.agent_runs.append(agent_run)

