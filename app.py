import operator, uuid
from typing import TypedDict, List, Annotated

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="Self-Healing RAG", page_icon="🔄", layout="wide")

st.markdown("""
<style>
.stApp {
  background: radial-gradient(circle at 20% 0%, #1e1b4b 0%, #0b0b1a 40%, #050510 100%);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #15152a 0%, #0b0b1a 100%);
  border-right: 1px solid rgba(167,139,250,0.15);
}
h1 {
  background: linear-gradient(90deg, #a78bfa, #22d3ee);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 800 !important;
  letter-spacing: -0.02em;
}
[data-testid="stChatMessage"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(167,139,250,0.12);
  border-radius: 16px;
  padding: 16px 20px;
  backdrop-filter: blur(10px);
}
.stButton > button {
  background: linear-gradient(90deg, #7c3aed, #06b6d4);
  color: white;
  border: none;
  border-radius: 12px;
  font-weight: 600;
  padding: 0.6rem 1.2rem;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 30px -10px rgba(124,58,237,0.6);
}
[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.03);
  border: 1px dashed rgba(167,139,250,0.3);
  border-radius: 14px;
  padding: 8px;
}
.stChatInput textarea {
  background: rgba(255,255,255,0.05) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(167,139,250,0.2) !important;
}
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(167,139,250,0.12);
  border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

MAX_ATTEMPTS = 3

@st.cache_resource(show_spinner=False)
def get_models():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0), FastEmbedEmbeddings()
llm, embeddings = get_models()

answer_prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}""")

critic_prompt = ChatPromptTemplate.from_template(
    """You are a fact-checker. Your only job is to detect made-up information.
Check whether everything the answer says can be found in the context.
- If every statement is supported by the context, reply YES (a short answer is fine).
- Only reply NO if the answer contains something NOT in the context or contradicting it.
Do NOT reply NO just because the answer is brief.

Reply with exactly one word on the first line: YES or NO.
On the second line, give a one-sentence reason.

Context:
{context}

Answer:
{answer}""")

rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite the question to be clearer and more specific for searching documents. "
    "Return ONLY the rewritten question.\n\nQuestion: {question}")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_retriever(files):
    docs = []
    for f in files:
        if f.name.lower().endswith(".pdf"):
            reader = PdfReader(f)
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        else:
            text = f.read().decode("utf-8", errors="ignore")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": f.name}))
    if not docs:
        return None, 0, 0
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    vs = Chroma.from_documents(chunks, embeddings, collection_name=f"kb_{uuid.uuid4().hex[:8]}")
    return vs.as_retriever(search_kwargs={"k": 4}), len(docs), len(chunks)

class GraphState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    grounded: bool
    attempts: int
    retriever: object
    log: Annotated[List[str], operator.add]

def retrieve(state):
    docs = state["retriever"].invoke(state["question"])
    return {"documents": docs, "log": [f"🔎 Searched for: \"{state['question']}\" → found {len(docs)} chunks"]}

def generate(state):
    answer = llm.invoke(answer_prompt.format(
        context=format_docs(state["documents"]), question=state["original_question"])).content
    return {"generation": answer, "attempts": state["attempts"] + 1,
            "log": [f"✍️ Wrote an answer (attempt {state['attempts'] + 1})"]}

def grade(state):
    resp = llm.invoke(critic_prompt.format(
        context=format_docs(state["documents"]), answer=state["generation"])).content
    grounded = "YES" in resp.strip().split("\n")[0].upper()
    return {"grounded": grounded, "log": [f"🕵️ Critic verdict: {'grounded ✅' if grounded else 'not grounded ❌'}"]}

def rewrite_query(state):
    new_q = llm.invoke(rewrite_prompt.format(question=state["original_question"])).content.strip()
    return {"question": new_q, "log": ["🔁 Rephrased the question and retrying..."]}

def fallback(state):
    return {"generation": "I don't have enough information in these documents to answer that confidently.",
            "log": ["🛑 Couldn't ground an answer — responding honestly."]}

def decide(state):
    if state["grounded"]:
        return "good"
    return "give_up" if state["attempts"] >= MAX_ATTEMPTS else "retry"

@st.cache_resource(show_spinner=False)
def build_graph():
    wf = StateGraph(GraphState)
    for name, fn in [("retrieve", retrieve), ("generate", generate), ("grade", grade),
                     ("rewrite_query", rewrite_query), ("fallback", fallback)]:
        wf.add_node(name, fn)
    wf.add_edge(START, "retrieve")
    wf.add_edge("retrieve", "generate")
    wf.add_edge("generate", "grade")
    wf.add_conditional_edges("grade", decide,
        {"good": END, "give_up": "fallback", "retry": "rewrite_query"})
    wf.add_edge("rewrite_query", "retrieve")
    wf.add_edge("fallback", END)
    return wf.compile()
graph = build_graph()

st.title("🔄 Self-Healing RAG")
st.caption("Upload documents, ask questions, and watch the system fact-check and correct itself.")

st.session_state.setdefault("retriever", None)
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("📁 Your documents")
    uploaded = st.file_uploader("Upload files", type=["txt", "md", "pdf"], accept_multiple_files=True)
    if st.button("Build knowledge base", type="primary", use_container_width=True):
        if uploaded:
            with st.spinner("Reading and indexing your documents..."):
                retriever, n_docs, n_chunks = build_retriever(uploaded)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.messages = []
                st.success(f"Indexed {n_docs} document(s) into {n_chunks} chunks.")
            else:
                st.error("Couldn't read any text from those files.")
        else:
            st.warning("Please upload at least one file first.")
    st.info("✅ Ready — ask away!" if st.session_state.retriever
            else "Upload documents and click *Build knowledge base* to begin.")

if st.session_state.retriever is None:
    st.info("👈 Start by uploading documents in the sidebar.")
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["role"] == "assistant" and m.get("log"):
                with st.expander("🔍 How it worked"):
                    for line in m["log"]:
                        st.write(line)
    q = st.chat_input("Ask a question about your documents...")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking, fact-checking, and self-correcting..."):
                result = graph.invoke({
                    "question": q, "original_question": q, "documents": [],
                    "generation": "", "grounded": False, "attempts": 0,
                    "retriever": st.session_state.retriever, "log": [],
                })
            st.write(result["generation"])
            with st.expander("🔍 How it worked"):
                for line in result["log"]:
                    st.write(line)
        st.session_state.messages.append(
            {"role": "assistant", "content": result["generation"], "log": result["log"]})