import os
import re
import tempfile
from typing import List, TypedDict

import streamlit as st
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END

load_dotenv()

st.set_page_config(page_title="Corrective RAG Assistant", page_icon="🧠", layout="wide")

UPPER_TH = 0.7
LOWER_TH = 0.3


# -----------------------------------------------------------------
# State + structured output schemas
# -----------------------------------------------------------------
class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str


class DocEvalScore(BaseModel):
    score: float
    reason: str


class KeepOrDrop(BaseModel):
    keep: bool


class WebQuery(BaseModel):
    query: str


# -----------------------------------------------------------------
# Build the index + graph once per document set (cached across reruns)
# -----------------------------------------------------------------
@st.cache_resource(show_spinner="Building index from documents...")
def build_graph(pdf_paths: tuple):
    docs = []
    for p in pdf_paths:
        docs += PyPDFLoader(p).load()

    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    doc_eval_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict retrieval evaluator for RAG.\n"
         "You will be given ONE retrieved chunk and a question.\n"
         "Return a relevance score in [0.0, 1.0].\n"
         "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
         "- 0.0: chunk is irrelevant\n"
         "Be conservative with high scores.\n"
         "Also return a short reason.\n"
         "Output JSON only."),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ])
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

    filter_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict relevance filter.\n"
         "Return keep=true only if the sentence directly helps answer the question.\n"
         "Use ONLY the sentence. Output JSON only."),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ])
    filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user question into a web search query composed of keywords.\n"
         "Rules:\n"
         "- Keep it short (6-14 words).\n"
         "- If the question implies recency (e.g. recent/latest/last week/last month), "
         "add a constraint like (last 30 days).\n"
         "- Do NOT answer the question.\n"
         "- Return JSON with a single key: query"),
        ("human", "Question: {question}"),
    ])
    rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer ONLY using the provided context.\n"
         "If the context is empty or insufficient, say: 'I don't know.'"),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ])

    tavily = TavilySearch(max_results=5)

    def decompose_to_sentences(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def retrieve_node(state: State) -> State:
        return {"docs": retriever.invoke(state["question"])}

    def eval_each_doc_node(state: State) -> State:
        q = state["question"]
        scores, good = [], []
        for d in state["docs"]:
            out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
            scores.append(out.score)
            if out.score > LOWER_TH:
                good.append(d)

        if any(s > UPPER_TH for s in scores):
            return {"good_docs": good, "verdict": "CORRECT",
                    "reason": f"At least one retrieved chunk scored > {UPPER_TH}."}
        if len(scores) > 0 and all(s < LOWER_TH for s in scores):
            return {"good_docs": [], "verdict": "INCORRECT",
                    "reason": f"All retrieved chunks scored < {LOWER_TH}."}
        return {"good_docs": good, "verdict": "AMBIGUOUS",
                "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}."}

    def rewrite_query_node(state: State) -> State:
        out = rewrite_chain.invoke({"question": state["question"]})
        return {"web_query": out.query}

    def web_search_node(state: State) -> State:
        q = state.get("web_query") or state["question"]
        raw = tavily.invoke({"query": q})
        # langchain_tavily's TavilySearch returns a dict with a "results" key
        items = raw.get("results", []) if isinstance(raw, dict) else (raw or [])

        web_docs: List[Document] = []
        for r in items:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
        return {"web_docs": web_docs}

    def refine(state: State) -> State:
        q = state["question"]
        if state.get("verdict") == "CORRECT":
            docs_to_use = state["good_docs"]
        elif state.get("verdict") == "INCORRECT":
            docs_to_use = state["web_docs"]
        else:
            docs_to_use = state["good_docs"] + state["web_docs"]

        context = "\n\n".join(d.page_content for d in docs_to_use).strip()
        strips = decompose_to_sentences(context)
        kept = [s for s in strips if filter_chain.invoke({"question": q, "sentence": s}).keep]
        return {"strips": strips, "kept_strips": kept, "refined_context": "\n".join(kept).strip()}

    def generate(state: State) -> State:
        out = (answer_prompt | llm).invoke(
            {"question": state["question"], "context": state["refined_context"]}
        )
        return {"answer": out.content}

    def route_after_eval(state: State) -> str:
        return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"

    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_conditional_edges(
        "eval_each_doc", route_after_eval,
        {"refine": "refine", "rewrite_query": "rewrite_query"},
    )
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    return g.compile()


# -----------------------------------------------------------------
# Sidebar — knowledge base setup
# -----------------------------------------------------------------
st.sidebar.title("📚 Knowledge Base")
st.sidebar.caption("Upload PDFs to build the retriever's index, or place them in ./documents")

if st.sidebar.button("🗑️ Clear chat history"):
    st.session_state.history = []
    st.rerun()

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

pdf_paths: List[str] = []
if uploaded_files:
    tmp_dir = tempfile.mkdtemp()
    for f in uploaded_files:
        path = os.path.join(tmp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        pdf_paths.append(path)
else:
    default_dir = "./documents"
    if os.path.isdir(default_dir):
        pdf_paths = [
            os.path.join(default_dir, f)
            for f in sorted(os.listdir(default_dir))
            if f.lower().endswith(".pdf")
        ]

if not pdf_paths:
    st.sidebar.warning("No PDFs found. Upload at least one to continue.")
    st.stop()

st.sidebar.success(f"{len(pdf_paths)} document(s) loaded")
for p in pdf_paths:
    st.sidebar.caption(f"• {os.path.basename(p)}")

if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.sidebar.error("Missing OPENAI_API_KEY or TAVILY_API_KEY — add them to your .env file.")
    st.stop()

graph = build_graph(tuple(pdf_paths))

# -----------------------------------------------------------------
# Main chat UI
# -----------------------------------------------------------------
st.title("🧠 Corrective RAG Assistant")
st.caption("Answers are graded for relevance (CORRECT / INCORRECT / AMBIGUOUS) with automatic web-search fallback.")

VERDICT_BADGE = {"CORRECT": "🟢", "INCORRECT": "🔴", "AMBIGUOUS": "🟡"}

if "history" not in st.session_state:
    st.session_state.history = []


def render_details(turn: dict):
    with st.expander("Show retrieval details"):
        st.write(f"Retrieved chunks: {len(turn.get('docs', []))}")
        st.write(f"Sentences kept after filtering: {len(turn.get('kept_strips', []))}")
        if turn.get("web_docs"):
            st.write("Web sources used:")
            for wd in turn["web_docs"]:
                title = wd.metadata.get("title") or "source"
                url = wd.metadata.get("url") or ""
                st.write(f"- [{title}]({url})")


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        badge = VERDICT_BADGE.get(turn["verdict"], "⚪")
        st.caption(f"{badge} Verdict: **{turn['verdict']}** — {turn['reason']}")
        render_details(turn)

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke({"question": question})

        st.write(result["answer"])
        badge = VERDICT_BADGE.get(result["verdict"], "⚪")
        st.caption(f"{badge} Verdict: **{result['verdict']}** — {result['reason']}")
        render_details(result)

    st.session_state.history.append({**result, "question": question})