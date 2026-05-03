"""
Debales AI Assistant – LangGraph Agent
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv()

# ── LLM & embeddings ────────────────────────────────────────────────────────
llm = ChatOllama(model="llama3")
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# ── SERP tool ────────────────────────────────────────────────────────────────
serp = SerpAPIWrapper()
# ── Vector store (loaded lazily) ─────────────────────────────────────────────
_vectorstore: FAISS | None = None

def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        index_path = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                "FAISS index not found. Run `python scripts/ingest.py` first."
            )
        _vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return _vectorstore


# ── Graph state ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    route: str               # "rag" | "serp" | "both"
    rag_context: str
    serp_context: str
    final_answer: str


# ── Node: router ─────────────────────────────────────────────────────────────
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant for the Debales AI chatbot.

Classify the user query into ONE of three categories:
- "rag"  → the question is specifically about Debales AI (company, products, integrations, pricing, blog, team, use-cases)
- "serp" → the question is entirely about external topics unrelated to Debales AI
- "both" → the question mixes Debales AI topics with external information needs

Respond with ONLY the lowercase word: rag, serp, or both."""),
    ("human", "{query}"),
])

def router_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    chain = ROUTER_PROMPT | llm
    result = chain.invoke({"query": query})
    route = result.content.strip().lower()
    if route not in ("rag", "serp", "both"):
        route = "serp"   # safe fallback
    return {**state, "query": query, "route": route}


def route_decision(state: AgentState) -> Literal["rag_node", "serp_node", "both_node"]:
    return f"{state['route']}_node"   # type: ignore[return-value]


# ── Node: RAG retrieval ───────────────────────────────────────────────────────
def rag_node(state: AgentState) -> AgentState:
    vs = get_vectorstore()
    docs = vs.similarity_search(state["query"], k=4)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return {**state, "rag_context": context}


# ── Node: SERP search ─────────────────────────────────────────────────────────
def serp_node(state: AgentState) -> AgentState:
    try:
        result = serp.run(state["query"])
    except Exception as exc:
        result = f"[SERP error: {exc}]"
    return {**state, "serp_context": result}


# ── Node: Both ────────────────────────────────────────────────────────────────
def both_node(state: AgentState) -> AgentState:
    state = rag_node(state)
    state = serp_node(state)
    return state


# ── Node: Answer generation ───────────────────────────────────────────────────
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Debales AI assistant.

Use ONLY the provided context to answer the user's question.
If the context does not contain enough information, say so clearly – do NOT hallucinate.

RAG context (Debales AI knowledge base):
{rag_context}

Web search context:
{serp_context}

Guidelines:
- Be concise and accurate.
- Cite sources when available.
- Never make up facts about Debales AI."""),
    ("human", "{query}"),
])

def answer_node(state: AgentState) -> AgentState:
    chain = ANSWER_PROMPT | llm
    result = chain.invoke({
        "rag_context": state.get("rag_context", "N/A"),
        "serp_context": state.get("serp_context", "N/A"),
        "query": state["query"],
    })
    answer = result.content
    new_messages = list(state["messages"]) + [AIMessage(content=answer)]
    return {**state, "final_answer": answer, "messages": new_messages}


# ── Build LangGraph ───────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("router_node", router_node)
    g.add_node("rag_node",    rag_node)
    g.add_node("serp_node",   serp_node)
    g.add_node("both_node",   both_node)
    g.add_node("answer_node", answer_node)

    g.set_entry_point("router_node")

    g.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "rag_node":  "rag_node",
            "serp_node": "serp_node",
            "both_node": "both_node",
        },
    )

    for node in ("rag_node", "serp_node", "both_node"):
        g.add_edge(node, "answer_node")

    g.add_edge("answer_node", END)

    return g.compile()


# ── Public interface ──────────────────────────────────────────────────────────
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def chat(query: str, history: list[BaseMessage] | None = None) -> str:
    """Single-turn or multi-turn entry point."""
    messages = list(history or []) + [HumanMessage(content=query)]
    initial_state: AgentState = {
        "messages":     messages,
        "query":        "",
        "route":        "",
        "rag_context":  "",
        "serp_context": "",
        "final_answer": "",
    }
    result = get_graph().invoke(initial_state)
    return result["final_answer"]
