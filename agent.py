# agent.py
from __future__ import annotations

from typing import Literal, Optional, TypedDict, List

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent

from prompt import (
    SUPERVISOR_SYSTEM, RAG_SYSTEM, SUMMARIZER_SYSTEM, TEXT_SYSTEM
)
from tools import (
    string_search_tool,
    make_summarizer_tool,
    rag_build_index, 
    rag_load_index, 
    rag_init_reranker, 
    rag_search, rag_answer
)

from env import get_env

OPENAI_API_KEY = get_env("OPENAI_API_KEY")

class AgentState(TypedDict):
    messages: List[BaseMessage]
    route: Optional[str]


def create_llm_agent(model, tools, system_prompt: str, name: str):
    """
    Minimal agent factory using LangGraph's prebuilt ReAct agent.
    It supports tool calling and is invoked via:
        agent.invoke({"messages": state["messages"]})
    """
    return create_agent(model, 
                        tools, 
                        system_prompt=system_prompt, 
                        name=name)


def build_graph(llm, embeddings):
    # ---- Agents ----
    text_agent = create_llm_agent(llm, 
                                  [string_search_tool], 
                                  TEXT_SYSTEM, 
                                  "TextAgent")

    summarizer_tool = make_summarizer_tool(llm)
    summarizer_agent = create_llm_agent(llm, 
                                        [summarizer_tool], 
                                        SUMMARIZER_SYSTEM, 
                                        "SummarizerAgent")

    # Bind embeddings/llm into RAG tools so signatures stay simple for tool-calling
    def _rag_load_index(index_dir: str) -> str:
        """
        Loads a previously built FAISS + BM25 index into memory.

        Args:
            index_dir (str): Path to the directory containing FAISS files and bm25.pkl.

        Returns:
            str: Status message indicating success or an error.

        Notes:
            Must be called before _rag_search or _rag_answer if index is not already loaded.
        """
        return rag_load_index(index_dir=index_dir, embeddings=embeddings)

    def _rag_build_index(payload: str) -> str:
        """
        Builds a hybrid RAG index (FAISS + BM25) from local documents.

        Args:
            payload (str): JSON string.
                Supported formats:
                - List[str]: File paths. Index stored at "./faiss_store".
                - Dict with keys:
                    paths (List[str]): File paths.
                    index_dir (str): Output directory for the index.

        Returns:
            str: Status message indicating build success or an error.

        Side Effects:
            Writes FAISS index and bm25.pkl to disk and loads them into memory.
        """
        import json as _json
        data = _json.loads(payload)

        if isinstance(data, list):
            return rag_build_index(
                file_paths_json=_json.dumps(data),
                index_dir="./faiss_store",
                embeddings=embeddings,
            )

        return rag_build_index(
            file_paths_json=_json.dumps(data["paths"]),
            index_dir=data["index_dir"],
            embeddings=embeddings,
        )

    def _rag_init_reranker(model_name: str) -> str:
        """
        Initializes a local cross-encoder reranker for retrieval refinement.

        Args:
            model_name (str): HuggingFace model ID for the reranker.

        Returns:
            str: Status message indicating the reranker was loaded.

        Notes:
            Optional. If not called, retrieval runs without reranking.
        """
        return rag_init_reranker(model_name=model_name)

    def _rag_search(query: str) -> str:
        """
    Retrieves relevant document chunks using hybrid search.

    Args:
        query (str): Natural language search query.

    Returns:
        str: JSON string containing ranked chunks with:
            - rank (int)
            - chunk_id (int)
            - content (str)
            - source (str)
            - page (Optional[int])

    Notes:
        Uses BM25 + FAISS with reciprocal rank fusion and optional reranking.
    """
        return rag_search(query)

    def _rag_answer(query: str) -> str:
        """
        Answers a question using retrieved context with citations.

        Args:
            query (str): Natural language question.

        Returns:
            str: Answer text with inline citations and a citation list.

        Notes:
            Uses only retrieved document context.
            If information is insufficient, returns an explicit disclaimer.
        """
        return rag_answer(query, llm=llm)

    rag_agent = create_llm_agent(
        llm,
        [_rag_load_index, 
         _rag_build_index, 
         _rag_init_reranker, 
         _rag_search, 
         _rag_answer],
        RAG_SYSTEM,
        "RAGAgent",
    )

    # Supervisor (no tools)
    supervisor_agent = create_llm_agent(llm, 
                                        [], 
                                        SUPERVISOR_SYSTEM, 
                                        "Supervisor")
    # ---- Runners ----
    def run_supervisor(state: AgentState) -> AgentState:
        print("\n--- Supervisor ---")
        out = supervisor_agent.invoke({"messages": state["messages"]})
        state["messages"] = out["messages"]
        last = state["messages"][-1]
        decision = (last.content or "FINISH").strip() if hasattr(last, "content") else "FINISH"
        state["route"] = decision
        return state

    def run_text(state: AgentState) -> AgentState:
        print("\n--- TextAgent ---")
        out = text_agent.invoke({"messages": state["messages"]})
        state["messages"] = out["messages"]
        return state

    def run_summarizer(state: AgentState) -> AgentState:
        print("\n--- SummarizerAgent ---")
        out = summarizer_agent.invoke({"messages": state["messages"]})
        state["messages"] = out["messages"]
        return state

    def run_rag(state: AgentState) -> AgentState:
        print("\n--- RAGAgent ---")
        out = rag_agent.invoke({"messages": state["messages"]})
        state["messages"] = out["messages"]
        return state

    def route(state: AgentState) -> Literal["TextAgent", "SummarizerAgent", "RAGAgent", "FINISH"]:
        r = (state.get("route") or "FINISH").strip()
        if r in ("TextAgent", "SummarizerAgent", "RAGAgent", "FINISH"):
            return r
        return "FINISH"

    # ---- Graph ----
    g = StateGraph(AgentState)
    g.add_node("Supervisor", run_supervisor)
    g.add_node("TextAgent", run_text)
    g.add_node("SummarizerAgent", run_summarizer)
    g.add_node("RAGAgent", run_rag)

    g.set_entry_point("Supervisor")
    g.add_conditional_edges(
        "Supervisor",
        route,
        {
            "TextAgent": "TextAgent",
            "SummarizerAgent": "SummarizerAgent",
            "RAGAgent": "RAGAgent",
            "FINISH": END,
        },
    )

    # After any agent, return to supervisor for the next step
    g.add_edge("TextAgent", "Supervisor")
    g.add_edge("SummarizerAgent", "Supervisor")
    g.add_edge("RAGAgent", "Supervisor")

    return g.compile()
