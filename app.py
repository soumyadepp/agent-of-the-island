# app.py
from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage

from agent import build_graph
from ingest import ensure_index_from_dir, init_reranker
from env import get_env

OPENAI_API_KEY = get_env("OPENAI_API_KEY")

def main():
    # ---- CONFIG (set once) ----
    DATA_DIR = "C:/Users/Arya/Downloads/combine_excels/downloads"
    INDEX_DIR = "./faiss_store"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # ---------------------------

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1) Pre-req: build/load index from directory
    print(ensure_index_from_dir(DATA_DIR, 
                                INDEX_DIR, 
                                embeddings))

    # 2) Optional: init reranker
    print(init_reranker(RERANKER_MODEL))

    # 3) Build agentic graph
    graph = build_graph(llm=llm, embeddings=embeddings)

    # 4) Query loop
    state = {"messages": [], "route": None}
    print("\nReady. Ask questions. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        state["messages"].append(HumanMessage(content=q))
        state = graph.invoke(state)

        last = state["messages"][-2]
        print("\nAssistant:\n", getattr(last, "content", last), "\n")


if __name__ == "__main__":
    main()
