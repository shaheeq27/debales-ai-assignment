"""
cli.py – Interactive CLI for the Debales AI assistant.

Usage:
    python cli.py
"""

from __future__ import annotations

import sys
from langchain_core.messages import BaseMessage

# Add src/ to path so we can import agent
sys.path.insert(0, "src")
from agent import chat, get_graph  # noqa: E402  (after sys.path modification)


BANNER = """
╔══════════════════════════════════════════════╗
║      Debales AI – Intelligent Assistant      ║
║  Type 'exit' or Ctrl-C to quit              ║
╚══════════════════════════════════════════════╝
"""

def main() -> None:
    print(BANNER)

    # Warm up the graph (loads vector store, etc.)
    print("Loading knowledge base … ", end="", flush=True)
    try:
        get_graph()
        print("ready!\n")
    except FileNotFoundError:
        print("\n⚠  FAISS index not found.")
        print("   Run `python scripts/ingest.py` first, then start the CLI again.\n")
        sys.exit(1)

    history: list[BaseMessage] = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        print("Assistant: ", end="", flush=True)
        answer = chat(query, history)
        print(answer)
        print()

        # Maintain conversation history (last 10 exchanges to control token use)
        from langchain_core.messages import HumanMessage, AIMessage
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer))
        history = history[-20:]   # keep last 10 turns (20 messages)


if __name__ == "__main__":
    main()
