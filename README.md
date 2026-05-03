## 🚀 Overview

This project is an AI-powered assistant built using LangGraph that answers questions about Debales AI using a Retrieval-Augmented Generation (RAG) pipeline.

It combines:
- FAISS vector database for efficient document retrieval
- Intelligent routing between internal knowledge and external queries
- A chatbot interface (CLI and optional Streamlit UI)

To ensure the project runs without paid dependencies, OpenAI APIs were replaced with:
- HuggingFace sentence-transformer embeddings (local)
- Ollama (LLaMA3) as the local language model

The system crawls Debales AI’s website, builds a searchable knowledge base, and generates context-aware responses using a modular, production-style architecture.# 🤖 Debales AI Assistant — LangGraph RAG System

A production-style AI chatbot that answers questions about **Debales AI** using a Retrieval-Augmented Generation (RAG) pipeline, with intelligent routing between internal knowledge and external queries.

---

## 🔥 What this project does

This system:
- Answers company-specific questions using a **vector database (FAISS)**
- Falls back to external search when needed
- Uses a **LangGraph workflow** to route queries intelligently
- Runs **fully locally** (no paid APIs required)

---

## 🧠 Architecture Overview
User Query
│
▼
┌─────────────┐
│ Router Node │  → decides: rag | serp | both
└──────┬──────┘
│
┌───┴────────────┐
│                │                │
▼                ▼                ▼
RAG Node       SERP Node        Both Node
(FAISS)        (Search)         (Combined)
│                │                │
└────────────────┴────────────────┘
│
▼
┌──────────────┐
│ Answer Node  │
└──────────────┘
│
▼
Final Answer
---

## ⚙️ Tech Stack

- **LangChain + LangGraph** — workflow orchestration
- **FAISS** — vector database
- **Sentence Transformers** — local embeddings
- **Ollama (LLaMA3)** — local LLM
- **BeautifulSoup + Requests** — web scraping
- **Streamlit** — optional web UI

---

## 📁 Project Structure
debales-ai-assignment/
├── agent.py              # LangGraph workflow (core logic)
├── ingest.py             # Website crawler + FAISS builder
├── cli.py                # CLI chatbot interface
├── app.py                # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/shaheeq27/debales-ai-assignment.git
cd debales-ai-assignment
2. Create virtual environment
python3 -m venv venv
source venv/bin/activate
**3. Install dependencies
**pip install -r requirements.txt
pip install sentence-transformers
**4. Setup environment variables
**cp .env.example .env

🧱 Build Knowledge Base
python3 ingest.py

This will:
* Crawl ~60 pages from Debales AI website
* Chunk text into embeddings
* Store vectors in FAIS

🧠 Run Local LLM (Required)

Install Ollama:
👉 https://ollama.com/download

Then run:ollama run llama3
💬 Run the chatbot

CLI
python3 cli.py

⸻

🧪 Example Queries

Try asking:

* What is Debales AI?
* What industries does Debales AI serve?
* How does Debales AI help logistics companies?
* Compare Debales AI with competitors

⸻

💡 Key Features

* 🔍 RAG-based retrieval from real website data
* 🧠 LLM-based routing (rag / serp / both)
* ⚡ Local embeddings (no API cost)
* 🤖 Local LLM via Ollama
* 💬 Multi-turn conversation support

⸻

⚠️ Notes

* This project avoids paid APIs by using:
    * HuggingFace embeddings
    * Ollama (local LLM)
* Ensure Ollama is running before starting chatbot
* FAISS index is generated locally and not committed

⸻

📌 Design Decisions

* Avoid hallucination: model is grounded in retrieved context
* Modular architecture: ingestion, retrieval, generation separated
* Cost-efficient: fully local execution
