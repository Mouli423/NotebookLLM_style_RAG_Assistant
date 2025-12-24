# 📘 NotebookLLM-Style Multi-Source RAG Assistant

A Streamlit-based Agentic RAG application that allows users to query PDF documents, YouTube videos, and live web sources (Wikipedia, Arxiv, Web search) using LangChain, LangGraph-style patterns, and tool-augmented agents.

This project demonstrates context-aware RAG, agent tool usage, conversation memory, and mode-aware session handling.

##  Features
###  Multi-Input RAG Modes

PDF RAG – Upload multiple PDFs and ask contextual questions

YouTube RAG – Query YouTube video transcripts

Web / Wiki / Arxiv Search – Live research using tools

###  Context-Aware Conversations

History-aware question rewriting

Session-based memory

Follow-up question handling

###  Agentic Search Mode

Tool-calling agent with:

Wikipedia

Arxiv

DuckDuckGo Search

Tavily Search

Chooses tools dynamically for factual queries

###  NotebookLLM-Style Behavior

Uses retrieved context first

Falls back to general knowledge when needed

Avoids hallucination

Explicitly states uncertainty when data is insufficient


## Usage Guide
###  PDF RAG

Upload one or more PDF files

Ask questions related to the document

Ask follow-up questions using chat history

###  YouTube RAG

Paste a YouTube video URL

Ask questions about the video content

Supports conversational queries

###  Web / Wiki / Arxiv Search

Ask factual or research-based questions

Agent automatically selects:

Wikipedia

Arxiv

Web search

Returns summarized, accurate responses

###  Key Concepts Demonstrated

History-Aware Retrieval

Agentic Tool Selection

Multi-mode Input Handling

Session-based Chat Memory

Prompt-controlled Hallucination Prevention

Modular RAG Design

###  Future Improvements

Add LangGraph-based orchestration

Integrate MCP (Model Context Protocol)

Persistent memory using Redis / Postgres

UI enhancements (source highlighting)
