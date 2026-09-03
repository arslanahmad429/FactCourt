# Fact Court ⚖️

Fact Court is an AI-powered, multi-agent courtroom that investigates claims, gathers evidence via web scraping, conducts debates between prosecution and defense, and delivers a final verdict with a confidence score.

## Architecture Flowchart

Watch the video below to understand the multi-agent AI architecture powering Fact Court:

https://github.com/arslanahmad429/FactCourt/raw/main/architecture_flowchart.mp4

## Tech Stack
- **LangGraph:** Orchestrates the multi-agent workflow (Researcher, Router, Expert, Lawyers, Validator, Judge).
- **Streamlit:** Powers the interactive web interface.
- **Pinecone:** Universal vector database for storing and retrieving precedents (case law).
- **LangChain:** Handles LLM integration (supports Google Gemini, OpenAI, and Hugging Face).
- **DuckDuckGo & BeautifulSoup:** For real-time fact-checking and evidence gathering.
