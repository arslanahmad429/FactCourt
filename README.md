# Fact Court 

https://github.com/user-attachments/assets/d387a34e-5ebb-41da-bb41-4a6130509a58



Fact Court is an AI-powered, multi-agent courtroom that investigates claims, gathers evidence via web scraping, conducts debates between prosecution and defense, and delivers a final verdict with a confidence score.

## Architecture Flowchart

Watch the video below to understand the multi-agent AI architecture powering Fact Court:

<video src="architecture_flowchart.mp4" controls="controls" style="max-width: 100%;"></video>

## Tech Stack
- **LangGraph:** Orchestrates the multi-agent workflow (Researcher, Router, Expert, Lawyers, Validator, Judge).
- **Streamlit:** Powers the interactive web interface.
- **Pinecone:** Universal vector database for storing and retrieving precedents (case law).
- **LangChain:** Handles LLM integration (supports Google Gemini, OpenAI, and Hugging Face).
- **DuckDuckGo & BeautifulSoup:** For real-time fact-checking and evidence gathering.
