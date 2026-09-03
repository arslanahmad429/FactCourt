# Fact Court 

https://github.com/user-attachments/assets/d387a34e-5ebb-41da-bb41-4a6130509a58



Fact Court is an advanced, multi-agent AI framework designed to transcend traditional 'one-shot' LLM fact-checkers. Instead of relying on a single prompt, Fact Court operates as a fully autonomous legal system. It orchestrates a sophisticated LangGraph architecture where specialized AI nodes—acting as Researchers, Experts, Prosecution, and Defense—actively scrape the web, assemble evidence, query a Pinecone vector database for historical precedents, and aggressively cross-examine each other's arguments before submitting the case to an impartial AI Judge. This rigorous, adversarial product engineering ensures high-scrutiny, bias-resistant verdicts with absolute transparency.

🚀 **[Try the Live App Here: factcourt.streamlit.app](https://factcourt.streamlit.app)**

## Architecture Flowchart

Watch the video below to understand the multi-agent AI architecture powering Fact Court:

<video src="architecture_flowchart.mp4" controls="controls" style="max-width: 100%;"></video>

## Tech Stack
- **LangGraph:** Orchestrates the multi-agent workflow (Researcher, Router, Expert, Lawyers, Validator, Judge).
- **Streamlit:** Powers the interactive web interface.
- **Pinecone:** Universal vector database for storing and retrieving precedents (case law).
- **LangChain:** Handles LLM integration (supports Google Gemini, OpenAI, and Hugging Face).
- **DuckDuckGo & BeautifulSoup:** For real-time fact-checking and evidence gathering.
