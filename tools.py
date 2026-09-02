import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()

# We will initialize Pinecone lazily inside functions to prevent import crashes
# if the environment variables aren't loaded yet on Streamlit Cloud.
index_name = os.getenv("PINECONE_INDEX_NAME", "fact-court-db")

def get_pinecone_client():
    """Lazily load the Pinecone client, checking Streamlit secrets if needed."""
    api_key = os.getenv("PINECONE_API_KEY")
    
    # Fallback for Streamlit Cloud secrets just in case
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("PINECONE_API_KEY")
        except Exception:
            pass
            
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set. Please add it to your Streamlit Secrets.")
        
    return Pinecone(api_key=api_key)

def ensure_pinecone_index():
    """Ensure the Pinecone index exists."""
    pc = get_pinecone_client()
    if index_name not in pc.list_indexes().names():
        try:
            pc.create_index(
                name=index_name,
                dimension=1536, # OpenAI text-embedding-3-small dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1' # Default free tier region
                )
            )
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        except Exception as e:
            print(f"Error creating index: {e}")

def search_web(query: str, max_results: int = 3) -> list:
    """Perform a web search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def scrape_url(url: str) -> str:
    """Deep scrape a URL for its main text content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        # Return first 3000 chars to avoid massive context
        return text[:3000] 
    except Exception as e:
        print(f"Scraping error: {e}")
        return ""
