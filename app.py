import streamlit as st
from graph import fact_court_app
from tools import ensure_pinecone_index
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Fact Court", layout="wide")

st.sidebar.title("Fact Court Settings")

# BYOK setup
provider = st.sidebar.selectbox("Select LLM Provider", ["OpenAI", "Google Gemini", "Hugging Face"])
api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")

st.title("Fact Court")
st.markdown("Enter your query below. The Fact Court will investigate, debate , investigate , cross check , credibility check ,  scrutiny , arugument , historical referencing , compliance  and deliver a verdict based on facts , real life realities , logical facts , data ,reasoning and rules and regulations .")

import base64

claim = st.text_area("Enter what you want to findout here...")
image_upload = st.file_uploader("Optional: Upload image evidence", type=["jpg", "png", "jpeg"])

if st.button("Submit to Court"):
    if not api_key:
        st.error("Please enter your API Key in the sidebar.")
    elif not claim:
        st.warning("Please enter a claim.")
    else:
        # Handle Image
        image_b64 = None
        if image_upload:
            image_bytes = image_upload.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
        # Initialize DB
        with st.spinner("Initializing Global Court Records (Pinecone)..."):
            try:
                ensure_pinecone_index()
            except Exception as e:
                st.warning(f"Pinecone note: {e}")
        
        st.write("---")
        st.info("Court is in Session... and actively investigating will deliver a final verdict shortly")
        
        initial_state = {
            "claim": claim,
            "image_data": image_b64,
            "provider": provider,
            "api_key": api_key,
            "iterations": 0
        }
        
        # Stream the LangGraph execution
        try:
            for step in fact_court_app.stream(initial_state):
                for node_name, state_update in step.items():
                    st.write(f" **{node_name}** completed.")
            
            # Get final state
            final_state = fact_court_app.invoke(initial_state)
            
            st.write("---")
            st.subheader("Final Verdict")
            st.success(final_state.get('final_verdict', 'No verdict reached.'))
            st.metric(label="Confidence Score", value=f"{final_state.get('confidence_score', 0)}%")
            
            with st.expander("See Full Court Report"):
                st.write(final_state.get('court_report', ''))
                
            with st.expander(" See Provenance Ledger (Evidence)"):
                st.json(final_state.get('provenance_ledger', []))
                
        except Exception as e:
            st.error(f"Court Error: {e}")
