from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace

def get_llm(provider: str, api_key: str):
    """Dynamically load the LLM based on user selection."""
    if not api_key:
        raise ValueError("API Key is required")
        
    if provider == "OpenAI":
        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.2)
    elif provider == "Google Gemini":
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key, temperature=0.2)
    elif provider == "Hugging Face":
        # Note: ChatHuggingFace usually wraps a HuggingFaceEndpoint
        from langchain_huggingface import HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token=api_key)
        return ChatHuggingFace(llm=llm)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Simple Prompts
RESEARCH_PROMPT = """You are a forensic researcher. Analyze this claim and extract key search queries.
Claim: {claim}
Output a comma-separated list of 3 search queries (neutral, pro-bias, anti-bias)."""

ROUTER_PROMPT = """Does this claim require highly specialized Medical, Legal, or Scientific knowledge to verify?
Claim: {claim}
Answer ONLY 'yes' or 'no'."""

EXPERT_PROMPT = """You are an expert witness. Provide a factual brief on this topic based on scientific/legal consensus.
Claim: {claim}"""

SUPPORTER_PROMPT = """You are the Supporter. Using ONLY the facts in the Provenance Ledger, argue that the claim is true or partially true.
Claim: {claim}
Ledger: {ledger}
Expert Brief (if any): {expert_brief}"""

SKEPTIC_CROSS_EXAM_PROMPT = """You are the Skeptic. Cross-examine the Supporter's argument. Attack the credibility of their sources or logic.
Supporter Argument: {argument}
Ledger: {ledger}"""

SKEPTIC_PROMPT = """You are the Skeptic. Using ONLY the facts in the Provenance Ledger, argue that the claim is FALSE.
Claim: {claim}
Ledger: {ledger}
Expert Brief (if any): {expert_brief}"""

SUPPORTER_CROSS_EXAM_PROMPT = """You are the Supporter. Cross-examine the Skeptic's argument. Attack their logic.
Skeptic Argument: {argument}
Ledger: {ledger}"""

VALIDATOR_PROMPT = """You are the Validator. Did the Supporter or Skeptic invent facts not found in the Ledger? 
Ledger: {ledger}
Supporter: {supporter_arg}
Skeptic: {skeptic_arg}
Output 'yes' if they hallucinated, or 'no' if they strictly used the ledger. Also provide brief notes."""

EVALUATOR_PROMPT = """You are the Evaluator. Review the arguments and issue a final verdict (True, Mostly True, Half True, Mostly False, False).
Claim: {claim}
Supporter: {supporter_arg}
Skeptic: {skeptic_arg}
Ledger: {ledger}

Provide a confidence score (0-100), the short verdict, and a detailed court report explaining your reasoning."""
