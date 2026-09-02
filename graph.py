import json
from langgraph.graph import StateGraph, START, END
from state import FactCourtState
from agents import (
    get_llm, RESEARCH_PROMPT, ROUTER_PROMPT, EXPERT_PROMPT,
    SUPPORTER_PROMPT, SKEPTIC_CROSS_EXAM_PROMPT, SKEPTIC_PROMPT,
    SUPPORTER_CROSS_EXAM_PROMPT, VALIDATOR_PROMPT, EVALUATOR_PROMPT
)
from tools import search_web, scrape_url
import datetime

# --- Node Functions ---

def researcher_node(state: FactCourtState):
    """Gathers evidence via search and scraping."""
    llm = get_llm(state['provider'], state['api_key'])
    queries_text = llm.invoke(RESEARCH_PROMPT.format(claim=state['claim'])).content
    queries = [q.strip() for q in queries_text.split(',') if q.strip()]
    
    ledger = []
    for q in queries[:2]: # limit to 2 for speed
        results = search_web(q, max_results=2)
        for r in results:
            url = r.get('href', '')
            if url:
                scraped_text = scrape_url(url)
                if scraped_text:
                    ledger.append({
                        "url": url,
                        "text": scraped_text[:1000], # Keep it concise for context limit
                        "timestamp": str(datetime.datetime.now())
                    })
    
    return {"provenance_ledger": ledger, "iterations": state.get("iterations", 0) + 1}

def topic_router(state: FactCourtState):
    """Decides if an expert is needed."""
    llm = get_llm(state['provider'], state['api_key'])
    ans = llm.invoke(ROUTER_PROMPT.format(claim=state['claim'])).content.strip().lower()
    needs_expert = 'yes' in ans
    return {"needs_expert": needs_expert}

def expert_node(state: FactCourtState):
    """Provides expert brief."""
    llm = get_llm(state['provider'], state['api_key'])
    brief = llm.invoke(EXPERT_PROMPT.format(claim=state['claim'])).content
    return {"expert_brief": brief}

def supporter_node(state: FactCourtState):
    llm = get_llm(state['provider'], state['api_key'])
    arg = llm.invoke(SUPPORTER_PROMPT.format(
        claim=state['claim'], 
        ledger=json.dumps(state['provenance_ledger']),
        expert_brief=state.get('expert_brief', 'None')
    )).content
    return {"supporter_argument": arg}

def skeptic_cross_exam_node(state: FactCourtState):
    llm = get_llm(state['provider'], state['api_key'])
    arg = llm.invoke(SKEPTIC_CROSS_EXAM_PROMPT.format(
        argument=state['supporter_argument'],
        ledger=json.dumps(state['provenance_ledger'])
    )).content
    return {"skeptic_cross_exam": arg}

def skeptic_node(state: FactCourtState):
    llm = get_llm(state['provider'], state['api_key'])
    arg = llm.invoke(SKEPTIC_PROMPT.format(
        claim=state['claim'], 
        ledger=json.dumps(state['provenance_ledger']),
        expert_brief=state.get('expert_brief', 'None')
    )).content
    return {"skeptic_argument": arg}

def supporter_cross_exam_node(state: FactCourtState):
    llm = get_llm(state['provider'], state['api_key'])
    arg = llm.invoke(SUPPORTER_CROSS_EXAM_PROMPT.format(
        argument=state['skeptic_argument'],
        ledger=json.dumps(state['provenance_ledger'])
    )).content
    return {"supporter_cross_exam": arg}

def validator_node(state: FactCourtState):
    llm = get_llm(state['provider'], state['api_key'])
    validation = llm.invoke(VALIDATOR_PROMPT.format(
        ledger=json.dumps(state['provenance_ledger']),
        supporter_arg=state['supporter_argument'],
        skeptic_arg=state['skeptic_argument']
    )).content
    
    is_valid = 'yes' not in validation.lower()
    return {"is_valid_evidence": is_valid, "validation_notes": validation}

def evaluator_node(state: FactCourtState):
    llm = get_llm(state['provider'], state['api_key'])
    verdict_raw = llm.invoke(EVALUATOR_PROMPT.format(
        claim=state['claim'],
        supporter_arg=state['supporter_argument'],
        skeptic_arg=state['skeptic_argument'],
        ledger=json.dumps(state['provenance_ledger'])
    )).content
    
    return {
        "final_verdict": "Verdict: " + verdict_raw[:100], 
        "court_report": verdict_raw,
        "confidence_score": 90 # Hardcoded for simplicity right now
    }

# --- Edges Logic ---
def route_after_router(state: FactCourtState):
    if state['needs_expert']:
        return "expert_node"
    return "supporter_node"

def route_after_validator(state: FactCourtState):
    if state['is_valid_evidence'] or state['iterations'] >= 2:
        return "evaluator_node"
    return "researcher_node"

# --- Graph Assembly ---
workflow = StateGraph(FactCourtState)

workflow.add_node("researcher_node", researcher_node)
workflow.add_node("topic_router", topic_router)
workflow.add_node("expert_node", expert_node)
workflow.add_node("supporter_node", supporter_node)
workflow.add_node("skeptic_cross_exam_node", skeptic_cross_exam_node)
workflow.add_node("skeptic_node", skeptic_node)
workflow.add_node("supporter_cross_exam_node", supporter_cross_exam_node)
workflow.add_node("validator_node", validator_node)
workflow.add_node("evaluator_node", evaluator_node)

workflow.add_edge(START, "researcher_node")
workflow.add_edge("researcher_node", "topic_router")
workflow.add_conditional_edges("topic_router", route_after_router)
workflow.add_edge("expert_node", "supporter_node")
workflow.add_edge("supporter_node", "skeptic_cross_exam_node")
workflow.add_edge("skeptic_cross_exam_node", "skeptic_node")
workflow.add_edge("skeptic_node", "supporter_cross_exam_node")
workflow.add_edge("supporter_cross_exam_node", "validator_node")
workflow.add_conditional_edges("validator_node", route_after_validator)
workflow.add_edge("evaluator_node", END)

fact_court_app = workflow.compile()
