from typing import TypedDict, List, Dict, Any, Optional

class FactCourtState(TypedDict):
    # Inputs
    claim: str
    image_data: Optional[str]
    provider: str 
    api_key: str 
    
    # Investigation
    precedents: List[Dict[str, Any]]
    provenance_ledger: List[Dict[str, str]]
    
    # Topic Routing
    needs_expert: bool
    expert_brief: str
    
    # Debate
    supporter_argument: str
    skeptic_cross_exam: str
    skeptic_argument: str
    supporter_cross_exam: str
    
    # Validation
    is_valid_evidence: bool
    validation_notes: str
    
    # Rulings
    confidence_score: int
    final_verdict: str
    court_report: str
    
    # Loop control
    iterations: int
