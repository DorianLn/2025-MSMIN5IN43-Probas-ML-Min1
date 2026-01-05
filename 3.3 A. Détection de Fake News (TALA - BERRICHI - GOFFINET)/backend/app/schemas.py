from pydantic import BaseModel
from typing import Dict

class AnalysisRequest(BaseModel):
    text: str
    model: str  # 'camembert', 'bert', ou 'roberta'

class FactorDetail(BaseModel):
    score: float
    label: str

class AnalysisResponse(BaseModel):
    isReliable: bool
    confidence: float
    factors: Dict[str, FactorDetail]
    summary: str