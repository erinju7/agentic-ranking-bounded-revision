from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retriever import get_top_k_calls


app = FastAPI(
    title="P58 Funding Call Retrieval API",
    version="0.1.0",
    description="Returns top candidate funding calls for Copilot Studio reranking.",
)


class RetrievalRequest(BaseModel):
    project_objective: str = Field(..., min_length=1)
    project_title: str = ""
    project_keywords: List[str] = Field(default_factory=list)
    totalCost_eur: Optional[float] = None
    ecMaxContribution_eur: Optional[float] = None
    duration_months: Optional[float] = None
    top_k: int = Field(default=50, ge=1, le=200)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve_candidate_calls")
def retrieve_candidate_calls(req: RetrievalRequest) -> Dict[str, List[Dict[str, Any]]]:
    results = get_top_k_calls(
        project_objective=req.project_objective,
        project_title=req.project_title,
        project_keywords=req.project_keywords,
        total_cost_eur=req.totalCost_eur,
        ec_contribution_eur=req.ecMaxContribution_eur,
        duration_months=req.duration_months,
        k=req.top_k,
    )
    return {
        "candidates": [
            {
                "funding_call_id": row["funding_call_id"],
                "funding_call_description": row["funding_call_description"],
                "historical_profile_summary": row["historical_profile_summary"],
                "retriever_score": row["score"],
                "evidence_scores": {
                    "call_description": row["call_description_score"],
                    "historical_objectives": row["historical_objectives_score"],
                    "keywords": row["keywords_score"],
                    "budget": row["budget_score"],
                    "duration": row["duration_score"],
                },
                "n_historical_projects": row["n_projects"],
                "median_totalCost_eur": row["median_totalCost_eur"],
                "median_ecMaxContribution_eur": row["median_ecMaxContribution_eur"],
                "median_duration_months": row["median_duration_months"],
            }
            for row in results
        ]
    }
