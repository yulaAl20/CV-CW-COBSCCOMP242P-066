from __future__ import annotations

from pydantic import BaseModel, Field


class GradeProbability(BaseModel):
    grade: int
    name: str
    probability: float


class PredictionOut(BaseModel):
    grade: int
    grade_name: str
    grade_description: str
    expected_grade: float = Field(description="Continuous severity between 0 and 4")
    grade_probabilities: list[GradeProbability]
    confidence: float
    uncertainty: float
    entropy: float
    probability_ungradable: float
    probability_any_dr: float
    probability_referable: float
    mc_samples: int
    inference_ms: float


class TriageOut(BaseModel):
    action: str
    label: str
    reason: str
    urgency: str
    rule: int


class ImagesOut(BaseModel):

    original: str
    preprocessed: str
    heatmap: str | None = None
    stages: dict[str, str] | None = None


class AnalysisOut(BaseModel):
    request_id: str
    filename: str
    demo_mode: bool = False
    warnings: list[str] = []
    prediction: PredictionOut
    triage: TriageOut
    images: ImagesOut
    timing_ms: dict[str, float]


class HealthOut(BaseModel):
    status: str
    version: str
    model: dict
    demo_mode: bool
    thresholds: dict
