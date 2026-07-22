"""Pydantic models defining the API surface.

Declaring these as `response_model` on each route gives generated OpenAPI docs at
/docs for free -- a concrete gain over the Flask version.
"""

from pydantic import BaseModel, Field

from . import config


class HealthResponse(BaseModel):
    status: str = Field(description="'ok', or 'degraded' when the model is absent")
    model_loaded: bool
    model_error: str | None = None
    classes: list[str]
    diseases_known: list[str]
    input_size: list[int] | None = None
    confidence_threshold: float
    kb_entries: int


class DiseaseDetail(BaseModel):
    common_names: list[str]
    symptoms: list[str]
    recommendation: str
    detectable_by_image: bool = Field(
        description="True for the 3 classes the CNN can predict. The knowledge "
        "base covers 13 diseases; the image model only distinguishes 3."
    )


class DiseasesResponse(BaseModel):
    count: int
    diseases: dict[str, DiseaseDetail]


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    recommendation: str
    uncertain: bool
    margin: float
    message: str | None = None


class RecommendRequest(BaseModel):
    text: str = Field(max_length=config.MAX_TEXT_CHARS)


class RecommendationItem(BaseModel):
    disease: str
    matched_terms: list[str]
    symptoms: list[str]
    recommendation: str


class RecommendResponse(BaseModel):
    diseases: list[str]
    recommendations: list[RecommendationItem]
    ambiguous: bool = Field(
        description="True when a single matched term maps to several diseases, "
        "so the result is a candidate list rather than a diagnosis."
    )
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
