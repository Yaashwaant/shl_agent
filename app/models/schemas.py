"""
Pydantic models for API request/response schemas.
Schema is non-negotiable per assignment spec.
"""
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional, Literal


class Message(BaseModel):
    """A single conversation turn."""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """
    POST /chat request body.
    Stateless: carries full conversation history on every call.
    """
    messages: List[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def validate_turn_cap(cls, v: List[Message]) -> List[Message]:
        if len(v) > 8:
            raise ValueError("Conversation exceeds the maximum of 8 turns.")
        return v


class Recommendation(BaseModel):
    """A single SHL assessment recommendation. Schema is non-negotiable per spec."""
    name: str
    url: str
    test_type: str  # Single-letter code or comma-separated: "K", "A,K", "P" etc.


class ChatResponse(BaseModel):
    """
    POST /chat response body.
    Schema is non-negotiable per SHL spec.
    'recommendations' is null when agent is clarifying or refusing.
    'recommendations' is a list of 1-10 items when agent has committed to a shortlist.
    'end_of_conversation' is true only when the agent considers the task complete.
    """
    reply: str
    recommendations: Optional[List[Recommendation]] = None
    end_of_conversation: bool = False

    model_config = ConfigDict(
        # Ensures 'recommendations' always appears in JSON output as null,
        # never omitted — critical for evaluator schema compliance.
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "reply": "Here are 3 assessments for a mid-level Java developer.",
                "recommendations": [
                    {
                        "name": "Core Java (Advanced Level) (New)",
                        "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
                        "test_type": "K",
                    }
                ],
                "end_of_conversation": False,
            }
        }
    )


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "ok"
