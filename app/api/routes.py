"""
FastAPI router for the chat and health endpoints.
"""
import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from app.services.agent import get_agent, AgentState
from app.core.circuit_breaker import CircuitBreakerError

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30.0

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service readiness. First call may take up to 2 minutes on cold start.",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """GET /health — returns {status: ok} when service is ready."""
    return HealthResponse(status="ok")


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational assessment advisor",
    description=(
        "Stateless endpoint. Carries full conversation history on every call. "
        "Returns agent reply + optional structured recommendations."
    ),
    tags=["Chat"],
)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """
    POST /chat — main conversational endpoint.

    Stateless: accepts full conversation history, returns next agent reply
    plus structured recommendations (null when clarifying/refusing,
    1–10 items when committed to a shortlist).
    """
    start_time = time.perf_counter()

    # Turn cap is enforced by the ChatRequest Pydantic validator (422 if exceeded).
    # The validator already rejects >8 messages before this handler runs.

    logger.info(
        f"Chat request: {len(request.messages)} message(s), "
        f"last role={request.messages[-1].role!r}"
    )

    # Convert messages to dict format for agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Initialize agent state
    initial_state: AgentState = {
        "messages": messages,
        "intent": "",
        "retrieved_items": [],
        "recommendations": None,
        "reply": "",
        "end_of_conversation": False,
        "clarification_needed": False,
        "extracted_context": {},
    }

    async def _run_agent():
        agent = get_agent()
        return await agent.ainvoke(initial_state)

    try:
        final_state = await asyncio.wait_for(_run_agent(), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {REQUEST_TIMEOUT}s")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try again or simplify your query.",
        )
    except CircuitBreakerError as e:
        logger.error(f"Circuit breaker tripped: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please try again in a moment.",
        )
    except Exception as e:
        logger.exception(f"Agent error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal agent error. Please try again.",
        )

    # Build recommendations — null when clarifying/refusing, 1–10 items when committed.
    raw_recs = final_state.get("recommendations")
    recommendations = None

    if raw_recs:  # Only process when the agent committed to a shortlist
        validated_recs = []
        for rec in raw_recs[:10]:  # Hard cap at 10 per spec
            try:
                r = Recommendation(
                    name=rec.get("name", ""),
                    url=rec.get("url", ""),
                    test_type=rec.get("test_type", ""),
                )
                # Only include if all required fields are non-empty
                if r.name and r.url and r.test_type:
                    validated_recs.append(r)
                else:
                    logger.warning(f"Skipping incomplete recommendation (missing required fields): {rec}")
            except Exception as e:
                logger.warning(f"Skipping malformed recommendation: {rec} — {e}")

        # If validation produced nothing, keep null — don't return an empty array
        recommendations = validated_recs if validated_recs else None

    response = ChatResponse(
        reply=final_state.get("reply", "I'm having trouble responding. Please try again."),
        recommendations=recommendations,
        end_of_conversation=final_state.get("end_of_conversation", False),
    )

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"Chat response: intent={final_state.get('intent')!r}, "
        f"recs={len(recommendations) if recommendations else 0}, "
        f"eoc={response.end_of_conversation}, "
        f"elapsed={elapsed:.0f}ms"
    )

    return response
