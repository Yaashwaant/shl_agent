"""
Circuit Breaker implementation for protecting external service calls.

Protects LLM (Gemini) and ChromaDB from cascading failures using the
CLOSED → OPEN → HALF_OPEN state machine pattern.
"""
import asyncio
import logging
import time
from enum import Enum
from threading import Lock
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "CLOSED"       # Normal operation — calls go through
    OPEN = "OPEN"           # Failing — calls are short-circuited immediately
    HALF_OPEN = "HALF_OPEN" # Recovery probe — one test call is allowed


class CircuitBreakerError(Exception):
    """Raised when a call is attempted while the circuit breaker is OPEN."""

    def __init__(self, service: str):
        super().__init__(f"Circuit breaker OPEN for service '{service}'. Try again later.")
        self.service = service


class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    State transitions:
        CLOSED   → calls go through normally; failures are counted
        OPEN     → calls are rejected immediately with CircuitBreakerError
        HALF_OPEN → one probe call is allowed after reset_timeout elapses;
                    success → CLOSED, failure → OPEN
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        """Return current state, transitioning OPEN → HALF_OPEN if timeout elapsed."""
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and self._last_failure_time is not None
                and time.monotonic() - self._last_failure_time >= self.reset_timeout
            ):
                logger.info(f"[CircuitBreaker:{self.name}] OPEN → HALF_OPEN (timeout elapsed)")
                self._state = CircuitState.HALF_OPEN
            return self._state

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a synchronous callable through the circuit breaker."""
        if self.state == CircuitState.OPEN:
            logger.warning(f"[CircuitBreaker:{self.name}] OPEN — rejecting call")
            raise CircuitBreakerError(self.name)
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    async def call_async(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute an async or sync callable through the circuit breaker."""
        if self.state == CircuitState.OPEN:
            logger.warning(f"[CircuitBreaker:{self.name}] OPEN — rejecting async call")
            raise CircuitBreakerError(self.name)
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"[CircuitBreaker:{self.name}] HALF_OPEN probe succeeded → CLOSED")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            logger.warning(
                f"[CircuitBreaker:{self.name}] Failure #{self._failure_count} "
                f"(threshold={self.failure_threshold})"
            )
            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    logger.error(f"[CircuitBreaker:{self.name}] Threshold reached → OPEN")
                self._state = CircuitState.OPEN

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, "
            f"state={self._state.value}, "
            f"failures={self._failure_count})"
        )


# ── Pre-configured singletons ────────────────────────────────────────────── #

llm_circuit_breaker = CircuitBreaker(
    name="gemini_llm",
    failure_threshold=5,
    reset_timeout=60.0,
)

vector_store_circuit_breaker = CircuitBreaker(
    name="chroma_vector_store",
    failure_threshold=3,
    reset_timeout=30.0,
)
