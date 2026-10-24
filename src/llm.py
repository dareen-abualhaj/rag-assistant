"""Groq (OpenAI-compatible) chat client with timeouts, retries and clear errors."""
import logging
import os
import time
from typing import Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

load_dotenv()
logger = logging.getLogger(__name__)

BASE_URL = "https://api.groq.com/openai/v1"
TIMEOUT_S = 30.0
MAX_ATTEMPTS = 4        # 1 try + 3 retries
BACKOFF_BASE_S = 2.0    # waits 2s, 4s, 8s
MAX_WAIT_S = 30.0


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMConfigError(LLMError):
    """Missing/invalid API key or wrong model id (not retryable)."""


class LLMContextTooLongError(LLMError):
    """Prompt exceeded the model's token limit."""


class LLMUnavailableError(LLMError):
    """Rate limit / timeout / server error persisted after all retries."""


def _get_client() -> OpenAI:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMConfigError("GROQ_API_KEY is missing. Add it to your .env file.")
    # max_retries=0: we implement our own retry policy below
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=TIMEOUT_S, max_retries=0)


def _wait_time(exc: Exception, attempt: int) -> float:
    """Honour the server's retry-after header if present, else exponential backoff."""
    response = getattr(exc, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return min(float(header), MAX_WAIT_S)
            except ValueError:
                pass
    return min(BACKOFF_BASE_S ** attempt, MAX_WAIT_S)


def generate(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    client: Optional[OpenAI] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Call the chat API and return the answer text.

    Retries on rate limits (429), timeouts, connection errors and 5xx.
    Raises a clear LLMError subclass for everything else.
    """
    model = model or os.getenv("LLM_MODEL")
    if not model:
        raise LLMConfigError("LLM_MODEL is missing. Set it in your .env file.")
    client = client or _get_client()

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content.strip()
            last_error = LLMError("The model returned an empty response.")
        except AuthenticationError as e:
            raise LLMConfigError("Invalid API key (401). Check GROQ_API_KEY in .env.") from e
        except NotFoundError as e:
            raise LLMConfigError(
                f"Model '{model}' not found (404). Free model IDs change often: "
                "check console.groq.com/docs/models and update LLM_MODEL in .env."
            ) from e
        except BadRequestError as e:
            msg = str(e).lower()
            if "token" in msg or "context" in msg or "length" in msg:
                raise LLMContextTooLongError(f"Prompt too long for the model: {e}") from e
            raise LLMError(f"Bad request (400): {e}") from e
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_error = e
        except APIStatusError as e:
            if e.status_code >= 500:
                last_error = e
            else:
                raise LLMError(f"API error {e.status_code}: {e}") from e

        if attempt < MAX_ATTEMPTS:
            wait = _wait_time(last_error, attempt)
            logger.warning(
                "LLM call failed (%s), retry %d/%d in %.1fs",
                type(last_error).__name__, attempt, MAX_ATTEMPTS - 1, wait,
            )
            sleep(wait)

    raise LLMUnavailableError(
        f"LLM unavailable after {MAX_ATTEMPTS} attempts "
        f"(last error: {type(last_error).__name__})."
    )
