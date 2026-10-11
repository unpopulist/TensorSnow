class RequestError(ValueError):
    """A request refusal: HTTP 400, or a named error after a stream starts."""


class CapacityError(RequestError):
    """A transient capacity refusal: HTTP 429 + Retry-After (see admission_response)."""


class RoundError(RuntimeError):
    """A failed round's type and message, without its engine frames or exception payloads."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class ContextLengthError(RequestError):
    """A prompt and its reply past the context window: OpenAI's context_length_exceeded, which clients compact on."""

    code = "context_length_exceeded"


CONTEXT_LIMIT = "This server's maximum context length is"      # OpenAI's wording, which clients match to compact


def admission_response(exc: Exception) -> tuple[int, dict[str, str] | None]:
    """(status, headers) of a refusal on every API route: a capacity refusal is 429 with ``Retry-After`` (not a bare
    503), so the same client backs off instead of retrying at once; anything else is 400. A deliberate difference
    from vLLM, whose 503 lets a load balancer route to another instance: one server has no other to route to
    (Mia's 0081)."""

    return (429, {"Retry-After": "5"}) if isinstance(exc, CapacityError) else (400, None)


def refusal(problem: str) -> RequestError:
    """A refusal string as its error: a context-window one carries OpenAI's code."""

    return (ContextLengthError if problem.startswith(CONTEXT_LIMIT) else RequestError)(problem)


def error_body(exc: Exception, param: str | None = None) -> dict:
    """OpenAI's error object: the message, its type, and where there is a code clients key on, the field and code
    (``param``: ``messages`` for a chat completion, ``prompt`` for a completion, as OpenAI names them)."""

    body = {"message": str(exc), "type": "invalid_request_error"}
    if getattr(exc, "code", None):
        body["param"] = param
        body["code"] = exc.code
    return body
