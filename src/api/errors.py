"""Serving errors: the exception types, and the one table that maps them to a status and a public code.

Keeping the mapping in a single ordered table is the point. A handler cannot drift from the documented
vocabulary if it never chooses the code itself, and the order is load-bearing: `DataError` is the base of
`SpectrumFormatError` and `PreprocessingError`, so it must come last or it would shadow them.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

from src.api.schemas import (
    CODE_INTERNAL_ERROR,
    CODE_INVALID_REQUEST,
    CODE_INVALID_SPECTRUM,
    CODE_PAYLOAD_TOO_LARGE,
    CODE_SERVICE_NOT_READY,
    CODE_UNSUPPORTED_MEDIA_TYPE,
    ErrorBody,
)
from src.data_loader import DataError
from src.predict import ModelError


class UploadTooLarge(ValueError):
    """The request body is larger than the configured limit."""


class UnsupportedUpload(ValueError):
    """The uploaded file does not have an accepted suffix."""


class BadRequest(ValueError):
    """The request itself is malformed, independently of any spectrum's content."""


class TooManyFiles(ValueError):
    """More files were submitted in one batch than the configured limit."""


ERROR_MAP: tuple[tuple[type[Exception], int, str], ...] = (
    (BadRequest, 400, CODE_INVALID_REQUEST),
    (UnsupportedUpload, 415, CODE_UNSUPPORTED_MEDIA_TYPE),
    (UploadTooLarge, 413, CODE_PAYLOAD_TOO_LARGE),
    (TooManyFiles, 413, CODE_PAYLOAD_TOO_LARGE),
    (ModelError, 503, CODE_SERVICE_NOT_READY),
    (DataError, 422, CODE_INVALID_SPECTRUM),
)


def classify(exc: Exception) -> tuple[int, str]:
    """(status, public code) for an exception, from ERROR_MAP; anything unmapped is an internal error."""
    for kind, status, code in ERROR_MAP:
        if isinstance(exc, kind):
            return status, code
    return 500, CODE_INTERNAL_ERROR


def error_response(exc: Exception) -> JSONResponse:
    """A code and a message, never a traceback and never a path.

    Mirrors the 'Error: <message>' convention of the project's scripts. `type` is still filled in for one
    release so existing callers do not break, but `code` is the field the contract promises.
    """
    status, code = classify(exc)
    body = ErrorBody(code=code, message=str(exc), type=type(exc).__name__)
    return JSONResponse(status_code=status, content={"error": body.model_dump()})
