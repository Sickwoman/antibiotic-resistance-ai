"""The Version 0.9 serving package: a thin HTTP layer over the frozen inference path of src.predict.

Module responsibilities, kept separate on purpose:

- `app`                — application construction, lifespan wiring, routes, exception handlers. No
                         inference logic.
- `inference_service`  — loading the bundle and zones, readiness, and the prediction path.
- `metadata`           — the explicit allow-list behind `GET /model-info`.
- `errors`             — the exception types and the single table mapping them to a status and a code.
- `schemas`            — the declared request/response models.

**Monkeypatching note.** The names re-exported below are bindings in *this* module, so patching
`src.api.<name>` rebinds only the alias and the defining module goes on using its own. A test that needs to
intercept `predict_spectrum_file` or `TemporaryDirectory` must patch `src.api.inference_service`, which is
the module that calls them. This bit us once: patching a re-exported alias fails silently, so the test keeps
passing while testing nothing.
"""

from src.api.app import API_VERSION, create_app
from src.api.errors import (
    ERROR_MAP,
    BadRequest,
    TooManyFiles,
    UnsupportedUpload,
    UploadTooLarge,
    classify,
    error_response,
)
from src.api.inference_service import (
    DETAIL_MODEL_UNAVAILABLE,
    DETAIL_ZONES_UNUSABLE,
    Limits,
    Service,
    check_suffix,
    default_model_path,
    default_zones_path,
    load_artifacts,
    stream_to_file,
)
from src.api.metadata import PROJECT_MODEL_VERSION, model_info

__all__ = [
    "API_VERSION",
    "DETAIL_MODEL_UNAVAILABLE",
    "DETAIL_ZONES_UNUSABLE",
    "ERROR_MAP",
    "PROJECT_MODEL_VERSION",
    "BadRequest",
    "Limits",
    "Service",
    "TooManyFiles",
    "UnsupportedUpload",
    "UploadTooLarge",
    "check_suffix",
    "classify",
    "create_app",
    "default_model_path",
    "default_zones_path",
    "error_response",
    "load_artifacts",
    "model_info",
    "stream_to_file",
]
