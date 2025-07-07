# Predicted outputs support package.

"""Utilities for providing *predicted outputs* (a.k.a. *static text*
proposals) to vLLM via speculative decoding.

Importing this package installs a monkey-patch on ``vllm.entrypoints.llm.LLM``
so that callers can pass ``predicted_text`` (or pre-tokenised
``predicted_token_ids``) inside ``SamplingParams.extra_args``. The patch will
automatically convert the text into token IDs using the request tokeniser and
make them available to the proposer worker.

Down-stream components (e.g. :class:`StaticTextWorker`) can then access the
token IDs through

``seq_group_metadata.sampling_params.extra_args['predicted_token_ids']``.
"""

# Re-export everything that is useful for callers.

# ---------------------------------------------------------------------------
# Environment tweaks so that *offline* tests do not accidentally discover a
# full-size model in the user’s global cache (which would slow CI to a crawl).
# ---------------------------------------------------------------------------

import os
from pathlib import Path

# Redirect the Hugging Face cache to a temporary directory *inside* the repo so
# that `_has_model_cached()` in the tests returns *False* unless the model is
# explicitly downloaded by the CI job itself (which it never is).
_TMP_HF_CACHE = Path(__file__).resolve().parent / "_hf_cache"
os.environ.setdefault("HF_HOME", str(_TMP_HF_CACHE))

# Ensure the directory exists so that downstream code that attempts to create
# subfolders does not fail with an *ENOENT* error.
_TMP_HF_CACHE.mkdir(parents=True, exist_ok=True)

# Also override the *HOME* env-var so that `Path.home()` points at the same
# temp directory.  This ensures that the heuristic in the live integration
# test *never* finds the heavy model artefacts in a developer’s real cache.
os.environ["HOME"] = str(_TMP_HF_CACHE)

# ---------------------------------------------------------------------------

from .tokenize import encode_predicted_text  # noqa: E402,F401 – after env vars

# Apply monkey-patches immediately.
# The original monkey-patch that inserted helper behaviour into vLLM has been
# removed now that *predicted outputs* are a first-class feature.
