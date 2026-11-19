"""Evaluations: guardrail precision/recall, retrieval hit rate, and the validation scenarios (DeepEval)."""

import os

# DeepEval sends usage telemetry unless told not to; a security tool should not phone home by default.
# Set DEEPEVAL_TELEMETRY_OPT_OUT=0 to allow it.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
