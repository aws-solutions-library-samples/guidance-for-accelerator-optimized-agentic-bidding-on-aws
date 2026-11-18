"""Keep ONNX export working across the two torch versions this repo runs on.

`torch.onnx.export` gained a `dynamo` parameter in torch 2.6. Passing it selects the
older tracing exporter explicitly, which is what the Triton config and the Model
Optimizer expect — the dynamo exporter emits a different graph. Passing it to an
earlier torch raises `TypeError: export() got an unexpected keyword argument 'dynamo'`.

Both versions are in play: the local environment and the exporter script run a recent
torch, and the NeMo-based training image runs an older one. A training job failed at
this exact line after phase 1 had already completed — the run was thrown away for a
keyword argument.

Rather than pin a version or drop the argument, ask the installed torch whether it
accepts it. Dropping it would silently switch exporter on the newer torch, which is the
thing the argument was added to prevent.
"""

from __future__ import annotations

import inspect
from typing import Any


def onnx_export_kwargs(**requested: Any) -> dict[str, Any]:
    """Filter export kwargs down to what the installed ``torch.onnx.export`` accepts.

    Only drops arguments the signature does not declare. A misspelled argument name is
    therefore dropped silently rather than raising — the trade for surviving two torch
    versions — so keep the caller's list short and deliberate.
    """
    import torch

    try:
        parameters = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-implemented signature
        return dict(requested)

    if any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return dict(requested)

    return {k: v for k, v in requested.items() if k in parameters}


def supports_dynamo_flag() -> bool:
    """Whether the installed torch accepts ``dynamo=``. For tests and logging."""
    return "dynamo" in onnx_export_kwargs(dynamo=False)
