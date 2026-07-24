"""Triton Python-backend canary ROUTER (Option D live canary split).

This model is deployed under the SAME name the ARTF container already calls
(e.g. ``dlrm_bid_shader``). The ARTF container is byte-for-byte unchanged — it
still calls Triton by model name. This router forwards each request to either the
``<model>_stable`` or ``<model>_canary`` TensorRT model and returns the result
unchanged.

ARTF compliance (no external dependency in the bidstream):
- The canary split percentage and the stable/canary target model names are read
  ONCE from this model's config ``parameters`` at load time (``initialize``).
  Changing the split is a control-plane action (edit config + Triton reload),
  never a per-request external lookup. ``execute`` does ZERO network/S3/DynamoDB
  I/O — it only draws a random number and dispatches an in-process BLS call.

Generic: the router forwards exactly the inputs/outputs declared in its own
config.pbtxt, so a single model.py works for DLRM, NCF, and Wide&Deep routers.

Config parameters (all strings):
- stable_model         : required, e.g. "dlrm_bid_shader_stable"
- canary_model         : optional, e.g. "dlrm_bid_shader_canary"
- canary_traffic_pct   : optional float 0..100 (default "0")

Batching: this router uses max_batch_size 0 / no dynamic batching (per-request
routing). GPU batching still happens in the stable/canary TensorRT models, which
keep their own dynamic_batching — Triton coalesces the concurrent BLS calls.
"""

import json
import random

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {}) or {}

        def _param(key, default):
            entry = params.get(key)
            if isinstance(entry, dict):
                return entry.get("string_value", default)
            return default

        self.stable_model = _param("stable_model", "").strip()
        self.canary_model = _param("canary_model", "").strip()
        try:
            self.canary_traffic_pct = float(_param("canary_traffic_pct", "0"))
        except (TypeError, ValueError):
            self.canary_traffic_pct = 0.0
        self.canary_traffic_pct = max(0.0, min(100.0, self.canary_traffic_pct))

        # Forward exactly the inputs/outputs this router declares.
        self.input_names = [i["name"] for i in self.model_config.get("input", [])]
        self.output_names = [o["name"] for o in self.model_config.get("output", [])]
        # Optional variant echo output — only produced if the router declares it.
        self._emit_variant = "served_variant" in self.output_names
        self._sub_output_names = [n for n in self.output_names if n != "served_variant"]

        if not self.stable_model:
            raise pb_utils.TritonModelException(
                "canary router requires a 'stable_model' config parameter"
            )

    def _pick_target(self):
        """Return (target_model_name, variant). Randomized canary assignment is a
        legitimate real traffic split (not fabricated data)."""
        if (
            self.canary_model
            and self.canary_traffic_pct > 0.0
            and (random.random() * 100.0) < self.canary_traffic_pct
        ):
            return self.canary_model, "canary"
        return self.stable_model, "stable"

    def execute(self, requests):
        responses = []
        for request in requests:
            target_model, variant = self._pick_target()

            inputs = []
            for name in self.input_names:
                tensor = pb_utils.get_input_tensor_by_name(request, name)
                if tensor is not None:
                    inputs.append(tensor)

            infer_request = pb_utils.InferenceRequest(
                model_name=target_model,
                requested_output_names=self._sub_output_names,
                inputs=inputs,
            )
            infer_response = infer_request.exec()

            if infer_response.has_error():
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            f"canary router -> {target_model}: {infer_response.error().message()}"
                        ),
                    )
                )
                continue

            out_tensors = list(infer_response.output_tensors())
            if self._emit_variant:
                import numpy as np

                out_tensors.append(
                    pb_utils.Tensor(
                        "served_variant",
                        np.array([variant.encode("utf-8")], dtype=object),
                    )
                )
            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
        return responses
