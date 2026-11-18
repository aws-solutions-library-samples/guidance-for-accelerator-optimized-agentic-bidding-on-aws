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
- stable_version_arn   : optional, the SageMaker Model Package version/ARN
                         currently served by stable_model. Set by
                         CanaryDeployer at promote() time (control-plane).
- canary_version_arn   : optional, the SageMaker Model Package version/ARN
                         currently served by canary_model. Set by
                         CanaryDeployer at deploy_canary() time.

Optional per-request override (load-test-only, never used by real auction
traffic): a ``target_variant`` input tensor ("stable" | "canary"). This is a
Triton-level input the DLRM/NCF containers' Triton clients pass ONLY when the
orchestrator's load-test invocation set the out-of-band
``X-Load-Test-Target-Variant`` HTTP header on that request (see
source/orchestrator/loadtest_targeting.py) — no other code path constructs
this tensor, so real auction traffic can never reach this override. Providing
it is still a per-request Triton *input* the router already declares in its
own config, not an external network/S3/DynamoDB lookup, so it does not
introduce the kind of external dependency this router's ARTF-compliance
design avoids.

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
        # Version-ARN parameters: control-plane only, written by CanaryDeployer
        # (see source/deployment/model_deployer.py::set_router_split and
        # promote_engine). Empty string means "not yet known" — never fabricated.
        self.stable_version_arn = _param("stable_version_arn", "").strip()
        self.canary_version_arn = _param("canary_version_arn", "").strip()

        # Forward exactly the inputs/outputs this router declares.
        self.input_names = [i["name"] for i in self.model_config.get("input", [])]
        self.output_names = [o["name"] for o in self.model_config.get("output", [])]
        # Optional variant echo output — only produced if the router declares it.
        self._emit_variant = "served_variant" in self.output_names
        # Optional model-version echo output — only produced if declared.
        self._emit_version = "served_model_version" in self.output_names
        # Optional per-request override input — only honored if declared. Real
        # auction traffic never supplies this tensor (see module docstring).
        self._accept_target_override = "target_variant" in self.input_names
        _echo_outputs = {"served_variant", "served_model_version"}
        self._sub_output_names = [n for n in self.output_names if n not in _echo_outputs]
        self._sub_input_names = [n for n in self.input_names if n != "target_variant"]

        if not self.stable_model:
            raise pb_utils.TritonModelException(
                "canary router requires a 'stable_model' config parameter"
            )

    def _version_arn_for(self, variant: str) -> str:
        """Return the version-ARN config parameter for the picked variant.

        Empty string ("") if not yet set — this is a real "unknown" state
        (e.g. router deployed before CanaryDeployer ever wrote a version),
        not a fabricated placeholder.
        """
        return self.canary_version_arn if variant == "canary" else self.stable_version_arn

    def _pick_target(self, override: str | None = None):
        """Return (target_model_name, variant).

        ``override``, when "stable" or "canary", forces that variant for THIS
        request only — used exclusively by the load-test-only per-request
        input tensor (see module docstring). Any other value (including None)
        falls through to the normal randomized split, which is a legitimate
        real traffic split (not fabricated data).
        """
        if override == "canary" and self.canary_model:
            return self.canary_model, "canary"
        if override == "stable":
            return self.stable_model, "stable"

        if (
            self.canary_model
            and self.canary_traffic_pct > 0.0
            and (random.random() * 100.0) < self.canary_traffic_pct
        ):
            return self.canary_model, "canary"
        return self.stable_model, "stable"

    def _read_target_override(self, request) -> str | None:
        """Read the optional per-request target_variant input, if declared.

        Returns None if the router doesn't declare this input, the request
        didn't supply it, or the value isn't "stable"/"canary" — in all of
        those cases _pick_target falls through to the normal random split.
        """
        if not self._accept_target_override:
            return None
        tensor = pb_utils.get_input_tensor_by_name(request, "target_variant")
        if tensor is None:
            return None
        try:
            raw = tensor.as_numpy().flat[0]
            value = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        except Exception:
            return None
        return value if value in ("stable", "canary") else None

    def execute(self, requests):
        responses = []
        for request in requests:
            override = self._read_target_override(request)
            target_model, variant = self._pick_target(override)

            inputs = []
            for name in self._sub_input_names:
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
            if self._emit_variant or self._emit_version:
                import numpy as np

                if self._emit_variant:
                    out_tensors.append(
                        pb_utils.Tensor(
                            "served_variant",
                            np.array([variant.encode("utf-8")], dtype=object),
                        )
                    )
                if self._emit_version:
                    # Real "unknown" (empty string) if CanaryDeployer hasn't
                    # written a version-ARN yet — never fabricated.
                    version_arn = self._version_arn_for(variant)
                    out_tensors.append(
                        pb_utils.Tensor(
                            "served_model_version",
                            np.array([version_arn.encode("utf-8")], dtype=object),
                        )
                    )
            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
        return responses
