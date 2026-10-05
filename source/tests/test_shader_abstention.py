"""The bid shader abstains when there is no prediction, and says so.

`triton_inference.predict_ctr` used to return a CTR of 0.5 on any Triton error. That
is a fabricated prediction, and it is indistinguishable downstream from a real one:
0.5 x the conversion-value estimate x the shade factor is a plausible price, so a
shaded bid was published on the strength of a number no model produced.

It also hides a deployment whose every inference fails: a container sending input
names the served engine does not declare shows no symptom beyond a slightly
different shaded price, with Triton's own counter at `success=0` while the
orchestrator response reads `status: ok`.

No mutation is a legitimate ARTF response: the auction proceeds at the original
price. A mutation derived from a placeholder is a wrong answer presented as a right
one. These tests pin the abstention, and the reason travelling far enough to be seen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

from shared import dlrm_features  # noqa: E402
from shared.artf_types import ContainerInvocationModel, Metadata  # noqa: E402

CONTAINER_DIR = SOURCE_DIR / "containers" / "dlrm_bid_shader"


def _load_triton_inference(client):
    """Load triton_inference.py by path with a stub Triton client installed.

    `app.py` imports it as `container.triton_inference`, a package name that only
    exists inside the built image, so it is loaded from its file here — the same
    approach tests/test_dlrm_triton_inputs.py takes.
    """
    spec = importlib.util.spec_from_file_location(
        "_ti_under_test", CONTAINER_DIR / "triton_inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # The client is cached per thread (module._local), not in a module-level
    # singleton, so an injected stub has to go where _get_client() looks for it.
    module._local.client = client
    return module


class _Raising:
    """A Triton client whose infer() fails the way an unreachable server does."""

    def __init__(self, module) -> None:
        self._module = module

    def infer(self, **_kwargs):
        raise self._module_exception()

    def _module_exception(self):
        from tritonclient.utils import InferenceServerException

        return InferenceServerException("unexpected inference input 'sparse_device'")


def _dense_and_categorical():
    dense = np.zeros((1, dlrm_features.DENSE_WIDTH), dtype=np.float32)
    categorical = [
        np.zeros((1, 1), dtype=np.int64)
        for _ in dlrm_features.TRITON_CATEGORICAL_INPUTS
    ]
    return dense, categorical


class TestPredictCtrRaisesRatherThanSubstituting:
    def test_a_triton_error_raises_inference_unavailable(self) -> None:
        module = _load_triton_inference(client=None)
        module._local.client = _Raising(module)
        dense, categorical = _dense_and_categorical()

        with pytest.raises(module.InferenceUnavailable) as exc:
            module.predict_ctr(dense=dense, categorical=categorical)

        # The server's own message survives, so the cause is diagnosable from the
        # container's log rather than only from Triton's.
        assert "sparse_device" in str(exc.value)

    def test_the_hardcoded_fallback_is_gone(self) -> None:
        """Asserted against the source as well as the behaviour: a `return 0.5` added
        back in a later edit would pass the test above if it sat on a different
        branch."""
        text = (CONTAINER_DIR / "triton_inference.py").read_text()

        assert "return 0.5" not in text
        assert "fallback CTR" not in text

    def test_inference_unavailable_is_exported_for_the_caller(self) -> None:
        module = _load_triton_inference(client=None)

        assert issubclass(module.InferenceUnavailable, RuntimeError)

    def test_an_unreachable_server_also_raises_inference_unavailable(self) -> None:
        """The gap a live run found.

        `InferenceServerException` covers errors the SERVER reported. It does not
        cover not reaching the server: with Triton scaled to zero the client raises a
        connection error, which escaped `predict_ctr`, propagated out of `mutate`, and
        reached the orchestrator as `status: error` with the reason lost. No
        fabricated shade was published — but the abstention did not fire either.
        """

        class _Unreachable:
            def infer(self, **_kwargs):
                raise ConnectionRefusedError("[Errno 61] Connection refused")

        module = _load_triton_inference(client=_Unreachable())
        dense, categorical = _dense_and_categorical()

        with pytest.raises(module.InferenceUnavailable) as exc:
            module.predict_ctr(dense=dense, categorical=categorical)

        # The exception type is carried so a server-reported error and an unreachable
        # server stay distinguishable in a log, even though they are the same fact to
        # the caller.
        assert "ConnectionRefusedError" in str(exc.value)

    def test_an_unreadable_result_also_raises(self) -> None:
        """A 200 whose payload has no ctr_prediction is not a prediction either."""

        class _Garbage:
            def infer(self, **_kwargs):
                class _R:
                    def as_numpy(self, _name):
                        return None

                return _R()

        module = _load_triton_inference(client=_Garbage())
        dense, categorical = _dense_and_categorical()

        with pytest.raises(module.InferenceUnavailable):
            module.predict_ctr(dense=dense, categorical=categorical)

    def test_a_caller_shape_error_still_raises_valueerror(self) -> None:
        """The widened catch must not swallow a programming error into "unavailable".

        A wrong number of categorical arrays is the caller disagreeing with the
        feature spec, which is a bug to fix rather than a transient absence of a
        prediction — so it is raised before the call, outside the try.
        """
        module = _load_triton_inference(client=None)
        dense, categorical = _dense_and_categorical()

        with pytest.raises(ValueError) as exc:
            module.predict_ctr(dense=dense, categorical=categorical[:-1])

        assert not isinstance(exc.value, module.InferenceUnavailable)


def _load_app(predict):
    """Load `app.py` in Triton mode with `predict` standing in for the inference call.

    It imports `container.triton_inference`, a package path that only exists inside
    the built image, so that module is registered in sys.modules first. `USE_TRITON`
    is read from the environment at import time, hence the env var.
    """
    import types

    fake_ti = types.ModuleType("container.triton_inference")

    class InferenceUnavailable(RuntimeError):
        pass

    fake_ti.InferenceUnavailable = InferenceUnavailable
    fake_ti.predict_ctr = predict

    container_pkg = types.ModuleType("container")
    container_pkg.triton_inference = fake_ti

    saved = {k: sys.modules.get(k) for k in ("container", "container.triton_inference")}
    sys.modules["container"] = container_pkg
    sys.modules["container.triton_inference"] = fake_ti
    try:
        spec = importlib.util.spec_from_file_location(
            "_app_under_test", CONTAINER_DIR / "app.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module, InferenceUnavailable
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _shade_request():
    from shared.artf_types import RTBRequest

    return RTBRequest(
        id="abstain-1",
        applicable_intents=["BID_SHADE"],
        imp=[{"id": "1", "bidfloor": 2.5}],
        site={"domain": "espn.com"},
        device={"devicetype": 2, "geo": {"country": "USA"}},
        bid_response={
            "id": "abstain-1",
            "seatbid": [{"seat": "amt", "bid": [{"id": "b1", "impid": "1", "price": 35.0}]}],
        },
    )


class TestTheContainerAbstains:
    """Driven through `mutate`, not asserted against the source.

    `app.py` is loaded by file path with a stub for `container.triton_inference`,
    because that package path only exists inside the built image.
    """

    def test_no_mutation_is_emitted_when_there_is_no_prediction(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("USE_TRITON", "true")

        holder: dict = {}

        def _predict(**_kwargs):
            raise holder["exc"]("Triton did not answer")

        module, unavailable = _load_app(_predict)
        holder["exc"] = unavailable

        response = module.mutate(_shade_request())

        assert response.mutations == [], (
            "a mutation was emitted with no prediction behind it"
        )

    def test_the_reason_is_reported_on_the_response(self, monkeypatch) -> None:
        monkeypatch.setenv("USE_TRITON", "true")

        holder: dict = {}

        def _predict(**_kwargs):
            raise holder["exc"]("unexpected inference input 'sparse_device'")

        module, unavailable = _load_app(_predict)
        holder["exc"] = unavailable

        response = module.mutate(_shade_request())

        reason = response.metadata.abstained_reason or ""
        assert reason.startswith("inference_unavailable:")
        # The server's message is carried, so the cause is readable without
        # correlating against Triton's own logs.
        assert "sparse_device" in reason

    def test_a_working_prediction_still_shades(self, monkeypatch) -> None:
        """The abstention must not have turned the shader off."""
        monkeypatch.setenv("USE_TRITON", "true")

        module, _ = _load_app(lambda **_kwargs: (0.2, "stable", "v1"))

        response = module.mutate(_shade_request())

        assert len(response.mutations) == 1
        assert response.metadata.abstained_reason is None

    def test_both_backends_define_the_exception_name(self) -> None:
        """`mutate` catches one name regardless of which backend was compiled in, so
        the in-process PyTorch path declares it too rather than the call site
        branching on USE_TRITON."""
        text = (CONTAINER_DIR / "app.py").read_text()

        assert "class InferenceUnavailable(RuntimeError):" in text


class TestTheReasonReachesTheConsumer:
    """An abstention recorded only in the container's log is a silent failure.

    `status: no_mutations` cannot distinguish a model that found no reason to act
    from one that could not obtain a prediction, so the reason has to travel:
    container metadata -> ContainerCallOutcome -> ContainerInvocationModel ->
    Metadata.containers.
    """

    def test_metadata_carries_an_abstention_reason(self) -> None:
        assert Metadata().abstained_reason is None
        assert Metadata(abstained_reason="inference_unavailable: x").abstained_reason

    def test_the_per_container_record_carries_it(self) -> None:
        record = ContainerInvocationModel(
            name="dlrm-bid-shader",
            status="no_mutations",
            latency_ms=1.0,
            abstained_reason="inference_unavailable: boom",
        )

        assert "boom" in (record.abstained_reason or "")

    def test_it_defaults_to_none_so_older_readers_are_unaffected(self) -> None:
        """None, not "", so "declined for ordinary reasons" and "declined for a
        reason it is reporting" stay distinguishable."""
        record = ContainerInvocationModel(
            name="x", status="no_mutations", latency_ms=1.0
        )

        assert record.abstained_reason is None

    def test_the_outcome_type_carries_it(self) -> None:
        from orchestrator.container_registry import ContainerCallOutcome

        assert ContainerCallOutcome(reached=True).abstained_reason is None
        assert (
            ContainerCallOutcome(reached=True, abstained_reason="r").abstained_reason
            == "r"
        )

    def test_both_transports_read_it_from_the_response(self) -> None:
        """gRPC, REST /mutate and MCP all build their outcome through one helper
        (``_outcome_from_rtb_response``), so the reason is read once and reaches the
        consumer whichever transport answered. Pinned as: the helper reads it, and
        every transport branch returns through the helper."""
        text = (SOURCE_DIR / "orchestrator" / "app.py").read_text()

        assert text.count('metadata.get("abstained_reason") or None') == 1
        # One call per transport: gRPC reply, REST 200 body, MCP content text.
        assert text.count("return _outcome_from_rtb_response(") == 3

    def test_the_timer_wrapper_passes_it_to_the_per_container_record(self) -> None:
        text = (SOURCE_DIR / "orchestrator" / "app.py").read_text()

        assert "abstained_reason=outcome.abstained_reason," in text
