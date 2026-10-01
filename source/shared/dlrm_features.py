"""The DLRM bid shader's feature vector — one definition, three consumers.

`dlrm_bid_shader` builds this vector from a live bid request; the SageMaker
training container (`source/training/container/train.py`) rebuilds it from
stored outcome rows; the Glue ETL job
(`source/etl/glue_feature_engineering.py`) produces the columns it reads from.
This module is the single definition of the vector, for the same reason
`yield_features.py` is for the yield containers: a per-consumer copy lets
training drift from serving, and the drift is silent. Widths match, dtypes
match, nothing raises — the model simply learns one meaning for a position and
is served another.

No I/O, no inference calls, no randomness. Every value derives from a field on
the bid request, from a stored outcome column, or from the request timestamp.

Vector layout (fixed, DENSE_WIDTH + CATEGORICAL_WIDTH, never varies with input
completeness):

    continuous
      [0] bid_floor    raw float, the impression's floor
      [1] hour_norm    UTC hour / 24, [0, 1)
      [2] is_weekend   0.0 / 1.0, from UTC weekday
      [3] has_video    0.0 / 1.0, from imp.video presence

    categorical (embedding indices)
      [4] site_domain
      [5] device_type
      [6] geo_country

Each categorical hashes into its own vocabulary size (VOCAB_SIZES) rather than
a single shared constant, so the collision rate of each is a chosen property.

Held out by design:
  - `shade_factor_used`, `conversion_value_estimate_used` — the shader's own
    parameters. A prediction conditioned on the policy that generated its
    training data cannot be used to evaluate a change to that policy.
  - `shaded_price`, `roi`, `price_paid` — consequences of the bid.
  - raw `user.id`, `device.ua` — little signal once hashed into a bounded
    table, and a privacy obligation the signal does not justify. Audience
    membership belongs here as a segment, not as an individual.

CHANGING THIS VECTOR IS A BREAKING CHANGE. Bump FEATURE_SPEC_VERSION, re-export
the ONNX signature (`source/triton/export_models.py` and the model's
`config.pbtxt`), and retrain — embedding tables are keyed to this vocabulary,
so previous weights do not carry over.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

#: Incremented on any change to the layout, the normalisation, or the hash.
#: A trained model records this in its manifest; a serving build refuses a
#: model whose recorded version it does not recognise, because matching widths
#: are not evidence that both sides mean the same thing by position two.
FEATURE_SPEC_VERSION = 1

#: Versions a build produced from THIS source can serve. Currently the one it
#: emits, and nothing else -- there is no back-compatibility shim, because an
#: older vector would need its own normalisation and its own vocabularies to be
#: interpreted, not just a narrower slice of this one.
#:
#: This exists so "recognised" has a single definition. A promotion path that
#: hardcoded `== FEATURE_SPEC_VERSION` would have to be edited in every place it
#: appears the first time two versions are legitimately servable.
SUPPORTED_FEATURE_SPEC_VERSIONS: frozenset[int] = frozenset({FEATURE_SPEC_VERSION})

#: Filename a producer writes beside its exported model, and a promotion path
#: reads before compiling it. Named here so the trainer, the genesis exporter and
#: the optimizer cannot disagree about it.
MANIFEST_FILENAME = "manifest.json"


def manifest(model_type: str, **extra: Any) -> dict[str, Any]:
    """The provenance record a producer writes beside an exported model.

    Carries the feature-spec version and the vector it describes, so a promotion
    path can refuse a model it cannot interpret instead of compiling an engine
    whose inputs happen to be the right width. `extra` is merged in for
    producer-specific fields (the training objective, the label column, the base
    model version) without this module needing to know about them.
    """
    record: dict[str, Any] = {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "model_type": model_type,
        "dense_columns": list(DENSE_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "vocab_sizes": dict(VOCAB_SIZES),
        "triton_input_names": [TRITON_DENSE_INPUT, *TRITON_CATEGORICAL_INPUTS],
    }
    record.update(extra)
    return record

#: Continuous features, in vector order. These names are the ETL column names,
#: so the training side selects by this list directly.
DENSE_COLUMNS: tuple[str, ...] = (
    "bid_floor",
    "hour_norm",
    "is_weekend",
    "has_video",
)

#: Categorical features, in vector order. Also ETL column names.
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "site_domain",
    "device_type",
    "geo_country",
)

#: Per-feature vocabulary size. Sized to the cardinality each feature actually
#: has: a domain space is large, a device-type space is small, and the set of
#: country codes is bounded and known. One shared constant would either waste
#: table for the small features or collide heavily on the large one.
VOCAB_SIZES: Mapping[str, int] = {
    "site_domain": 10_000,
    "device_type": 32,
    "geo_country": 256,
}

DENSE_WIDTH = len(DENSE_COLUMNS)
CATEGORICAL_WIDTH = len(CATEGORICAL_COLUMNS)
FEATURE_WIDTH = DENSE_WIDTH + CATEGORICAL_WIDTH

#: The serving signature's input names. Declared here so the container
#: (`dlrm_bid_shader/app.py`), the exporter (`source/triton/export_models.py`)
#: and the model's `config.pbtxt` cannot disagree about what an input is called.
#:
#: The categorical inputs are named after the features they carry. The previous
#: names -- `sparse_user`, `sparse_domain`, `sparse_device` -- described a
#: different set, and an input name that describes the wrong feature is the
#: same class of problem as a position that means the wrong thing.
TRITON_DENSE_INPUT = "dense_features"
TRITON_CATEGORICAL_INPUTS: tuple[str, ...] = tuple(
    f"sparse_{column}" for column in CATEGORICAL_COLUMNS
)

#: Returned for a categorical whose value is absent. Slot 0 is reserved for it
#: in every vocabulary, so "missing" is a value the model can learn rather than
#: an arbitrary domain sharing its embedding.
UNKNOWN_INDEX = 0


def feature_names() -> tuple[str, ...]:
    """The full vector's column names, in order."""
    return DENSE_COLUMNS + CATEGORICAL_COLUMNS


def hash_to_idx(value: Any, feature: str) -> int:
    """Hash a categorical value into `feature`'s vocabulary.

    Index 0 is reserved for absent values, so real values occupy
    [1, VOCAB_SIZES[feature]). Both the serving path and the training path call
    this, which is what makes a given domain land in the same embedding slot
    whichever side computed it.
    """
    vocab = VOCAB_SIZES[feature]
    if value is None:
        return UNKNOWN_INDEX
    text = str(value).strip().lower()
    if not text:
        return UNKNOWN_INDEX
    digest = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()  # nosec B324
    return 1 + (int(digest, 16) % (vocab - 1))


# ---------------------------------------------------------------------------
# Normalisation — one implementation, called by both build functions
# ---------------------------------------------------------------------------

def normalise_hour(hour_of_day: int | float | None) -> float:
    """UTC hour to [0, 1). Out-of-range or absent encodes to 0.0."""
    if hour_of_day is None:
        return 0.0
    try:
        hour = int(hour_of_day)
    except (TypeError, ValueError):
        return 0.0
    if hour < 0 or hour > 23:
        return 0.0
    return hour / 24.0


def weekend_flag(day_of_week: int | float | None) -> float:
    """1.0 for Saturday or Sunday, else 0.0.

    `day_of_week` follows `datetime.weekday()`: Monday is 0, Sunday is 6.
    """
    if day_of_week is None:
        return 0.0
    try:
        day = int(day_of_week)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if day >= 5 else 0.0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# The two build paths
# ---------------------------------------------------------------------------

def build_from_bid_request(
    bid_request: Mapping[str, Any],
    imp: Mapping[str, Any] | None = None,
    request_time: datetime | None = None,
) -> tuple[list[float], list[int]]:
    """Serving path: build the vector from a live OpenRTB bid request.

    Args:
        bid_request: The full bid request.
        imp: The impression being priced. Defaults to the first entry of
            `imp[]`, which is what a single-impression request has.
        request_time: Real wall-clock time. Defaults to now in UTC; tests pass
            a fixed value.

    Returns:
        `(dense, categorical)` — DENSE_WIDTH floats and CATEGORICAL_WIDTH
        embedding indices. Never raises; an absent field encodes to a defined
        neutral value.
    """
    if request_time is None:
        request_time = datetime.now(timezone.utc)

    if imp is None:
        imps = bid_request.get("imp") or [{}]
        imp = imps[0] if imps else {}

    site = bid_request.get("site") or bid_request.get("app") or {}
    device = bid_request.get("device") or {}
    geo = device.get("geo") or {}

    dense = [
        _as_float(imp.get("bidfloor", 0.0)),
        normalise_hour(request_time.hour),
        weekend_flag(request_time.weekday()),
        1.0 if imp.get("video") else 0.0,
    ]
    categorical = [
        hash_to_idx(site.get("domain"), "site_domain"),
        hash_to_idx(device.get("devicetype"), "device_type"),
        hash_to_idx(geo.get("country"), "geo_country"),
    ]
    return dense, categorical


def build_from_row(row: Mapping[str, Any]) -> tuple[list[float], list[int]]:
    """Training path: build the vector from one stored outcome row.

    `row` is a mapping of ETL column names — a pandas Series, a dict, or a
    Spark Row's `asDict()`. The column names are DENSE_COLUMNS and
    CATEGORICAL_COLUMNS, except that `hour_norm` and `is_weekend` are derived
    from the stored `hour_of_day` and `day_of_week` rather than stored
    pre-normalised, so the normalisation itself lives only here.

    Returns the same `(dense, categorical)` pair `build_from_bid_request`
    returns, for the same inputs.
    """
    dense = [
        _as_float(row.get("bid_floor", 0.0)),
        normalise_hour(row.get("hour_of_day")),
        weekend_flag(row.get("day_of_week")),
        1.0 if row.get("has_video") else 0.0,
    ]
    categorical = [
        hash_to_idx(row.get("site_domain"), "site_domain"),
        hash_to_idx(row.get("device_type"), "device_type"),
        hash_to_idx(row.get("geo_country"), "geo_country"),
    ]
    return dense, categorical


def flatten(dense: Sequence[float], categorical: Sequence[int]) -> list[float]:
    """The single-tensor form: dense floats followed by categorical indices.

    The training model takes one width-FEATURE_WIDTH tensor and splits it
    internally; the serving signature takes the two parts as separate named
    inputs. Both orderings come from here so they cannot disagree.
    """
    return [float(x) for x in dense] + [float(i) for i in categorical]
