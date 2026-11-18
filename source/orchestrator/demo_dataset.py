"""Demo training dataset generator (FR1).

Generates a synthetic, seeded training dataset for one trainable model type,
checks it with the same rules the trainer applies, and stores it in its own S3
prefix beside the real training prefixes. The data is for demonstration only:
the load-test engagement pool holds no conversions, so without this dataset a
DLRM training run has nothing to learn a conversion probability from.

Storage, in ``TRAINING_DATA_BUCKET`` (one dataset per model type; each
generation replaces the previous one):

- ``demo-training-data/`` for ``dlrm_bid_shader``
- ``demo-training-data-deal-yield-floor/`` for ``deal_yield_manager_floor``
- ``demo-training-data-deal-yield-margin/`` for ``deal_yield_manager_margin``

Each prefix holds ``part-00000.parquet`` and ``_demo_metadata.json``. The
metadata records the parameters, seed, row count, expected positives and the
pre-write check report, and is what the Governance page shows.

Row shapes follow the existing labelled training data so the trainers read
them without a code path of their own:

- DLRM rows carry the columns ``etl/glue_feature_engineering.py`` writes to
  ``training-data/``: the ``raw_bid_outcomes`` columns plus ``roi``,
  ``shade_ratio``, ``label``, ``label_conversion``, ``label_click`` and
  ``win_rate_bucket``. ``source`` is ``"demo_dataset"`` and
  ``outcome_provenance`` is ``"simulated"``.
- Deal yield rows are ``label`` followed by the 7 ``FEATURE_COLUMNS`` of
  ``etl/glue_deal_yield_feature_engineering.py``. The built-in XGBoost
  container treats every column after the first as a feature, so the
  provenance lives in the metadata object and the prefix name, not in a
  column.

The orchestrator image has no pandas or torch, so ``check_dlrm_rows``
reimplements the label and categorical checks of the trainer's
``validate_dataset`` over plain column lists. ``tests/test_demo_dataset.py``
runs both on the same rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import boto3

from shared.dlrm_features import CATEGORICAL_COLUMNS
from shared.shading_policy import DEFAULT_CONVERSION_VALUE
from shared.yield_exploration import (
    FLOOR_MULTIPLIER_MAX,
    FLOOR_MULTIPLIER_MIN,
    MARGIN_VALUE_MAX,
    MARGIN_VALUE_MIN,
)

# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------

DEMO_PREFIXES: dict[str, str] = {
    "dlrm_bid_shader": "demo-training-data",
    "deal_yield_manager_floor": "demo-training-data-deal-yield-floor",
    "deal_yield_manager_margin": "demo-training-data-deal-yield-margin",
}
DATA_OBJECT_NAME = "part-00000.parquet"
METADATA_OBJECT_NAME = "_demo_metadata.json"

DEMO_SOURCE = "demo_dataset"
DEMO_PROVENANCE = "simulated"
DEMO_NOTICE = (
    "Synthetic demonstration data generated from the rates shown. "
    "It does not describe real advertiser or auction behaviour."
)

# ---------------------------------------------------------------------------
# Sizing (FR1.3, FR1.4)
# ---------------------------------------------------------------------------

ROW_CAP = 100_000
DLRM_CONVERSION_TARGET = 50
DEAL_YIELD_LABEL_TARGET = 500
DEFAULT_SEED = 20261115

DLRM_MODEL_TYPE = "dlrm_bid_shader"
DEAL_YIELD_MODEL_TYPES = frozenset({"deal_yield_manager_floor", "deal_yield_manager_margin"})


@dataclass(frozen=True)
class ParamSpec:
    """One user-settable rate with its bounds and default (fractions, not percent)."""

    name: str
    label: str
    minimum: float
    maximum: float
    default: float


_DLRM_PARAMS = (
    ParamSpec("win_rate", "Win rate", 0.10, 1.00, 0.90),
    ParamSpec("click_rate", "Click rate given a win", 0.005, 0.50, 0.02),
    ParamSpec("conversion_rate", "Conversion rate given a click", 0.01, 1.00, 0.05),
)
_DEAL_YIELD_PARAMS = (
    ParamSpec("clear_share", "Share of deals that clear", 0.10, 1.00, 0.60),
)

PARAM_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "dlrm_bid_shader": _DLRM_PARAMS,
    "deal_yield_manager_floor": _DEAL_YIELD_PARAMS,
    "deal_yield_manager_margin": _DEAL_YIELD_PARAMS,
}

# Label ranges match what each yield container can output after its safety clamp
# (shared/yield_exploration.py).
DEAL_YIELD_LABEL_BOUNDS: dict[str, tuple[float, float]] = {
    "deal_yield_manager_floor": (FLOOR_MULTIPLIER_MIN, FLOOR_MULTIPLIER_MAX),
    "deal_yield_manager_margin": (MARGIN_VALUE_MIN, MARGIN_VALUE_MAX),
}

# ---------------------------------------------------------------------------
# Row shapes (FR1.7)
# ---------------------------------------------------------------------------

# Same order and names as etl/glue_deal_yield_feature_engineering.py's
# FEATURE_COLUMNS. The Glue script is not in the orchestrator image, so the list
# is restated here and pinned by tests/test_demo_dataset.py.
DEAL_YIELD_FEATURE_COLUMNS: tuple[str, ...] = (
    "is_first_price",
    "is_second_price",
    "bidfloor",
    "bidfloor_tier",
    "category_tier",
    "hour_norm",
    "weekday_norm",
)
DEAL_YIELD_COLUMNS: tuple[str, ...] = ("label", *DEAL_YIELD_FEATURE_COLUMNS)

# Thresholds from glue_deal_yield_feature_engineering.py.
_REMNANT_BIDFLOOR_MAX = 1.0
_PREMIUM_BIDFLOOR_MIN = 5.0

# (column, arrow type name) in the order the DLRM Glue job writes them: the
# raw_bid_outcomes columns (feedback_pipeline_cfn.yaml), then the derived ones.
DLRM_COLUMNS: tuple[tuple[str, str], ...] = (
    ("request_id", "string"),
    ("timestamp", "double"),
    ("model_version", "string"),
    ("source", "string"),
    ("model_type", "string"),
    ("original_price", "double"),
    ("shaded_price", "double"),
    ("bid_floor", "double"),
    ("won", "bool"),
    ("price_paid", "double"),
    ("impression", "bool"),
    ("click", "bool"),
    ("conversion", "bool"),
    ("conversion_value", "double"),
    ("user_id_hash", "string"),
    ("site_domain", "string"),
    ("device_type", "string"),
    ("hour_of_day", "int32"),
    ("day_of_week", "int32"),
    ("geo_country", "string"),
    ("has_video", "bool"),
    ("shade_factor_used", "double"),
    ("conversion_value_estimate_used", "double"),
    ("outcome_provenance", "string"),
    ("roi", "double"),
    ("shade_ratio", "double"),
    ("label", "int32"),
    ("label_conversion", "int32"),
    ("label_click", "int32"),
    ("win_rate_bucket", "int32"),
)

# The trainer's default objective is "cpa", which trains on label_conversion
# (train.py _OBJECTIVE_LABELS). training_trigger.py sends no objective.
DLRM_OBJECTIVE = "cpa"
DLRM_LABEL_COLUMN = "label_conversion"

# Fixed vocabularies: more than one value each, far fewer values than rows, so
# the trainer's constant and all-unique categorical checks both pass.
_SITE_DOMAINS = (
    "news.example", "sports.example", "weather.example", "recipes.example",
    "travel.example", "finance.example", "autos.example", "games.example",
    "music.example", "health.example", "tech.example", "style.example",
)
_DEVICE_TYPES = ("mobile", "desktop", "tablet", "ctv")
_GEO_COUNTRIES = ("US", "CA", "GB", "DE", "FR", "BR", "JP", "AU")
_USER_POOL = 5_000

# Conversion value matches orchestrator/outcome_simulator.py's default, so a
# converted demo row is valued like a simulated load-test conversion.
_DEMO_CONVERSION_VALUE = 25.0
_DEMO_SHADE_FACTOR = 0.85
_DEMO_MODEL_VERSION = "demo-dataset"

# Monday 2026-01-05 00:00 UTC. Timestamps are offsets from this fixed instant so
# the same seed gives the same bytes, and day_of_week (Monday = 0, as
# datetime.weekday() in feedback_integration.py) agrees with the timestamp.
_BASE_EPOCH = datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UnknownDemoModelTypeError(ValueError):
    """The model type has no demo dataset (unknown, or not trainable)."""

    def __init__(self, model_type: str):
        super().__init__(
            f"Model type '{model_type}' has no demo dataset. "
            f"Supported model types: {sorted(PARAM_SPECS)}."
        )
        self.model_type = model_type


class DemoParamsError(ValueError):
    """A generation parameter is unknown, not a number, or out of bounds."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


class DemoDatasetUnfitError(ValueError):
    """The generated rows fail the trainer's dataset check (FR1.9)."""


class DemoDatasetStorageError(RuntimeError):
    """An S3 call for the demo dataset failed. Carries the operation and key."""

    def __init__(self, operation: str, key: str, cause: Exception):
        super().__init__(f"S3 {operation} failed for '{key}': {cause}")
        self.operation = operation
        self.key = key


# ---------------------------------------------------------------------------
# Parameters and sizing
# ---------------------------------------------------------------------------


def _specs_for(model_type: str) -> tuple[ParamSpec, ...]:
    specs = PARAM_SPECS.get(model_type)
    if specs is None:
        raise UnknownDemoModelTypeError(model_type)
    return specs


def param_bounds(model_type: str) -> list[dict[str, Any]]:
    """The parameter table for model_type, as the status and estimate endpoints return it."""
    return [asdict(spec) for spec in _specs_for(model_type)]


def validate_params(model_type: str, params: Mapping[str, Any] | None) -> dict[str, float]:
    """Return a complete parameter set: supplied values checked, missing ones defaulted.

    Raises DemoParamsError naming the field for an unknown name, a value that is
    not a finite number, or a value outside its bound.
    """
    specs = _specs_for(model_type)
    supplied = dict(params or {})
    known = {spec.name for spec in specs}
    for name in supplied:
        if name not in known:
            raise DemoParamsError(
                name, f"'{name}' is not a parameter of {model_type}. Parameters: {sorted(known)}."
            )

    resolved: dict[str, float] = {}
    for spec in specs:
        raw = supplied.get(spec.name, spec.default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DemoParamsError(spec.name, f"'{spec.name}' must be a number, got {raw!r}.")
        value = float(raw)
        if not math.isfinite(value) or value < spec.minimum or value > spec.maximum:
            raise DemoParamsError(
                spec.name,
                f"'{spec.name}' must be between {spec.minimum} and {spec.maximum}, got {raw!r}.",
            )
        resolved[spec.name] = value
    return resolved


@dataclass(frozen=True)
class DatasetSizing:
    """How many rows a generation produces and how many positives to expect.

    ``rows`` is the number of records generated: bid rows for DLRM, offered deals
    for deal yield (only cleared deals become training rows).
    """

    model_type: str
    rows: int
    row_cap: int
    capped: bool
    positive_kind: str
    target_positives: int
    expected_positives: float
    expected_wins: float | None = None
    expected_clicks: float | None = None


def _ceil_rows(target: float, rate: float) -> int:
    # The small tolerance keeps an exact quotient (500 / 0.5) from rounding up
    # because of float error.
    return int(math.ceil(target / rate - 1e-9))


def size_dataset(model_type: str, params: Mapping[str, float]) -> DatasetSizing:
    """Compute the row count for a validated parameter set (FR1.3, FR1.4)."""
    _specs_for(model_type)
    if model_type == DLRM_MODEL_TYPE:
        conversions_per_row = params["win_rate"] * params["click_rate"] * params["conversion_rate"]
        wanted = _ceil_rows(DLRM_CONVERSION_TARGET, conversions_per_row)
        rows = min(ROW_CAP, wanted)
        return DatasetSizing(
            model_type=model_type,
            rows=rows,
            row_cap=ROW_CAP,
            capped=wanted > ROW_CAP,
            positive_kind="conversions",
            target_positives=DLRM_CONVERSION_TARGET,
            expected_positives=round(rows * conversions_per_row, 2),
            expected_wins=round(rows * params["win_rate"], 2),
            expected_clicks=round(rows * params["win_rate"] * params["click_rate"], 2),
        )

    wanted = _ceil_rows(DEAL_YIELD_LABEL_TARGET, params["clear_share"])
    rows = min(ROW_CAP, wanted)
    return DatasetSizing(
        model_type=model_type,
        rows=rows,
        row_cap=ROW_CAP,
        capped=wanted > ROW_CAP,
        positive_kind="labelled_deals",
        target_positives=DEAL_YIELD_LABEL_TARGET,
        expected_positives=round(rows * params["clear_share"], 2),
    )


# ---------------------------------------------------------------------------
# Generation (FR1.7, FR1.8)
# ---------------------------------------------------------------------------


def _win_rate_bucket(rate: float) -> int:
    """Same binning as glue_feature_engineering.compute_win_rate_buckets."""
    return 9 if rate >= 1.0 else int(math.floor(rate * 10))


def _dlrm_row(index: int, seed: int, rng: random.Random, params: Mapping[str, float]) -> dict:
    hour = rng.randrange(24)
    day = rng.randrange(7)
    bid_floor = round(rng.uniform(0.25, 3.0), 4)
    original_price = round(bid_floor + rng.uniform(0.5, 6.0), 4)
    shaded_price = round(max(bid_floor, original_price * _DEMO_SHADE_FACTOR), 4)
    won = rng.random() < params["win_rate"]
    click = won and rng.random() < params["click_rate"]
    conversion = click and rng.random() < params["conversion_rate"]
    price_paid = round(rng.uniform(bid_floor, shaded_price), 4) if won else None
    conversion_value = _DEMO_CONVERSION_VALUE if conversion else None
    user = rng.randrange(_USER_POOL)
    row = {
        "request_id": f"demo-{seed}-{index:06d}",
        "timestamp": _BASE_EPOCH + day * 86400 + hour * 3600 + rng.randrange(3600),
        "model_version": _DEMO_MODEL_VERSION,
        "source": DEMO_SOURCE,
        "model_type": DLRM_MODEL_TYPE,
        "original_price": original_price,
        "shaded_price": shaded_price,
        "bid_floor": bid_floor,
        "won": won,
        "price_paid": price_paid,
        "impression": won,
        "click": click,
        "conversion": conversion,
        "conversion_value": conversion_value,
        "user_id_hash": hashlib.sha256(f"demo-user-{user}".encode()).hexdigest()[:32],
        "site_domain": rng.choice(_SITE_DOMAINS),
        "device_type": rng.choice(_DEVICE_TYPES),
        "hour_of_day": hour,
        "day_of_week": day,
        "geo_country": rng.choice(_GEO_COUNTRIES),
        "has_video": rng.random() < 0.3,
        "shade_factor_used": _DEMO_SHADE_FACTOR,
        "conversion_value_estimate_used": DEFAULT_CONVERSION_VALUE,
        "outcome_provenance": DEMO_PROVENANCE,
    }
    # Derived columns, computed as engineer_features() computes them with a
    # closed attribution window: a won row with no response is a real 0.
    has_value = price_paid is not None and price_paid > 0 and conversion_value is not None
    row["roi"] = (conversion_value - price_paid) / price_paid if has_value else None
    row["shade_ratio"] = shaded_price / original_price
    row["label"] = int(bool(won and has_value and conversion_value > price_paid))
    row["label_conversion"] = int(conversion)
    row["label_click"] = int(click)
    return row


def generate_dlrm_rows(params: Mapping[str, float], rows: int, seed: int) -> dict[str, list]:
    """Seeded DLRM rows as columns, in DLRM_COLUMNS order."""
    rng = random.Random(seed)
    records = [_dlrm_row(i, seed, rng, params) for i in range(rows)]

    wins: dict[tuple[int, str], list[int]] = {}
    for record in records:
        key = (record["hour_of_day"], record["device_type"])
        counts = wins.setdefault(key, [0, 0])
        counts[0] += int(record["won"])
        counts[1] += 1
    for record in records:
        won_count, total = wins[(record["hour_of_day"], record["device_type"])]
        record["win_rate_bucket"] = _win_rate_bucket(won_count / total)

    return {name: [record[name] for record in records] for name, _ in DLRM_COLUMNS}


def _bidfloor_tier(bidfloor: float) -> float:
    if bidfloor <= _REMNANT_BIDFLOOR_MAX:
        return 0.0
    if bidfloor >= _PREMIUM_BIDFLOOR_MIN:
        return 2.0
    return 1.0


def generate_deal_yield_rows(
    model_type: str, params: Mapping[str, float], deals: int, seed: int
) -> dict[str, list]:
    """Seeded deal yield rows as columns: ``label`` then the 7 features.

    Every offered deal draws the same values whether or not it clears, so the
    stream does not depend on which deals cleared. Only cleared deals become
    rows, as the Glue job keeps only won deals.
    """
    low, high = DEAL_YIELD_LABEL_BOUNDS[model_type]
    rng = random.Random(seed)
    columns: dict[str, list] = {name: [] for name in DEAL_YIELD_COLUMNS}
    for _ in range(deals):
        cleared = rng.random() < params["clear_share"]
        auction_type = rng.choice((1, 2))
        bidfloor = round(rng.uniform(0.2, 8.0), 4)
        category_tier = float(rng.choice((0, 1, 2)))
        hour = rng.randrange(24)
        day = rng.randrange(7)
        label = round(rng.uniform(low, high), 6)
        if not cleared:
            continue
        columns["label"].append(label)
        columns["is_first_price"].append(1.0 if auction_type == 1 else 0.0)
        columns["is_second_price"].append(1.0 if auction_type == 2 else 0.0)
        columns["bidfloor"].append(bidfloor)
        columns["bidfloor_tier"].append(_bidfloor_tier(bidfloor))
        columns["category_tier"].append(category_tier)
        columns["hour_norm"].append(hour / 24.0)
        columns["weekday_norm"].append(day / 7.0)
    return columns


# ---------------------------------------------------------------------------
# Pre-write checks (FR1.9)
# ---------------------------------------------------------------------------


def check_dlrm_rows(
    columns: Mapping[str, list],
    label_column: str = DLRM_LABEL_COLUMN,
    objective: str = DLRM_OBJECTIVE,
) -> dict[str, Any]:
    """The label and categorical checks of train.validate_dataset, over column lists.

    Message wording follows the trainer so the person sees the same diagnosis the
    training job would have produced.
    """
    if label_column not in columns:
        raise DemoDatasetUnfitError(
            f"objective '{objective}' trains on '{label_column}', which is not in the "
            f"dataset. Columns present: {sorted(columns)}."
        )
    labels = columns[label_column]
    total_rows = len(labels)
    labelled = [v for v in labels if v is not None]
    n_labelled = len(labelled)
    if n_labelled == 0:
        raise DemoDatasetUnfitError(
            f"all {total_rows} rows have a NULL '{label_column}', so none of them is "
            "labelled."
        )
    positives = sum(1 for v in labelled if v > 0)
    positive_rate = positives / n_labelled
    if positives == 0 or positives == n_labelled:
        which = "no positives" if positives == 0 else "no negatives"
        raise DemoDatasetUnfitError(
            f"'{label_column}' has {which} across {n_labelled} labelled rows "
            f"(positive rate {positive_rate:.3f}). A single-class label trains the model to "
            "predict a constant, at a loss that falls steadily and means nothing."
        )

    cardinality: dict[str, int] = {}
    for column in CATEGORICAL_COLUMNS:
        if column not in columns:
            raise DemoDatasetUnfitError(f"categorical feature '{column}' is not in the dataset.")
        distinct = len(set(columns[column]))
        cardinality[column] = distinct
        if distinct <= 1:
            raise DemoDatasetUnfitError(
                f"categorical feature '{column}' has {distinct} distinct value(s) across "
                f"{total_rows} rows, so it carries no information and its embedding cannot "
                "be trained."
            )
        if total_rows > 1 and distinct == total_rows:
            raise DemoDatasetUnfitError(
                f"categorical feature '{column}' has one distinct value per row "
                f"({distinct} of {total_rows}). Every embedding row would be seen exactly "
                "once, which memorises rather than learns."
            )

    return {
        "objective": objective,
        "label_column": label_column,
        "total_rows": total_rows,
        "labelled_rows": n_labelled,
        "positives": positives,
        "positive_rate": round(positive_rate, 6),
        "categorical_cardinality": cardinality,
    }


def check_deal_yield_rows(model_type: str, columns: Mapping[str, list]) -> dict[str, Any]:
    """The deal yield check: rows present, labels set and inside the container bounds,
    and every feature column present with ``label`` first."""
    low, high = DEAL_YIELD_LABEL_BOUNDS[model_type]
    names = list(columns)
    if names[: len(DEAL_YIELD_COLUMNS)] != list(DEAL_YIELD_COLUMNS):
        raise DemoDatasetUnfitError(
            f"columns must be {list(DEAL_YIELD_COLUMNS)} with 'label' first, got {names}."
        )
    labels = columns["label"]
    if not labels:
        raise DemoDatasetUnfitError(
            "no deal cleared, so there are no labelled rows. Raise the clear share."
        )
    if any(v is None for v in labels):
        raise DemoDatasetUnfitError("'label' has NULL values; every cleared deal needs a label.")
    outside = [v for v in labels if v < low or v > high]
    if outside:
        raise DemoDatasetUnfitError(
            f"{len(outside)} label(s) fall outside [{low}, {high}], the range "
            f"{model_type} can output."
        )
    return {
        "label_column": "label",
        "total_rows": len(labels),
        "labelled_rows": len(labels),
        "label_min": min(labels),
        "label_max": max(labels),
        "label_bounds": [low, high],
    }


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


def _arrow_schema(model_type: str):
    import pyarrow as pa

    if model_type == DLRM_MODEL_TYPE:
        types = {"string": pa.string(), "double": pa.float64(), "bool": pa.bool_(), "int32": pa.int32()}
        return pa.schema([(name, types[kind]) for name, kind in DLRM_COLUMNS])
    return pa.schema([(name, pa.float64()) for name in DEAL_YIELD_COLUMNS])


def to_parquet_bytes(model_type: str, columns: Mapping[str, list]) -> bytes:
    """Serialize columns to Parquet with the model type's fixed schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _arrow_schema(model_type)
    table = pa.Table.from_pydict({name: columns[name] for name in schema.names}, schema=schema)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedDemoDataset:
    model_type: str
    parquet: bytes
    metadata: dict[str, Any]


def data_key(model_type: str) -> str:
    return f"{DEMO_PREFIXES[model_type]}/{DATA_OBJECT_NAME}"


def metadata_key(model_type: str) -> str:
    return f"{DEMO_PREFIXES[model_type]}/{METADATA_OBJECT_NAME}"


def generate_demo_dataset(
    model_type: str,
    params: Mapping[str, Any] | None,
    seed: int = DEFAULT_SEED,
    *,
    now: datetime | None = None,
) -> GeneratedDemoDataset:
    """Validate, size, generate and check a dataset, and build its metadata.

    Raises UnknownDemoModelTypeError, DemoParamsError or DemoDatasetUnfitError.
    Nothing is written; see write_demo_dataset.
    """
    resolved = validate_params(model_type, params)
    sizing = size_dataset(model_type, resolved)
    if model_type == DLRM_MODEL_TYPE:
        columns = generate_dlrm_rows(resolved, sizing.rows, seed)
        report = check_dlrm_rows(columns)
        actual_positives = report["positives"]
    else:
        columns = generate_deal_yield_rows(model_type, resolved, sizing.rows, seed)
        report = check_deal_yield_rows(model_type, columns)
        actual_positives = report["labelled_rows"]

    generated_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    metadata = {
        "model_type": model_type,
        "demonstration_only": True,
        "notice": DEMO_NOTICE,
        "source": DEMO_SOURCE,
        "outcome_provenance": DEMO_PROVENANCE,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "params": resolved,
        "sizing": asdict(sizing),
        "rows_written": report["total_rows"],
        "actual_positives": actual_positives,
        "check": report,
        "data_key": data_key(model_type),
    }
    return GeneratedDemoDataset(
        model_type=model_type,
        parquet=to_parquet_bytes(model_type, columns),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# S3 (FR1.10, FR1.11)
# ---------------------------------------------------------------------------


def _s3_client():
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    return boto3.client("s3", region_name=region)


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    """Every object key under prefix (paginated ListObjectsV2)."""
    keys: list[str] = []
    kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
    while True:
        try:
            resp = client.list_objects_v2(**kwargs)
        except Exception as exc:
            raise DemoDatasetStorageError("ListObjectsV2", prefix, exc) from exc
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        token = resp.get("NextContinuationToken")
        if not resp.get("IsTruncated") or not token:
            return keys
        kwargs["ContinuationToken"] = token


def _delete_keys(client, bucket: str, keys: list[str]) -> None:
    # DeleteObjects takes at most 1,000 keys per request and reports per-key
    # failures in Errors rather than raising.
    for start in range(0, len(keys), 1000):
        batch = keys[start:start + 1000]
        try:
            resp = client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
            )
        except Exception as exc:
            raise DemoDatasetStorageError("DeleteObjects", batch[0], exc) from exc
        errors = resp.get("Errors") or []
        if errors:
            first = errors[0]
            raise DemoDatasetStorageError(
                "DeleteObjects",
                first.get("Key", batch[0]),
                RuntimeError(f"{first.get('Code')}: {first.get('Message')} ({len(errors)} key(s) not deleted)"),
            )


def write_demo_dataset(bucket: str, dataset: GeneratedDemoDataset, client=None) -> dict[str, Any]:
    """Replace the model type's demo dataset: delete the prefix, then write data and metadata.

    The metadata object is written last, so its presence means the data object
    was written. Any S3 failure raises DemoDatasetStorageError.
    """
    client = client or _s3_client()
    prefix = f"{DEMO_PREFIXES[dataset.model_type]}/"
    existing = list_keys(client, bucket, prefix)
    if existing:
        _delete_keys(client, bucket, existing)

    data = data_key(dataset.model_type)
    meta = metadata_key(dataset.model_type)
    try:
        client.put_object(
            Bucket=bucket, Key=data, Body=dataset.parquet, ContentType="application/x-parquet"
        )
    except Exception as exc:
        raise DemoDatasetStorageError("PutObject", data, exc) from exc
    try:
        client.put_object(
            Bucket=bucket,
            Key=meta,
            Body=json.dumps(dataset.metadata, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:
        raise DemoDatasetStorageError("PutObject", meta, exc) from exc
    return dataset.metadata


def _is_missing_key(exc: Exception) -> bool:
    response = getattr(exc, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    return code in {"NoSuchKey", "404", "NotFound"}


def read_demo_metadata(bucket: str, model_type: str, client=None) -> dict[str, Any] | None:
    """The stored metadata for model_type's demo dataset, or None when none exists."""
    _specs_for(model_type)
    client = client or _s3_client()
    key = metadata_key(model_type)
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _is_missing_key(exc):
            return None
        raise DemoDatasetStorageError("GetObject", key, exc) from exc
    try:
        return json.loads(resp["Body"].read())
    except (ValueError, KeyError) as exc:
        raise DemoDatasetStorageError("GetObject", key, exc) from exc
