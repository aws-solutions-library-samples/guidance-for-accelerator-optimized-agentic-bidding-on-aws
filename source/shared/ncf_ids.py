"""The Deal Scorer's id space, thresholds and decision rule. Stdlib only.

One definition read by every producer and reader of the NCF model: the serving
container (containers/ncf_deal_manager/app.py), the deploy-time sample trainer and
its acceptance gate (training/ncf_sample/), and the SageMaker trainer
(training/container/train.py, which receives this file through
training/stage_shared.sh). Training and serving hash ids with the same function and
decide with the same thresholds, so they cannot drift.

NCF sees only two ids: a user bucket and an item (deal) bucket, each the md5 of the
id modulo the embedding table size.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

#: Rows in the user embedding tables.
NB_USERS = 2000
#: Rows in the item (deal) embedding tables.
NB_ITEMS = 2000

#: A deal scoring at or above this is activated (library candidates only).
ACTIVATE_THRESHOLD = 0.60
#: An offered deal scoring below this is suppressed. Scores between the two
#: thresholds leave the deal as it is.
SUPPRESS_THRESHOLD = 0.30

#: Prefix of the model version label of a model trained on sample interactions.
SAMPLE_VERSION_PREFIX = "ncf-neumf-sample-"
#: The full label: the prefix plus the first 12 hex characters of the training
#: fingerprint (training/ncf_sample/build.py).
MODEL_LABEL_RE = r"^ncf-neumf-sample-[0-9a-f]{12}$"

_MODEL_LABEL = re.compile(MODEL_LABEL_RE)


def bucket(value: str, n: int = NB_USERS) -> int:
    """The embedding row for an id: md5 of the UTF-8 id, modulo ``n``."""
    return int(hashlib.md5(value.encode(), usedforsecurity=False).hexdigest(), 16) % n  # nosec B324


def is_model_label(value: object) -> bool:
    """Whether ``value`` is a sample-trained model version label."""
    return isinstance(value, str) and _MODEL_LABEL.match(value) is not None


def classify(
    ids: Sequence[str],
    scores: Sequence[float],
    activate: float = ACTIVATE_THRESHOLD,
    suppress: float = SUPPRESS_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """Split scored ids into (to_activate, to_suppress), keeping the input order.

    ``score >= activate`` activates and ``score < suppress`` suppresses. With
    ``suppress <= activate`` an id lands in at most one list.
    """
    if len(ids) != len(scores):
        raise ValueError(f"{len(ids)} ids but {len(scores)} scores")
    to_act = [i for i, s in zip(ids, scores) if float(s) >= activate]
    to_sup = [i for i, s in zip(ids, scores) if float(s) < suppress]
    return to_act, to_sup


def restrict(
    offered_ids: Iterable[str],
    candidate_ids: Iterable[str],
    to_act: Iterable[str],
    to_sup: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Activate only library candidates and suppress only offered deals.

    Activating a deal the request already offers changes nothing in the applier
    and would show a decision that did nothing. Suppressing a library deal would
    remove a deal the publisher never offered on this request.
    """
    offered = set(offered_ids)
    candidates = set(candidate_ids)
    return (
        [d for d in to_act if d in candidates],
        [d for d in to_sup if d in offered],
    )
