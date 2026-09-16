"""Content Taxonomy 3.x support in metrics_enricher's brand-safety score.

Before this change `_brand_safety` compared the request's categories against a set of
Content Taxonomy **1.0** codes. A request declaring 3.x matched none of them and scored
0.60 -- the floor for "categories present, none recognised" -- which is WORSE than the
0.80 a request with no categories at all receives. Declaring a modern taxonomy was
penalised, and the parenting fixture made that visible.

The legacy baseline below was captured from the implementation BEFORE the change and is
committed rather than recomputed, so these tests would fail if the 1.0 path drifted.
"""

import importlib.util
import json
import os
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)

from shared import iab_taxonomy  # noqa: E402

_SAMPLES = os.path.join(_ROOT, "frontend-react", "public", "samples")

# Captured from the pre-change implementation against the real fixture files.
LEGACY_BASELINE = {
    "banner-basic.json": {"brand_safety": 1.0, "viewability": [0.5]},
    "bid-shading.json": {"brand_safety": 1.0, "viewability": [0.5]},
    "isv-ecosystem.json": {"brand_safety": 0.8666666666666667, "viewability": [0.55, 0.55]},
    "video-deals.json": {"brand_safety": 1.0, "viewability": [0.62]},
    "yield-optimizer.json": {"brand_safety": 1.0, "viewability": [0.62]},
}


@pytest.fixture(scope="module")
def enricher():
    path = os.path.join(_ROOT, "containers", "metrics_enricher", "app.py")
    spec = importlib.util.spec_from_file_location("metrics_enricher_taxonomy_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestLegacyEquivalence:
    """Every request that does not declare a modern taxonomy scores exactly as before."""

    @pytest.mark.parametrize("name", sorted(LEGACY_BASELINE))
    def test_existing_fixtures_are_unchanged(self, name, enricher):
        with open(os.path.join(_SAMPLES, name), encoding="utf-8") as handle:
            payload = json.load(handle)
        request = payload["bid_request"]
        site = request.get("site", request.get("app", {}))
        want = LEGACY_BASELINE[name]
        assert enricher._brand_safety(site) == want["brand_safety"]
        assert [enricher._viewability(i) for i in request.get("imp", [])] == want["viewability"]

    def test_the_1_0_safe_set_is_untouched(self, enricher):
        assert enricher._SAFE_CATS == {
            "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7",
            "IAB8", "IAB9", "IAB10", "IAB12", "IAB13", "IAB17", "IAB19", "IAB20",
        }

    @pytest.mark.parametrize("cattax", [None, 1])
    def test_absent_or_1_0_cattax_uses_the_legacy_path(self, cattax, enricher):
        site = {"cat": ["IAB17"]}
        if cattax is not None:
            site["cattax"] = cattax
        assert enricher._brand_safety(site) == 1.0

    def test_no_categories_still_scores_the_neutral_default(self, enricher):
        assert enricher._brand_safety({}) == 0.80
        assert enricher._brand_safety({"cat": []}) == 0.80

    def test_an_unrecognised_1_0_code_still_scores_the_floor(self, enricher):
        assert enricher._brand_safety({"cat": ["IAB99"]}) == 0.60


class TestContentTaxonomy3x:
    def test_the_defect_this_change_fixes(self, enricher):
        """A parenting article declaring 3.1 scored 0.60 before, and 0.60 is lower than
        the 0.80 given to a request carrying no categories at all.
        """
        site = {"cat": ["192"], "cattax": 9}
        assert enricher._brand_safety(site) == 1.0
        assert enricher._brand_safety(site) > enricher._brand_safety({})

    @pytest.mark.parametrize("category_id,tier1", [
        ("192", "Family and Relationships"),
        ("483", "Sports"),
        ("210", "Food & Drink"),
        ("596", "Technology & Computing"),
        ("422", "Pets"),
        ("464", "Science"),
    ])
    def test_suitable_tier1_categories_score_full(self, category_id, tier1, enricher):
        assert iab_taxonomy.content_tier1(category_id) == tier1
        assert enricher._brand_safety({"cat": [category_id], "cattax": 9}) == 1.0

    @pytest.mark.parametrize("category_id,tier1", [
        ("380", "Crime"),
        ("381", "Disasters"),
        ("383", "Law"),
        ("386", "Politics"),
        ("389", "War and Conflicts"),
        ("453", "Religion & Spirituality"),
    ])
    def test_unsuitable_tier1_categories_score_the_floor(self, category_id, tier1, enricher):
        assert iab_taxonomy.content_tier1(category_id) == tier1
        assert enricher._brand_safety({"cat": [category_id], "cattax": 9}) == 0.60

    def test_a_mixed_request_scores_the_suitable_fraction(self, enricher):
        assert enricher._brand_safety({"cat": ["192", "380"], "cattax": 9}) == 0.80

    def test_an_unknown_3_x_code_is_not_assumed_safe(self, enricher):
        assert enricher._brand_safety({"cat": ["not-a-real-id"], "cattax": 9}) == 0.60

    @pytest.mark.parametrize("cattax", [7, 9])  # 3.0 and 3.1; 6 is 2.2, not 3.x
    def test_every_3_x_taxonomy_version_takes_the_new_path(self, cattax, enricher):
        assert iab_taxonomy.resolve_taxonomy(cattax) == iab_taxonomy.TAXONOMY_CONTENT_3_X
        assert enricher._brand_safety({"cat": ["192"], "cattax": cattax}) == 1.0

    @pytest.mark.parametrize("cattax", [2, 5, 6])  # 1.1, 2.1, 2.2 -- none is 3.x
    def test_a_pre_3_x_taxonomy_falls_back_to_the_legacy_path(self, cattax, enricher):
        # Their codes are numeric while the 1.0 safe set is alphanumeric, so nothing
        # matches and the floor is correct. Only 3.x gets tier-1 resolution.
        assert iab_taxonomy.resolve_taxonomy(cattax) != iab_taxonomy.TAXONOMY_CONTENT_3_X
        assert enricher._brand_safety({"cat": ["192"], "cattax": cattax}) == 0.60


class TestSuitabilityIsNotSpecialCategoryData:
    """SCD is a privacy control about what may be inferred from what someone read. It
    says nothing about whether an advertiser wants to appear beside the content, and
    conflating the two was explicitly rejected during U2.
    """

    def test_an_scd_flagged_category_can_still_be_brand_safe(self, enricher):
        # 186 Family and Relationships is SCD-flagged and plainly brand-safe.
        assert iab_taxonomy.is_special_category_data("186") is True
        assert enricher._brand_safety({"cat": ["186"], "cattax": 9}) == 1.0

    def test_an_unflagged_category_can_still_be_unsuitable(self, enricher):
        # 380 Crime is not SCD-flagged and plainly is not brand-safe.
        assert iab_taxonomy.is_special_category_data("380") is False
        assert enricher._brand_safety({"cat": ["380"], "cattax": 9}) == 0.60

    def test_the_unsuitable_set_is_tier1_names_not_scd_flags(self, enricher):
        assert enricher._UNSUITABLE_TIER1_3X == {
            "Crime", "Disasters", "Law", "Politics",
            "Religion & Spirituality", "Sensitive Topics", "War and Conflicts",
        }


class TestModelVersion:
    def test_it_was_bumped(self, enricher):
        """The frontend and the load-test aggregation surface model_version as the
        served version. Leaving v1 would report old behaviour for new output.
        """
        assert enricher.MODEL_VERSION == "metrics-rules-v2-iab-taxonomy"


class TestNeverRaises:
    @pytest.mark.parametrize("site", [
        {}, {"cat": None}, {"cat": []}, {"cat": [None]}, {"cat": [1, 2]},
        {"cat": ["192"], "cattax": "nine"}, {"cat": ["192"], "cattax": None},
        {"cattax": 9}, {"cat": ["192"] * 50, "cattax": 9},
    ])
    def test_brand_safety_is_total(self, site, enricher):
        value = enricher._brand_safety(site)
        assert 0.0 <= value <= 1.0
