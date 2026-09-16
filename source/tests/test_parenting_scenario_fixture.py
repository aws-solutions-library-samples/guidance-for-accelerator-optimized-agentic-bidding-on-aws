"""The parenting narrative fixture, asserted against the real container code.

The point of this fixture is that the story the Auction Theater tells is the story the
containers actually produce. So these tests call the real scoring functions rather than
asserting a hand-written expectation.

WHAT CANNOT BE ASSERTED HERE, and why:

  Three of the five containers this scenario invokes call NVIDIA Triton for inference --
  ``ncf_deal_manager`` (deal activation and suppression), ``yield_optimizer_floor`` and
  ``yield_optimizer_margin``. Their output for this fixture is only observable against a
  deployed stack with Triton serving the models, so the deal, floor and margin mutations
  the narrative describes are NOT covered by these tests. What is covered is the two
  rule-based containers, ``widedeep_segment_activator`` and ``metrics_enricher``, whose
  behaviour is fully determined by the request.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "containers", "widedeep_segment_activator"))

from shared import iab_taxonomy  # noqa: E402

_SAMPLES = os.path.join(_ROOT, "frontend-react", "public", "samples")
_FIXTURE = os.path.join(_SAMPLES, "parenting-narrative.json")

# FR-32. No expecting-parent or maternity segment exists in Audience Taxonomy 1.1, so
# the narrative is written around the identifiers that do.
FR32_IDS = {"350", "354", "98", "7"}


@pytest.fixture(scope="module")
def payload():
    with open(_FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def widedeep():
    import importlib
    return importlib.import_module("app")


@pytest.fixture(scope="module")
def enricher():
    import importlib.util
    path = os.path.join(_ROOT, "containers", "metrics_enricher", "app.py")
    spec = importlib.util.spec_from_file_location("metrics_enricher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestFixtureShape:
    def test_the_fixture_exists_and_parses(self, payload):
        assert payload["id"] == "sample-parenting-001"

    def test_it_declares_content_taxonomy_3_1(self, payload):
        site = payload["bid_request"]["site"]
        assert site["cattax"] == 9
        assert site["cat"] == ["192"]
        assert iab_taxonomy.resolve_taxonomy(site["cattax"]) == iab_taxonomy.TAXONOMY_CONTENT_3_X

    def test_category_192_is_real_and_is_not_special_category_data(self):
        # 192 Parenting is a real Content Taxonomy 3.1 id and is NOT flagged, which is
        # why it may derive an interest segment at all. Its tier-1 parent 186 IS
        # flagged, and the flag is per-row, so 192 is unaffected.
        assert "192" in iab_taxonomy.CONTENT_CATEGORIES
        assert iab_taxonomy.is_special_category_data("192") is False
        assert iab_taxonomy.is_special_category_data("186") is True

    def test_it_carries_real_deals_with_floors_and_auction_types(self, payload):
        deals = payload["bid_request"]["imp"][0]["pmp"]["deals"]
        assert len(deals) == 3
        for deal in deals:
            assert isinstance(deal["bidfloor"], (int, float))
            assert deal["at"] in (1, 2, 3)
        # Floors descend across the deals, so a floor movement is visible.
        assert [d["bidfloor"] for d in deals] == [3.40, 2.10, 0.85]

    def test_the_declared_intents_match_the_containers_the_card_lists(self, payload):
        assert set(payload["applicable_intents"]) == {
            "ACTIVATE_SEGMENTS", "ADD_METRICS", "ACTIVATE_DEALS",
            "SUPPRESS_DEALS", "ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN",
        }


class TestTheRealContainerProducesTheNarrative:
    def test_all_four_fr32_identifiers_are_activated(self, payload, widedeep):
        got = set(widedeep._score_segments(payload["bid_request"]))
        assert FR32_IDS <= got, f"missing {sorted(FR32_IDS - got)}"

    def test_the_activation_is_exactly_what_is_expected(self, payload, widedeep):
        # Pinned in full, so a rule change that silently adds or drops a segment fails.
        assert widedeep._score_segments(payload["bid_request"]) == [
            "350", "354", "7", "98", "ctx-mobile", "ctx-premium",
        ]

    def test_a_real_mutation_is_emitted_not_just_a_score(self, payload, widedeep):
        from shared.artf_types import Intent, RTBRequest
        response = widedeep.mutate(RTBRequest(**payload))
        segment_mutations = [m for m in response.mutations if m.intent == Intent.ACTIVATE_SEGMENTS]
        assert len(segment_mutations) == 1
        assert FR32_IDS <= set(segment_mutations[0].ids.id)

    def test_the_segment_names_the_theater_will_show_are_real(self):
        expected = {
            "350": "Interest | Family and Relationships | Parenting",
            "354": "Interest | Family and Relationships | Parenting Babies and Toddlers",
            "98": "Demographic | Household Data | Parents with Children",
            "7": "Demographic | Age Range | 35-39",
        }
        for segment_id, name in expected.items():
            assert iab_taxonomy.condensed_name(segment_id) == name


class TestEachIdentifierComesFromTheRuleTheNarrativeClaims:
    """The narrative attributes each identifier to a source. If a different rule
    happened to supply it, the story would be wrong even though the output matched.
    """

    def test_350_comes_from_the_page_category(self, payload, widedeep):
        request = json.loads(json.dumps(payload["bid_request"]))
        request["site"]["cat"] = []
        assert "350" not in widedeep._score_segments(request)

    def test_7_comes_from_the_year_of_birth(self, payload, widedeep):
        request = json.loads(json.dumps(payload["bid_request"]))
        del request["user"]["yob"]
        assert "7" not in widedeep._score_segments(request)

    def test_1989_is_what_puts_the_user_in_bucket_7(self, payload):
        assert payload["bid_request"]["user"]["yob"] == 1989
        assert iab_taxonomy.age_bucket_id(1989) == "7"

    def test_98_and_354_come_only_from_the_asserted_audience_data(self, payload, widedeep):
        # BR-15a. Life-stage and household-composition identifiers are reachable only
        # from data a party has asserted about the person, never from page content:
        # deriving a life stage from what someone read is the inference Special
        # Category Data handling exists to discourage.
        request = json.loads(json.dumps(payload["bid_request"]))
        request["user"]["data"] = []
        got = set(widedeep._score_segments(request))
        assert "98" not in got
        assert "354" not in got
        # The content-derived and demographic identifiers survive.
        assert {"350", "7"} <= got

    def test_page_content_alone_never_yields_the_household_segment(self, widedeep):
        request = {
            "site": {"cat": ["192", "186", "197", "198"], "cattax": 9},
            "user": {"yob": 1989},
            "imp": [{"bidfloor": 2.60}],
            "device": {},
        }
        assert "98" not in widedeep._score_segments(request)

    def test_ctx_premium_comes_from_the_impression_floor(self, payload, widedeep):
        assert payload["bid_request"]["imp"][0]["bidfloor"] == 2.60
        request = json.loads(json.dumps(payload["bid_request"]))
        request["imp"][0]["bidfloor"] = 1.00
        assert "ctx-premium" not in widedeep._score_segments(request)

    def test_ctx_mobile_comes_from_the_user_agent(self, payload, widedeep):
        request = json.loads(json.dumps(payload["bid_request"]))
        request["device"]["ua"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        assert "ctx-mobile" not in widedeep._score_segments(request)


class TestSignalsEnricher:
    def test_it_emits_a_metrics_mutation(self, payload, enricher):
        from shared.artf_types import RTBRequest
        response = enricher.mutate(RTBRequest(**payload))
        assert len(response.mutations) == 1
        types = {m.type for m in response.mutations[0].add_metrics.metric}
        assert types == {"viewability", "brand_safety"}

    def test_brand_safety_reflects_the_content_not_the_taxonomy_version(self, payload, enricher):
        # The reason U4 extended this container: before, a parenting article declaring
        # Content Taxonomy 3.1 scored 0.60, the floor for "categories present, none
        # recognised" -- and therefore worse than a request with no categories (0.80).
        site = payload["bid_request"]["site"]
        assert enricher._brand_safety(site) == 1.0

    def test_the_metrics_the_theater_will_show(self, payload, enricher):
        from shared.artf_types import RTBRequest
        response = enricher.mutate(RTBRequest(**payload))
        by_type = {m.type: m.value for m in response.mutations[0].add_metrics.metric}
        assert by_type["brand_safety"] == 1.0
        # 0.50, not 0.70. `_viewability` reads `imp["pos"]`, but OpenRTB puts `pos` on
        # the Banner object, which is where this fixture and every existing one put it.
        # So the position is never read and the score falls to the pos-absent default.
        # Asserted as the real value rather than the intended one, and reported as a
        # pre-existing defect in the U4 summary rather than fixed here -- it affects
        # every fixture, not just this one.
        assert by_type["viewability"] == 0.50


class TestScenarioCardWiring:
    """The card and the fixture have to agree, or the UI fetches a file that does not
    describe what the card advertises.
    """

    def _card_source(self):
        path = os.path.join(_ROOT, "frontend-react", "src", "components", "ScenarioCard.jsx")
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_the_card_points_at_this_fixture(self):
        assert 'file: "parenting-narrative.json"' in self._card_source()

    def test_the_card_declares_the_intents_the_fixture_declares(self, payload):
        source = self._card_source()
        start = source.index('id: "parenting-narrative"')
        block = source[start:source.index("},", source.index("controls:", start))]
        for intent in payload["applicable_intents"]:
            assert intent in block, f"card does not tag {intent}"

    def test_the_illustrative_outcome_is_keyed_on_this_scenario(self):
        path = os.path.join(_ROOT, "frontend-react", "src", "utils", "theaterIllustrative.js")
        with open(path, encoding="utf-8") as handle:
            assert '"parenting-narrative"' in handle.read()
