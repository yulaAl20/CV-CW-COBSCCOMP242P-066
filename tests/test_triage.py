"""
Triage rule tests.

The triage policy is the part of the system a clinic would actually be exposed
to, so each of the four rules gets a case that must fire it, and the precedence
between rules is tested explicitly.
"""

import pytest

from backend.config import Settings
from backend.inference import Prediction
from backend.triage import decide


@pytest.fixture
def settings():
    return Settings()


def make_prediction(
    grade=0,
    confidence=0.9,
    uncertainty=0.05,
    ungradable=0.02,
):
    probabilities = [0.02] * 5
    probabilities[grade] = confidence
    return Prediction(
        grade=grade,
        grade_name=f"stage {grade}",
        expected_grade=float(grade),
        grade_probabilities=probabilities,
        confidence=confidence,
        uncertainty=uncertainty,
        entropy=0.4,
        probability_ungradable=ungradable,
        probability_any_dr=0.5,
        probability_referable=0.3,
        mc_samples=20,
        inference_ms=190.0,
    )


# -- rule 1: image quality ---------------------------------------------------


def test_poor_quality_asks_for_a_new_photograph(settings):
    decision = decide(make_prediction(ungradable=0.82), settings)
    assert decision.action == "recapture"
    assert decision.rule == 1
    assert decision.urgency == "blocked"


def test_quality_outranks_severity(settings):
    """An unreadable photo of a severe eye is still an unreadable photo."""
    decision = decide(make_prediction(grade=4, ungradable=0.9), settings)
    assert decision.action == "recapture"


# -- rule 2: certainty -------------------------------------------------------


def test_disagreement_between_passes_goes_to_a_human(settings):
    decision = decide(make_prediction(grade=2, uncertainty=0.8), settings)
    assert decision.action == "human"
    assert decision.rule == 2


def test_low_confidence_goes_to_a_human(settings):
    decision = decide(make_prediction(grade=2, confidence=0.31), settings)
    assert decision.action == "human"
    assert "31%" in decision.reason


def test_borderline_quality_tightens_the_confidence_bar(settings):
    """
    Confidence of 0.60 clears the plain 0.55 bar, but not when the photo is
    already close to being rejected on quality.
    """
    clean = decide(make_prediction(grade=1, confidence=0.60, ungradable=0.02), settings)
    assert clean.action == "routine"

    murky = decide(make_prediction(grade=1, confidence=0.60, ungradable=0.34), settings)
    assert murky.action == "human"
    assert murky.rule == 2


def test_certainty_outranks_severity(settings):
    decision = decide(make_prediction(grade=4, uncertainty=0.9), settings)
    assert decision.action == "human"


# -- rule 3: referable disease -----------------------------------------------


@pytest.mark.parametrize("grade", [2, 3, 4])
def test_referable_stages_go_to_ophthalmology(settings, grade):
    decision = decide(make_prediction(grade=grade), settings)
    assert decision.action == "ophthalmology"
    assert decision.rule == 3


def test_severe_and_proliferative_are_marked_urgent(settings):
    assert decide(make_prediction(grade=2), settings).urgency == "soon"
    assert decide(make_prediction(grade=3), settings).urgency == "urgent"
    assert decide(make_prediction(grade=4), settings).urgency == "urgent"


# -- rule 4: everything else -------------------------------------------------


@pytest.mark.parametrize("grade", [0, 1])
def test_non_referable_stages_get_a_routine_rescreen(settings, grade):
    decision = decide(make_prediction(grade=grade), settings)
    assert decision.action == "routine"
    assert decision.rule == 4
    assert decision.urgency == "routine"


# -- configurability ---------------------------------------------------------


def test_lowering_the_referral_grade_catches_mild_disease():
    """A clinic willing to over-refer can move the bar down to stage 1."""
    strict = Settings()
    strict.referral_grade = 1
    assert decide(make_prediction(grade=1), strict).action == "ophthalmology"
    assert decide(make_prediction(grade=0), strict).action == "routine"


def test_raising_the_quality_bar_rejects_more_photographs():
    fussy = Settings()
    fussy.ungradable_threshold = 0.15
    assert decide(make_prediction(ungradable=0.25), fussy).action == "recapture"


def test_every_decision_explains_itself(settings):
    cases = [
        make_prediction(ungradable=0.9),
        make_prediction(uncertainty=0.9),
        make_prediction(grade=3),
        make_prediction(grade=0),
    ]
    for prediction in cases:
        decision = decide(prediction, settings)
        assert len(decision.reason) > 30, "a clinician needs a reason, not a code"
        assert decision.rule in {1, 2, 3, 4}
