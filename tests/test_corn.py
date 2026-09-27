"""
Tests for the ordinal (CORN) decoding.

The reason the model uses an ordinal head instead of a five-way softmax is that
DR stages are ordered. These tests check the two properties that buys: the
predicted probabilities can never contradict the ordering, and the severity
score behaves like a position on the scale rather than a class index.
"""

import numpy as np
import pytest

from backend.inference import (
    corn_cumulative_probs,
    cumulative_to_grade_probs,
    sigmoid,
    softmax,
)


def test_sigmoid_matches_its_definition():
    x = np.array([-4.0, 0.0, 4.0])
    assert np.allclose(sigmoid(x), 1 / (1 + np.exp(-x)))
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_cumulative_probabilities_never_increase():
    """P(grade > k) must fall as k rises - this is what the cumprod guarantees."""
    rng = np.random.default_rng(7)
    logits = rng.normal(0, 3, (200, 4))
    cumulative = corn_cumulative_probs(logits)
    differences = np.diff(cumulative, axis=1)
    assert (differences <= 1e-9).all(), "an ordering violation slipped through"


def test_cumulative_probabilities_stay_in_range():
    logits = np.array([[10.0, 10.0, 10.0, 10.0], [-10.0, -10.0, -10.0, -10.0]])
    cumulative = corn_cumulative_probs(logits)
    assert (cumulative >= 0).all() and (cumulative <= 1).all()


def test_grade_probabilities_sum_to_one():
    rng = np.random.default_rng(11)
    cumulative = corn_cumulative_probs(rng.normal(0, 2, (50, 4)))
    probabilities = cumulative_to_grade_probs(cumulative)
    assert probabilities.shape == (50, 5)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert (probabilities >= -1e-9).all()


def test_confident_no_dr_puts_all_mass_on_stage_zero():
    cumulative = corn_cumulative_probs(np.array([[-9.0, -9.0, -9.0, -9.0]]))
    probabilities = cumulative_to_grade_probs(cumulative)[0]
    assert probabilities.argmax() == 0
    assert probabilities[0] > 0.99


def test_confident_proliferative_puts_all_mass_on_stage_four():
    cumulative = corn_cumulative_probs(np.array([[9.0, 9.0, 9.0, 9.0]]))
    probabilities = cumulative_to_grade_probs(cumulative)[0]
    assert probabilities.argmax() == 4
    assert probabilities[4] > 0.99


@pytest.mark.parametrize("target", [0, 1, 2, 3, 4])
def test_each_stage_is_reachable(target):
    """Logits set just above the target should decode back to that stage."""
    logits = np.full((1, 4), -8.0)
    logits[0, :target] = 8.0
    grade = int((corn_cumulative_probs(logits)[0] > 0.5).sum())
    assert grade == target


def test_severity_score_tracks_the_stage():
    """The continuous score should rise monotonically as the stage rises."""
    severities = []
    for target in range(5):
        logits = np.full((1, 4), -8.0)
        logits[0, :target] = 8.0
        severities.append(float(corn_cumulative_probs(logits).sum()))
    assert severities == sorted(severities)
    assert severities[0] < 0.1 and severities[-1] > 3.9


def test_severity_score_sits_between_stages_when_the_model_hesitates():
    """A case the model is torn between stage 1 and 2 should score near 1.5."""
    logits = np.array([[6.0, 0.0, -6.0, -6.0]])
    severity = float(corn_cumulative_probs(logits).sum())
    assert 1.3 < severity < 1.7


def test_softmax_is_a_distribution():
    values = softmax(np.array([[2.0, 1.0, 0.1]]))
    assert np.isclose(values.sum(), 1.0)
    assert values.argmax() == 0
