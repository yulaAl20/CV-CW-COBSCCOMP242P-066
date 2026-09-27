"""
Triage policy.

The rules are checked in order - quality first, then certainty, then severity:

    1. the photo is not gradable          -> re-capture the image
    2. the model is not certain enough    -> send to a human grader
    3. the stage is referable (>= 2)      -> refer to ophthalmology
    4. otherwise                          -> routine rescreen

Thresholds come from Settings and match Section 6.2 of the training notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TRIAGE_ACTIONS, Settings


@dataclass
class TriageDecision:
    action: str          # the stable key, e.g. "ophthalmology"
    label: str           # the wording shown to the user
    reason: str          # why this rule fired, in plain language
    urgency: str         # "routine" | "soon" | "urgent" | "blocked"
    rule: int            # which of the four rules decided it

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "label": self.label,
            "reason": self.reason,
            "urgency": self.urgency,
            "rule": self.rule,
        }


def decide(prediction, settings: Settings) -> TriageDecision:
    """Apply the four rules to one prediction."""
    p_ungradable = prediction.probability_ungradable
    confidence = prediction.confidence
    uncertainty = prediction.uncertainty
    grade = prediction.grade

    borderline_quality = p_ungradable > 0.6 * settings.ungradable_threshold

    # Rule 1 - image quality
    if p_ungradable > settings.ungradable_threshold:
        return TriageDecision(
            action="recapture",
            label=TRIAGE_ACTIONS["recapture"],
            reason=(
                f"The image quality check scores this photo {p_ungradable:.0%} likely "
                f"to be ungradable, above the {settings.ungradable_threshold:.0%} limit. "
                "Take another photograph before grading."
            ),
            urgency="blocked",
            rule=1,
        )

    # Rule 2 - certainty
    uncertain = uncertainty > settings.uncertainty_threshold
    unconfident = confidence < settings.confidence_threshold
    borderline_combination = borderline_quality and confidence < settings.confidence_threshold + 0.10

    if uncertain or unconfident or borderline_combination:
        if uncertain:
            reason = (
                f"Repeated predictions disagreed with each other (spread {uncertainty:.2f} "
                f"on the 0-4 severity scale, limit {settings.uncertainty_threshold:.2f})."
            )
        elif unconfident:
            reason = (
                f"The most likely stage only reached {confidence:.0%} probability, "
                f"below the {settings.confidence_threshold:.0%} required to report it."
            )
        else:
            reason = (
                "Image quality is borderline and confidence is low, so the stage is "
                "not reliable enough to report."
            )
        return TriageDecision(
            action="human",
            label=TRIAGE_ACTIONS["human"],
            reason=reason + " A human grader should read this image.",
            urgency="soon",
            rule=2,
        )

    # Rule 3 - referable disease
    if grade >= settings.referral_grade:
        return TriageDecision(
            action="ophthalmology",
            label=TRIAGE_ACTIONS["ophthalmology"],
            reason=(
                f"Graded {prediction.grade_name} with {confidence:.0%} confidence. "
                f"Stage {settings.referral_grade} and above is referable disease."
            ),
            urgency="urgent" if grade >= 3 else "soon",
            rule=3,
        )

    # Rule 4 - everything else
    return TriageDecision(
        action="routine",
        label=TRIAGE_ACTIONS["routine"],
        reason=(
            f"Graded {prediction.grade_name} with {confidence:.0%} confidence. "
            "No referable disease found; screen again at the normal interval."
        ),
        urgency="routine",
        rule=4,
    )
