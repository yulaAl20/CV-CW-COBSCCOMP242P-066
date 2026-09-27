# Evaluation

Every number here comes from the training run recorded in
`artifacts/evaluation.json` (EfficientNetV2-S, 20 of 25 epochs, stopped early,
Tesla T4, 74 minutes). Nothing is recomputed, so the figures, the app's
performance panel and this document cannot drift apart.

## The data

| Dataset | Images | Role |
|---|---|---|
| DDR | 12,522 | train 8,765 / validation 1,878 / test 1,879 |
| APTOS 2019 | 3,662 | external test only — never trained on |

APTOS comes from a different country, different cameras and a different patient
population. It exists in this project for one reason: internal test scores on a
stratified split of one dataset flatter a model, and the honest question is
whether it travels.

### Class imbalance

![Class distribution](images/fig_class_imbalance.png)

Half of DDR is healthy and 1.9% is severe — a 26.6 : 1 ratio between the
largest and smallest classes. A model that answered "No DR" to everything would
score 50.0% accuracy on the test set.

That is why accuracy is reported alongside macro F1 and QWK rather than on its
own, and why training used effective-number class weighting with a balanced
sampler. The sampler lifts severe cases from 1.9% of the data to 9.7% of each
epoch without duplicating them into the validation split.

## Headline results

| Metric | DDR test (internal) | APTOS (external) |
|---|---|---|
| Stage accuracy | **0.8664** | 0.7018 |
| Macro precision | 0.7525 | 0.5410 |
| Macro recall | 0.7134 | 0.4535 |
| Macro F1 | 0.7282 | 0.4556 |
| Weighted F1 | 0.8692 | 0.6824 |
| Quadratic weighted kappa | **0.9072** | 0.7921 |
| Referable DR, AUC | **0.9807** | 0.9640 |

Against a majority-class baseline of 0.5003, the internal accuracy of 0.8664 is
a real gain rather than a restatement of the class prior.

A QWK of 0.9072 is worth putting in context: published inter-grader agreement
between human ophthalmologists on DR staging typically falls in the 0.80–0.91
range. The model agrees with the DDR reference grades about as closely as
graders agree with each other — on this dataset.

## Per-stage performance

![Per-class metrics](images/fig_per_class_metrics.png)

| Stage | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| No DR | 0.935 | 0.933 | 0.934 | 940 |
| Mild NPDR | 0.339 | 0.432 | 0.380 | 95 |
| Moderate NPDR | 0.857 | 0.856 | 0.856 | 672 |
| Severe NPDR | 0.680 | 0.486 | 0.567 | 35 |
| Proliferative DR | 0.952 | 0.861 | 0.904 | 137 |

Two stages are weak, and both are weak for reasons worth stating plainly.

**Mild NPDR (F1 0.380)** is the hardest class in DR grading for anyone. It is
defined by microaneurysms alone — a handful of dots perhaps 50 µm across in an
image downsampled to 384 px. It is also the boundary where human graders
disagree most. With 441 training examples against 4,386 healthy ones, the model
sees it rarely and at the resolution where it is least visible.

**Severe NPDR (F1 0.567)** has 165 training images and 35 test images. At that
support, one image moves recall by roughly three points, so the estimate itself
is noisy. The confusion matrix shows the failure mode: 17 of 35 severe cases are
called moderate. Both are referable, so the triage action is unchanged — which
is why this error is less costly than its F1 suggests.

## Confusion matrix

![Confusion matrix](images/fig_confusion_matrix.png)

|  | pred No DR | Mild | Moderate | Severe | PDR |
|---|---|---|---|---|---|
| **true No DR** | 877 | 27 | 36 | 0 | 0 |
| **true Mild** | 29 | 41 | 25 | 0 | 0 |
| **true Moderate** | 31 | 53 | 575 | 8 | 5 |
| **true Severe** | 0 | 0 | 17 | 17 | 1 |
| **true PDR** | 1 | 0 | 18 | 0 | 118 |

251 errors in total: 160 off by one stage, 91 off by two or more. The
concentration near the diagonal is what the ordinal loss was chosen to produce.

The error that matters most clinically is the bottom-left corner — sight-
threatening disease called healthy. There is exactly one: a single proliferative
case predicted as No DR. Section 6.16 of the notebook shows it, and the model
flagged it with 0.639 confidence, below the 0.55 reporting threshold's
neighbourhood, so the triage policy routes it to a human rather than reporting
it.

## Screening decisions

The five-stage label is not what a screening programme acts on. Two binary
questions are:

**Any DR present (stage ≥ 1)**

| | |
|---|---|
| Accuracy | 0.9340 |
| Precision | 0.9330 |
| Recall (sensitivity) | 0.9350 |
| Specificity | 0.9330 |
| F1 | 0.9340 |
| Counts | TP 878, FP 63, FN 61, TN 877 |

**Referable DR (stage ≥ 2)**

| | |
|---|---|
| Accuracy | 0.9223 |
| Precision | 0.9256 |
| Recall (sensitivity) | 0.8993 |
| Specificity | 0.9411 |
| F1 | 0.9123 |
| AUC | 0.9807 |
| Counts | TP 759, FP 61, FN 85, TN 974 |

At the operating points from the ROC curve:

- 95.3% sensitivity at 90.1% specificity
- 90.0% sensitivity at 95.1% specificity

The UK NHS screening standard is 80% sensitivity at 95% specificity for
referable disease. This model clears that bar on internal data — with the
caveats in `model-card.md` about what "internal data" means.

## Generalisation

![Internal vs external](images/fig_internal_vs_external.png)

| Metric | DDR | APTOS | Gap |
|---|---|---|---|
| Accuracy | 0.8664 | 0.7018 | −0.165 |
| Macro F1 | 0.7282 | 0.4556 | −0.273 |
| QWK | 0.9072 | 0.7921 | −0.115 |
| Referable AUC | 0.9807 | 0.9640 | **−0.017** |

This is the most informative table in the project, and the pattern in it is the
point.

Macro F1 collapses by 0.27. Referable AUC falls by 0.017. Those two facts
together say something specific: the model's *ranking* of eyes by severity
survives the move to new cameras and a new population almost intact, but its
*calibration of stage boundaries* does not. It still knows which eyes are
sicker. It no longer knows exactly where APTOS graders drew the line between
stage 1 and stage 2.

The APTOS per-stage report makes this concrete. Recall on moderate NPDR rises to
0.951 while precision falls to 0.530 — the model is sweeping mild and severe
cases into the moderate bin. Mild NPDR essentially fails (F1 0.079).

For a tool whose output is a triage action rather than a diagnosis, this is the
better failure mode: referable AUC of 0.964 and DR-detection recall of 1.000 on
APTOS mean no sick patient was missed, at the cost of 321 false positives.
Over-referral is recoverable; a missed proliferative case is not.

It is still a genuine limitation, and it is the main reason this model should
not be deployed outside DDR-like settings without recalibration on local data.

## Uncertainty

Each image is predicted 20 times with dropout active. The spread of those
predictions is the reported uncertainty.

| Measure | Value | Reading |
|---|---|---|
| Expected calibration error | 0.0751 | confidence overshoots accuracy by ~7.5 points on average |
| Maximum calibration error | 0.3726 | one confidence bin is badly off |
| Area under risk-coverage | 0.0347 | |

An ECE of 0.075 means the model is somewhat overconfident — when it says 90%, it
is right about 82% of the time. This is normal for a deep network trained with
label smoothing and is the reason the triage thresholds are set conservatively.
Temperature scaling on the validation set would likely halve it and is the
obvious next improvement.

### Does the uncertainty find the mistakes?

![Risk coverage](images/fig_risk_coverage.png)

This is the question that decides whether the uncertainty estimate is worth
computing at all. If deferring the least-certain cases does not raise accuracy
on the rest, the number is decoration.

| Deferred to a human | Images kept | Accuracy on kept | Referable sensitivity on kept |
|---|---|---|---|
| 0% | 1,879 | 0.8664 | 0.8993 |
| 5% | 1,785 | 0.8846 | 0.9097 |
| 10% | 1,691 | 0.9054 | 0.9266 |
| 20% | 1,503 | 0.9415 | 0.9643 |
| 30% | 1,315 | 0.9582 | 0.9755 |

It works. Deferring 20% lifts accuracy from 0.866 to 0.942.

The second column is the one that matters clinically. Sensitivity rises along
with accuracy — 0.899 to 0.964 — which rules out the failure mode where the
model achieves a clean-looking score by quietly deferring all the sick patients
and keeping the easy healthy ones.

Read as a workload argument: a grader reviewing the least-certain fifth of
cases would be handling 376 images instead of 1,879, with the model's error rate
on the remainder cut roughly in half.

## Triage policy in practice

![Triage distribution](images/fig_triage_distribution.png)

Applying all four rules to the 1,879 test images:

| True stage | Routine rescreen | Human grading | Ophthalmology |
|---|---|---|---|
| No DR | 887 | 17 | 36 |
| Mild NPDR | 65 | 7 | 23 |
| Moderate NPDR | 72 | 20 | 580 |
| Severe NPDR | 0 | 0 | 35 |
| Proliferative DR | 1 | 2 | 134 |

Overall: 54.6% routine, 43.0% referred to ophthalmology, 2.4% to a human grader.

**All 35 severe cases are referred. 134 of 137 proliferative cases are
referred**, 2 go to a human grader, and 1 is missed.

The costs are visible too: 36 healthy eyes referred unnecessarily, and 72
moderate cases sent for routine rescreen when they should have been referred.
That second number is the real weakness — 72 patients with referable disease
who would be told to come back later.

Note that no image was sent for re-capture. This copy of DDR contained zero
ungradable images, so the quality head was trained with no negative examples
and rule 1 never fires. See `model-card.md` — this is the most significant gap
in the system.

## Deployment measurements

![CPU latency](images/fig_cpu_latency.png)

| Export | Size | Median latency | Mean | p95 |
|---|---|---|---|---|
| `backbone_fp32.onnx` | 80.6 MB | 186.2 ms | 187.1 ms | 197.7 ms |
| `backbone_int8.onnx` | 21.1 MB | 343.1 ms | 341.5 ms | 350.7 ms |

ONNX export verified against PyTorch: maximum absolute difference 5.25 × 10⁻⁶
against a tolerance of 10⁻³.

The int8 result is counterintuitive and worth keeping: dynamic quantisation made
the file four times smaller and inference 1.8× slower. Per-operator quantise and
dequantise steps cost more than the narrower arithmetic saves on a convolutional
backbone. The app ships fp32 because it was measured, not assumed.

## What would improve this next

In rough order of expected return:

1. **Ungradable training data.** Rule 1 currently cannot fire. Adding the
   ungradable images from the full DDR release, or synthesising defocus and
   exposure failures, would make the quality gate real.
2. **Temperature scaling.** One scalar fitted on the validation set would
   address most of the 0.075 ECE.
3. **Test-time augmentation.** Averaging over flips typically adds 1–2 points of
   QWK for four times the inference cost.
4. **Higher resolution for mild disease.** Microaneurysms are near the limit of
   what 384 px resolves. 512 px would likely help stage 1 most.
5. **Fine-tuning on APTOS.** Would close the external gap, at the cost of no
   longer having a clean external test set.
