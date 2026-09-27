# Model card

## What this is

A convolutional classifier that grades retinal fundus photographs on the
five-stage international diabetic retinopathy scale, estimates how uncertain it
is, and routes each case to one of four next steps.

| | |
|---|---|
| Version | 1.2.0 |
| Architecture | EfficientNetV2-S, ImageNet-21k pretrained, two heads |
| Input | one fundus photograph, any resolution, resized to 384 × 384 |
| Output | stage 0–4, continuous severity 0.0–4.0, confidence, uncertainty, quality estimate, triage action |
| Training data | DDR, 8,765 images |
| Licence | MIT (code). The datasets carry their own terms. |

## What it is not

**This is not a medical device.** It is a student prototype built for a
university coursework assignment. It has not been through clinical validation,
regulatory review, or prospective testing. It must not be used to make
decisions about a real patient.

## Intended use

Demonstrating an end-to-end deep learning system: preprocessing, transfer
learning, ordinal classification, uncertainty quantification, evaluation, and
deployment. The triage framing exists because it makes the uncertainty estimate
do visible work, not because the system is ready for a clinic.

## Out of scope

- Any clinical decision about a real patient
- Grading images from a capture device unlike those in DDR
- Detecting anything other than diabetic retinopathy. Glaucoma, AMD, retinal
  detachment and hypertensive retinopathy are invisible to this model; it will
  return a DR stage for an eye with any of them and say nothing about the actual
  pathology
- Paediatric or non-diabetic populations
- Operating without a human in the loop

## Limitations, in order of how much they would matter in practice

### 1. The quality gate has never seen a bad photograph

This is the most serious gap. The DDR copy used for training contained **zero
ungradable images**, so the quality head was trained with no negative examples.

Consequences:
- P(ungradable) is effectively uncalibrated and stays near zero
- Triage rule 1 never fired once across all 1,879 test images
- A genuinely unreadable photograph will be graded confidently instead of
  rejected

The rule and the head both exist and are wired up. What is missing is data.
Fixing it means sourcing the ungradable class from the full DDR release, or
synthesising defocus, over-exposure and partial-field failures.

### 2. Performance drops on unfamiliar data

On APTOS 2019 — different country, cameras and population — macro F1 falls from
0.728 to 0.456.

The shape of the drop matters. Referable-DR AUC falls only 0.017 (0.981 → 0.964)
and DR-detection recall is 1.000. The model still separates diseased from
healthy; what it loses is agreement on where stage boundaries sit. On APTOS it
over-predicts moderate NPDR, sweeping neighbouring stages into that bin.

Practical reading: the triage decision generalises considerably better than the
stage label. Do not report the stage as a finding outside DDR-like settings
without recalibrating on local data.

### 3. Mild NPDR is unreliable

F1 of 0.380 internally, 0.079 externally.

Stage 1 is defined by microaneurysms alone — features around 50 µm wide, near
the resolution limit at 384 px. It is also where human graders disagree most.
With 441 training examples it is both the rarest and the subtlest class.

The model frequently confuses stage 0, 1 and 2 with each other. Since the
referral boundary sits between 1 and 2, this is where triage errors concentrate:
72 moderate cases in the test set were sent for routine rescreen.

### 4. Severe NPDR estimates are noisy

35 test images. One image moves recall by about three points, so the reported
0.486 recall carries a wide confidence interval. 17 of 35 severe cases were
called moderate — both referable, so the action is unchanged, but the stage is
wrong.

### 5. The model is overconfident

ECE 0.0751. When it reports 90% confidence it is right about 82% of the time.
The confidence threshold of 0.55 is set low partly to compensate. Temperature
scaling on the validation set would address most of this and has not been done.

### 6. Single-image, single-field

Clinical DR grading uses multiple fields per eye and both eyes together. This
model sees one photograph in isolation and cannot use the contralateral eye,
prior visits, or fields outside the one captured.

### 7. Unknown demographic performance

DDR comes from 147 Chinese hospitals. No per-ethnicity, per-age or per-sex
breakdown was performed because the metadata is not in the dataset. Fundus
pigmentation varies substantially across populations and affects image
statistics, so performance may differ across groups in ways this evaluation
cannot detect.

## Training

| | |
|---|---|
| Epochs | 20 of 25 configured, stopped early on validation QWK |
| Batch size | 32 |
| Optimiser | AdamW, weight decay 1e-5 |
| Learning rates | 1e-4 backbone, 1e-3 heads |
| Schedule | 300-step warmup, then cosine |
| Loss | CORN ordinal + 0.3 × weighted cross-entropy for quality |
| Class balance | effective-number weights, β = 0.999, with a balanced sampler |
| Augmentation | flips, rotation ±180°, zoom 0.9–1.1, shift ±5%, brightness/contrast ±12%, occasional blur or noise |
| Regularisation | dropout 0.4, label smoothing 0.05, EMA 0.999, gradient clipping 1.0 |
| Model selection | highest validation QWK (epoch 15, 0.8960) |
| Hardware | Tesla T4, 74 minutes |

The backbone is frozen for epoch 0 while the randomly initialised heads learn,
then unfrozen. Without this the large gradients from the untrained heads would
flow back and damage the pretrained features in the first few hundred steps.

## Thresholds

| Threshold | Value | Effect of raising it |
|---|---|---|
| `DR_UNGRADABLE_THRESHOLD` | 0.50 | fewer re-capture requests, more low-quality images graded |
| `DR_UNCERTAINTY_THRESHOLD` | 0.45 | fewer cases to human graders, more uncertain calls reported |
| `DR_CONFIDENCE_THRESHOLD` | 0.55 | more cases to human graders, fewer low-confidence calls reported |
| `DR_REFERRAL_GRADE` | 2 | raising to 3 refers fewer patients and misses moderate disease |

These are settable through environment variables, which is a convenience for
retuning, not an invitation. They were chosen on the validation split and their
behaviour is characterised in `evaluation.md`. Changing them changes clinical
behaviour and invalidates the measured triage distribution.

## Privacy

The app holds no state. Uploaded images are decoded in memory, processed, and
discarded when the request ends. Nothing is written to disk, no database exists,
and no logging captures image content — only a request ID, filename, predicted
stage, action and duration.

This is deliberate. Retinal photographs are biometric identifiers under GDPR
Article 9 and comparable regimes. The safest place to store them is nowhere.

If you add persistence, you take on responsibility for lawful basis, retention
limits, access control and erasure rights.

## Ethical considerations

**Automation bias.** People defer to confident-looking machine output. The
interface shows the full probability distribution, the uncertainty band and the
rule that fired, specifically so the reader sees when the model is guessing.

**Failure asymmetry.** A missed proliferative case can end in blindness; an
unnecessary referral costs an appointment. Thresholds are set to over-refer, and
the reported operating points are chosen accordingly.

**Access.** Tools like this are often justified by extending screening where
ophthalmologists are scarce. Those are exactly the settings with the most
distribution shift from DDR and the least capacity to validate locally — the
APTOS result is a direct warning about that.

## Citation

```
DDR:   Li et al. (2019), "Diagnostic assessment of deep learning algorithms
       for diabetic retinopathy screening", Information Sciences 501, 511-522.
APTOS: APTOS 2019 Blindness Detection, Kaggle.
CORN:  Shi, Cao & Raschka (2023), "Deep Neural Networks for Rank-Consistent
       Ordinal Regression Based On Conditional Probabilities".
```
