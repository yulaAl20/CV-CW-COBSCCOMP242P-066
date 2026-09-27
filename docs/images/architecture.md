# Architecture

## The shape of the system

![System architecture](images/fig_architecture_system.png)

One FastAPI process serves both the interface and the JSON API. That is a
deliberate choice rather than a shortcut: a split frontend and backend would
mean a second deployment target, a CORS policy, and two sets of environment
variables to keep in step, none of which buys anything for a single-page tool
that one person uses at a time.

```
Browser ──upload──▶ FastAPI ──▶ preprocessing ──▶ ONNX backbone (once)
                                                       │
                                                       ▼
                                              MC dropout heads × 20
                                                       │
                                        ┌──────────────┼──────────────┐
                                        ▼              ▼              ▼
                                   CORN decode    quality head    CAM heatmap
                                        └──────────────┼──────────────┘
                                                       ▼
                                                 triage rules
                                                       │
Browser ◀──────────────────── one JSON response ───────┘
```

## Why the model is split in two

The exported model is not one file but two: an ONNX graph for the backbone and
a NumPy archive for the heads.

The reason is uncertainty. The app reports how much twenty stochastic forward
passes disagreed with each other, because a screening tool that cannot say
"I am not sure" is not safe to deploy. Running the whole network twenty times
would cost about 3.8 seconds per image on a CPU.

But dropout in this network sits *after* the backbone, immediately before the
two linear heads. Everything upstream of the dropout layer is deterministic, so
it only has to run once:

```python
features = backbone(image)          # 190 ms, once
for _ in range(20):                 # ~0.4 ms total
    dropped = dropout(features)
    grade_logits = grade_head(dropped)
```

Twenty samples therefore cost about 190 ms rather than 3,800 ms. The split is
not a packaging detail; it is what makes the uncertainty estimate practical.

## The network

![Model architecture](images/fig_architecture_model.png)

| Part | Detail |
|---|---|
| Backbone | EfficientNetV2-S, pretrained on ImageNet-21k, fine-tuned on ImageNet-1k |
| Parameters | 20.2M in the backbone, 7,686 in the two heads |
| Input | 384 × 384 × 3, ImageNet-normalised |
| Features | 1280-dimensional pooled vector |
| Grade head | Linear 1280 → 4, ordinal (CORN) |
| Quality head | Linear 1280 → 2, gradable / ungradable |
| Dropout | p = 0.4, kept active at inference for uncertainty |

### Why an ordinal head instead of five-way softmax

Diabetic retinopathy stages are ordered. Calling a proliferative eye "No DR" is
a far worse error than calling it "Severe", but cross-entropy scores both
identically because it treats the five stages as unrelated labels.

The CORN formulation replaces the five-way decision with four binary ones:
*is this worse than stage 0? worse than 1? worse than 2? worse than 3?* The
answers are chained through a cumulative product, which has two consequences:

1. **The output cannot contradict itself.** P(grade > k) can never rise as k
   rises, so the model is incapable of saying "probably worse than severe but
   probably not worse than mild". A softmax has no such guarantee.
2. **The loss scales with distance.** Measured on the trained model: predicting
   3 when the truth is 4 costs 1.018, predicting 0 costs 4.018.

It also yields a continuous severity score — the sum of the four cumulative
probabilities — which is what the interface plots on the ladder. A case scoring
1.9 is visibly sitting on the mild/moderate boundary; a discrete label would
have hidden that.

## Preprocessing

![Preprocessing pipeline](images/fig_pipeline_preprocessing.png)

Six steps, in `backend/preprocessing.py`, ported unchanged from the training
notebook:

| Step | Problem it solves | Measured effect |
|---|---|---|
| Crop to retina | Black letterbox wastes input resolution | ~25% of pixels removed |
| Pad to square | Non-square resize distorts lesion shape | shape preserved |
| Resize to 384 | Cameras range from 0.26 to 8.0 megapixels | one input size |
| CLAHE on green | Lesions are lowest-contrast where they matter | green-channel σ 46.0 → 48.5 |
| Subtract local average | Flash leaves a brightness gradient across the retina | gradient reduced 88% on average |
| Circular mask | Bright rim at the field-of-view edge is an artefact | rim removed |

**These functions must not drift from the training pipeline.** If serving
preprocesses differently from training, every number in `docs/evaluation.md`
stops describing the deployed app. `tests/test_preprocessing.py` pins the
properties that matter — output shape, normalisation constants, and that the
illumination step actually flattens a gradient.

## Triage

![Triage flow](images/fig_triage_flow.png)

A bare stage label is not a usable output for a clinic, because it looks
equally confident for a textbook proliferative case and for a blurred
photograph of the wrong eye. `backend/triage.py` converts the model's numbers
into one of four actions, checked in order:

| Rule | Condition | Action |
|---|---|---|
| 1 | P(ungradable) > 0.50 | Re-capture image |
| 2 | spread > 0.45, or confidence < 0.55 | Refer for human grading |
| 3 | predicted stage ≥ 2 | Refer to ophthalmology |
| 4 | otherwise | Routine rescreen |

The ordering is the clinically meaningful part. Quality is checked before
severity because an unreadable photograph of a severe eye is still an
unreadable photograph — grading it would produce a confident number from
essentially no evidence. Certainty is checked before severity for the same
reason in a subtler form.

Rule 2 also has a compound clause: a photograph that is *close* to the quality
cut-off raises the confidence bar by ten points rather than being rejected
outright. Borderline quality and borderline confidence together are worse than
either alone, and the rule reflects that.

Every response carries the rule number that fired, so a decision can be audited
after the fact rather than reverse-engineered.

## Explainability

The heatmap is a Class Activation Map, not Grad-CAM, and this is worth
explaining because the training notebook used Grad-CAM.

The network ends in global average pooling followed by a linear head. For that
topology the CAM identity holds exactly: the contribution of each spatial
position to the pooled score is the head weights dotted with the feature vector
at that position. No backward pass is required.

That matters because an ONNX graph has no backward pass. Grad-CAM cannot run on
the deployed model without shipping PyTorch, which would take the container from
about 600 MB to roughly 4 GB. CAM gives the same map for this architecture at
no cost.

The quantity being explained is the expected severity, matching the notebook.
`backend/explain.py` computes the sensitivity of that sum to each feature
channel through the cumulative product, then weights the feature map by it.

An occlusion-based fallback exists for architectures where CAM does not apply.
It needs 64 forward passes — roughly twelve seconds — so it is a demonstration
tool, not a serving path.

## Request flow

| Stage | Cost | Notes |
|---|---|---|
| Upload validation | < 1 ms | size, decodability, minimum dimensions |
| Fundus sanity check | ~3 ms | rejects obvious non-fundus uploads with a warning |
| Preprocessing | ~45 ms | six OpenCV operations |
| Backbone | ~190 ms | ONNX Runtime, CPU, the dominant cost |
| MC dropout heads × 20 | < 1 ms | pure NumPy on a 1280-vector |
| CORN decode + triage | < 1 ms | |
| CAM heatmap | ~25 ms | only when `backbone_cam.onnx` is present |
| JPEG encoding | ~20 ms | eight images when the pipeline view is on |

About 280 ms end to end on two vCPUs, which is well inside the threshold where
an interface feels responsive without needing a progress indicator — though
one is shown anyway, because free hosting tiers are slower than a laptop.

## Directory layout

```
backend/
  config.py          settings and thresholds, all env-overridable
  preprocessing.py   the six-step pipeline, ported from the notebook
  inference.py       ONNX session, MC dropout, CORN decoding
  explain.py         CAM and the occlusion fallback
  triage.py          the four rules
  metrics_store.py   reads recorded results from artifacts/
  demo.py            synthetic fallback when weights are absent
  schemas.py         API contract
  main.py            routes and static file serving

streamlit_app.py     Streamlit entry point: st.navigation over views/
views/
  reader.py          Streamlit reading page
  evidence.py        Streamlit evidence page
streamlit_ui.py      shared Streamlit theme, cached model loading, components

frontend/
  index.html         the reading page: upload, readout, triage
  evidence.html      measured performance, on its own page
  css/styles.css     design tokens and layout for both pages
  js/grades.js       the severity palette and formatters, shared
  js/api.js          one place where fetch happens
  js/charts.js       hand-drawn SVG charts
  js/app.js          reading-page logic
  js/evidence.js     evidence-page logic

artifacts/           recorded metrics from the training run
models/              exported weights (not in git — see models/README.md)
scripts/             export, download, figure generation
tests/               44 tests
docs/                this documentation and the figures
```

## Choices worth defending

**ONNX instead of TorchScript.** It removes the torch dependency at serving
time entirely. The container is about 600 MB rather than 4 GB, which is the
difference between fitting on a free tier and not.

**fp32 instead of int8.** The int8 export is a quarter of the size but, measured,
1.8× *slower* on CPU — dynamic quantisation adds per-operator conversion
overhead that a convolutional backbone does not repay. The int8 file is still
produced for anyone whose binding constraint is disk rather than latency.

**No database.** The app holds no state between requests. Nothing is written to
disk, which removes an entire class of privacy problem: a retinal photograph is
biometric data, and the safest place to store it is nowhere.

**Hand-drawn SVG charts.** Four small plots do not justify a charting library,
and inline SVG inherits the page's own colour tokens for free.

**Evidence on a separate page.** The test-set results used to sit at the bottom
of the reading page. That made the reading page long, and it put a wall of
aggregate statistics underneath a single patient's result, where it was both
irrelevant to the task at hand and too far down to be read. The two pages
answer different questions — *what is wrong with this eye* and *should I
believe this model at all* — and they are consulted by different people at
different times. Splitting them means the reading page is one screen and the
evidence is somewhere a reviewer will actually finish.

**One uvicorn worker.** Each worker loads its own 81 MB ONNX session. Two
workers on a 512 MB instance get killed by the OOM reaper. Scale with more
containers, not more workers.

**Two front ends rather than one.** Streamlit is the easy hosting path — no
Dockerfile, deploys from a GitHub push, free. FastAPI is the one with a JSON
API, which the notebook and any other client can call. Keeping both is cheap
*only because* `backend/` has no web framework in it: preprocessing, inference,
triage and the metrics store are imported by both and written once. The moment
grading logic appeared in a page file, the two could drift and the evaluation
would stop describing one of them — `tests/test_streamlit.py` asserts it has
not happened.

**The answer first, the numbers behind a reveal.** The reading page opens with
the action in plain words — "Send this patient to an eye specialist" — and the
three screening questions answered yes/no. The severity ladder, per-stage
probabilities, entropy, MC spread, the rule that fired and the preprocessing
stages all sit inside one expander. Someone working a clinic list needs a
decision; the probability vector is the evidence for that decision, and putting
the two side by side made the evidence compete with the answer. Progressive
disclosure, not omission: everything is one click away and nothing was removed,
because a reading nobody can check is worse than no reading. `test_streamlit.py`
asserts the reveal exists, starts collapsed, and still contains the figures.

**History in session state, never on disk.** The sidebar keeps the last ten
readings so cases can be compared without re-uploading, and clicking one
re-displays it without re-running inference. It lives in `st.session_state`,
which means it lasts as long as the browser tab and is gone on refresh. That is
the deliberate scope: the model card promises retinal photographs are stored
nowhere, because they are biometric identifiers under GDPR Article 9, and a
history written to disk would quietly break that promise. Entries hold
JPEG-encoded bytes rather than arrays — ten raw 384×384 entries would be about
40 MB of session state against under 2 MB as JPEG, which matters on a 1 GB host
where the ONNX session has already taken its share.

**`st.cache_resource` for the model, not `st.cache_data`.** The ONNX session is
a live object holding ~80 MB, not a value to copy. Without it, Streamlit would
reload the model on every checkbox click and each interaction would cost
several seconds.
