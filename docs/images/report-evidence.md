# Capturing graphic evidence for the report

The assignment asks for graphical evidence with supporting explanations at
every step. This is a checklist of what to capture and, more importantly, what
to write underneath each item — a screenshot with no argument attached earns
very little.

Twelve figures are already generated for you in `docs/images/`. The rest are
screenshots you take once the app is running.

---

## Already generated — just reference them

Run `python scripts/make_figures.py` and these appear in `docs/images/`:

| File | Use it for |
|---|---|
| `fig_architecture_system.png` | the deployment section |
| `fig_architecture_model.png` | the network design section |
| `fig_pipeline_preprocessing.png` | the preprocessing section |
| `fig_triage_flow.png` | the triage / decision section |
| `fig_class_imbalance.png` | the dataset section |
| `fig_confusion_matrix.png` | the evaluation section |
| `fig_per_class_metrics.png` | precision / recall / F1 — required by the brief |
| `fig_training_curves.png` | accuracy and loss curves — required by the brief |
| `fig_internal_vs_external.png` | generalisation |
| `fig_risk_coverage.png` | uncertainty |
| `fig_triage_distribution.png` | clinical behaviour |
| `fig_cpu_latency.png` | deployment trade-offs |

They are 160 dpi and sized for a two-column report at full width.

---

## Which app to screenshot

The repository has two front ends over one engine. Your report says **Streamlit
app**, so screenshot that one — `streamlit run streamlit_app.py`, or the hosted
Streamlit Cloud URL. The FastAPI service is worth one sentence in the
deployment section and one screenshot of `/docs`, as evidence that the engine
is reachable as an API too.

## Screenshots to take

### 1. The landing view

Load the app with the model badge showing **Model ready · 80.6 MB**.

*Write about:* the badge proves real weights are loaded rather than demo mode,
which is the difference between a working system and a mock-up.

### 2. A healthy eye graded

Upload a stage 0 image. Capture the whole result panel: the ladder, the
severity marker near 0, the triage card reading **Routine rescreen**.

*Write about:* what the marker position means, and why a continuous score is
shown alongside the discrete stage.

### 3. A referable case

Upload a moderate or worse image. Capture the same region.

*Write about:* rule 3 fired, the card says so, and the border colour changes
with urgency. Contrast it with screenshot 2 — same interface, different
decision, and the reason is stated rather than implied.

### 4. An uncertain case

Find an image where the app routes to **Refer for human grading**. Boundary
cases between stage 1 and 2 are the most likely candidates.

*Write about:* this is the screenshot that justifies the whole uncertainty
apparatus. The model declined to answer, and said why. Pair it with
`fig_risk_coverage.png` to show that declining actually improves accuracy on
what remains.

### 5. The preprocessing pipeline

Open the **Pipeline** tab with a case loaded. Six panels, in order.

*Write about:* each step and the problem it solves. The numbers are in
`docs/architecture.md` — 25% of pixels removed by cropping, illumination
gradient reduced 88%, green-channel contrast 46.0 → 48.5. Quantify rather than
asserting "it improves the image".

### 6. The evidence heatmap

Open the **Evidence** tab on a case with visible lesions.

*Write about:* warm regions pushed the severity estimate up. Say honestly
whether the heatmap landed on lesions or on the optic disc — an honest negative
observation reads better than an unsupported claim. Mention that this is CAM
rather than Grad-CAM and why (no backward pass in an ONNX graph).

### 7. The evidence button

Capture the call-to-action at the bottom of a completed reading.

*Write about:* why the evidence lives on its own page. A grader reading one
image needs that image's result; a reviewer deciding whether to trust the tool
needs the test-set numbers. Separating them keeps the reading page to one
screen and makes the evidence something a reviewer will finish.

### 7b. The evidence page

Visit `/evidence` and capture it in two or three screenshots: the headline
metric row, the generalisation table with the gap column, and the limitations
panel.

*Write about:* these are read from `artifacts/evaluation.json`, the same file
the figures come from, so the deployed app and the report cannot disagree about
what the model scores. The limitations panel is worth calling out — stating the
quality-gate gap in the product itself, not only in the report, is the kind of
thing that reads as engineering maturity.

### 8. Generated API documentation

Run the FastAPI service (`python app.py`) and visit `/docs`.

*Write about:* the same engine is reachable as a JSON API, not only through the
interface — and the schema is generated from the Pydantic models, so it cannot
drift from the implementation.

### 8b. Both front ends side by side

Screenshot the Streamlit reading page and the FastAPI reading page showing the
same uploaded image and the same stage.

*Write about:* this is the strongest architectural evidence in the project.
Only `backend/main.py` imports a web framework; preprocessing, inference,
ordinal decoding and the triage rules live in `backend/` and are imported by
both. Two interfaces cannot disagree about what the model said, because there
is only one implementation of the deciding. `tests/test_streamlit.py` asserts
no grading logic has leaked into a page file.

*Write about:* the schema is generated from the Pydantic models, so it cannot
drift from the implementation.

### 8c. The plain reading, and the same reading revealed

Two screenshots of the same case: the collapsed view, then with "Show the
numbers behind this reading" open.

*Write about:* who each view is for. The screener needs an action; the numbers
are the evidence for it. Note that nothing was removed — the severity ladder,
the per-stage probabilities and the rule that fired are all one click away, and
a test asserts the reveal still contains them. Hiding evidence and deferring it
are different things.

### 8d. The reading history

Read three images, then screenshot the sidebar.

*Write about:* it lets a grader compare cases without re-uploading, and clicking
an entry re-displays it without re-running inference. Then make the privacy
point: it is held in the browser session, not on disk, because retinal
photographs are biometric identifiers and the model card commits to storing
them nowhere. A design constraint you can name is worth more than a feature you
cannot justify.

### 9. Mobile layout

Narrow the browser to about 390 px, or use device emulation.

*Write about:* the two-column layout stacks and the severity ladder stays
legible. Relevant because screening often happens on tablets.

### 10. An error being handled

Upload a non-image file, or a photograph that is not a fundus.

*Write about:* the message says what went wrong and what to do about it. Showing
that failure paths were designed rather than left to chance is worth marks.

---

## Code screenshots

The brief asks for the codebase as graphic evidence. Screenshot **code that
carries an argument**, not boilerplate.

| File | Lines | Why it is worth showing |
|---|---|---|
| `backend/preprocessing.py` | `preprocess_fundus` | the six-step pipeline in one readable function |
| `backend/inference.py` | `_mc_dropout_heads` | the backbone-once / heads-twenty trick that makes uncertainty affordable |
| `backend/inference.py` | `corn_cumulative_probs` and `cumulative_to_grade_probs` | the ordinal decoding, four lines each |
| `backend/triage.py` | `decide` | the four rules, in order, each returning its own explanation |
| `backend/explain.py` | `severity_weights` | the CAM derivation through the cumulative product |
| `tests/test_corn.py` | `test_cumulative_probabilities_never_increase` | a property test, not an assertion of an expected value |

Use a light theme for print legibility, 13–14 pt, and crop tightly to the
function. Carbon (carbon.now.sh) or VS Code's *Copy with syntax highlighting*
both produce clean output.

### Repository evidence

Screenshot these too — they are what "maintained a GitHub repo" looks like:

- The **Insights → Network** graph, showing branches merging into `main`
- The commit list on `main`, with `feature:` and `update:` prefixes visible
- A merged pull request, ideally one with a description
- The **Actions** tab with green ticks
- The **Releases** page with the weights attached

---

## A note on writing the explanations

The marks are in the explanation, not the image. For each figure, write:

1. **What it shows** — one sentence
2. **What the numbers are** — the specific values, not "good performance"
3. **What it means** — the interpretation
4. **What it does not mean** — the limitation

Example:

> Figure 7 shows per-stage precision, recall and F1 on the 1,879-image DDR test
> set. Moderate NPDR reaches F1 0.856 and proliferative DR 0.904, but mild NPDR
> reaches only 0.380. The gap tracks class support almost exactly: mild NPDR has
> 441 training examples against 4,386 for the healthy class, and is defined by
> microaneurysms roughly 50 µm across — near the resolution limit at 384 px.
> This does not mean the model is unusable at stage 1; it means stage 1 output
> should not be reported as a finding, which is why the triage policy routes
> low-confidence cases to a human rather than publishing the label.

That paragraph demonstrates measurement, diagnosis of a cause, and an honest
boundary. A caption reading "Figure 7: per-class metrics" demonstrates none of
those things.

---

## Word count

Twenty pages is the limit. A rough allocation:

| Section | Pages |
|---|---|
| Introduction and dataset | 2 |
| Preprocessing, with figures | 3 |
| Augmentation and class balance | 2 |
| Architecture and transfer learning | 3 |
| Training | 2 |
| Evaluation | 4 |
| Application and deployment | 3 |
| Limitations and conclusion | 1 |
