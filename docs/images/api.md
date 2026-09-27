# API reference

Base URL is wherever the app is running. Interactive documentation is generated
at `/docs`.

Nothing is authenticated and nothing is stored. Uploaded images exist in memory
for the duration of one request and are discarded when it ends.

---

## `GET /api/health`

Whether the model loaded, and which thresholds are active.

```json
{
  "status": "ok",
  "version": "1.2.0",
  "model": {
    "ready": true,
    "explainability": true,
    "backbone": "/app/models/backbone_fp32.onnx",
    "backbone_size_mb": 80.6,
    "feature_size": 1280,
    "dropout_p": 0.4,
    "mc_dropout_samples": 20,
    "error": null
  },
  "demo_mode": false,
  "thresholds": {
    "ungradable": 0.5,
    "uncertainty": 0.45,
    "confidence": 0.55,
    "referral_grade": 2
  }
}
```

| `status` | Meaning |
|---|---|
| `ok` | weights loaded, predictions are real |
| `demo` | no weights; predictions are synthetic and labelled as such |
| `degraded` | no weights and demo mode disabled; `/api/predict` returns 503 |

`explainability` is false when `backbone_cam.onnx` is absent. Grading still
works; the heatmap does not.

Use this as your health check path. It does not touch the model, so it stays
fast under load.

---

## `POST /api/predict`

Grade an uploaded photograph. `multipart/form-data`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | required | JPEG, PNG or TIFF, up to `DR_MAX_UPLOAD_MB` |
| `show_stages` | bool | `true` | include each preprocessing step as an image |
| `show_heatmap` | bool | `true` | include the evidence overlay |

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@fundus.jpg" \
  -F "show_stages=false" \
  -F "show_heatmap=true"
```

### Response

```json
{
  "request_id": "a82e61f26cf8",
  "filename": "fundus.jpg",
  "demo_mode": false,
  "warnings": [],
  "prediction": {
    "grade": 2,
    "grade_name": "Moderate NPDR",
    "grade_description": "More than microaneurysms but less than severe. Referral threshold.",
    "expected_grade": 2.13,
    "grade_probabilities": [
      {"grade": 0, "name": "No DR",    "probability": 0.021},
      {"grade": 1, "name": "Mild",     "probability": 0.094},
      {"grade": 2, "name": "Moderate", "probability": 0.781},
      {"grade": 3, "name": "Severe",   "probability": 0.079},
      {"grade": 4, "name": "PDR",      "probability": 0.025}
    ],
    "confidence": 0.781,
    "uncertainty": 0.094,
    "entropy": 0.712,
    "probability_ungradable": 0.031,
    "probability_any_dr": 0.968,
    "probability_referable": 0.884,
    "mc_samples": 20,
    "inference_ms": 193.4
  },
  "triage": {
    "action": "ophthalmology",
    "label": "Refer to ophthalmology",
    "reason": "Graded Moderate NPDR with 78% confidence. Stage 2 and above is referable disease.",
    "urgency": "soon",
    "rule": 3
  },
  "images": {
    "original": "data:image/jpeg;base64,...",
    "preprocessed": "data:image/jpeg;base64,...",
    "heatmap": "data:image/jpeg;base64,...",
    "stages": null
  },
  "timing_ms": {"preprocess_ms": 44.6, "inference_ms": 193.4, "explain_ms": 24.1}
}
```

### Field meanings

| Field | What it is |
|---|---|
| `grade` | the reported stage, 0–4 |
| `expected_grade` | continuous severity, the sum of the four cumulative probabilities. Use this rather than `grade` when you need a position on the scale — 1.9 and 2.4 both report as stage 2 but mean different things |
| `grade_probabilities` | per-stage probabilities, derived from the cumulative ones. Always sums to 1 |
| `confidence` | the probability of the winning stage |
| `uncertainty` | the standard deviation of `expected_grade` across the 20 MC passes, on the 0–4 scale. Higher means the passes disagreed |
| `entropy` | Shannon entropy of the stage distribution. Complements `uncertainty`: entropy is high when the model spreads mass evenly, `uncertainty` is high when repeated passes disagree |
| `probability_ungradable` | the quality head's output. **See the model card — this is uncalibrated**, because the training copy of DDR contained no ungradable images |
| `probability_any_dr` | P(stage ≥ 1) directly from the ordinal head, not derived from `grade` |
| `probability_referable` | P(stage ≥ 2), likewise |
| `triage.rule` | which of the four rules decided the action, for auditing |
| `triage.urgency` | `routine`, `soon`, `urgent` or `blocked` — drives the colour in the interface |
| `warnings` | non-fatal problems: demo mode, an upload that does not look like a fundus, a missing CAM model |

Images are base64 JPEG data URLs so a full result arrives in one response.
That makes the body large — roughly 150 KB with the heatmap, 400 KB with all
the pipeline stages. Set `show_stages=false` and `show_heatmap=false` if you are
calling this programmatically and only want the numbers.

### Errors

| Status | When |
|---|---|
| 400 | the file is empty, not decodable as an image, or under 96 px on a side |
| 413 | the file exceeds `DR_MAX_UPLOAD_MB` |
| 503 | no weights and demo mode disabled |

```json
{"detail": "'notes.pdf' could not be read as an image. Upload a JPEG, PNG or TIFF fundus photograph."}
```

---

## `POST /api/predict/sample`

Same response, but grades one of the files in `samples/`. Takes `name` instead
of `file`. Paths are resolved inside `samples/` only, so `../` cannot escape it.

---

## `GET /api/config`

Stage labels, descriptions, triage action names and the active thresholds. The
interface calls this at start-up so the labels live in one place.

---

## `GET /api/metrics`

Everything in `artifacts/`: headline metrics for both test sets, per-stage
precision/recall/F1, the confusion matrix, 20 epochs of training history, the
deferral sweep, the internal-versus-external comparison, the training run
details and the CPU benchmarks.

This backs the `/evidence` page. Keys: `headline`, `per_class`, `confusion`,
`history`, `referral_sweep`, `comparison`, `run`, `deployment`.

These are recorded results from the training run, not computed live. If
`artifacts/evaluation.json` is missing you get `{"available": false}` and the
performance panel stays hidden.

---

## `GET /api/samples`

Lists the bundled examples, with the display label derived from each filename.

---

## Notes for programmatic use

**Do not call this concurrently on a small instance.** One ONNX session, one
worker, and each request holds about 190 ms of CPU. Requests queue rather than
parallelise.

**Batch by making sequential calls.** There is no batch endpoint. The backbone
accepts a batch dimension, but exposing it would let one request monopolise the
worker for minutes.

**`expected_grade` is the more useful number.** For sorting a worklist by
severity or setting your own referral threshold, use it rather than `grade` —
it preserves the information that rounding to a stage throws away.
