/* Interface logic: upload, render the readout, render the performance panel. */

(() => {
  const $ = (id) => document.getElementById(id);

  const STAGE_CAPTIONS = {
    raw: "As uploaded",
    cropped: "Cropped to the retina",
    resized: "Resized to 384 px",
    clahe: "Contrast enhanced",
    illumination: "Illumination corrected",
  };

  const VIEW_CAPTIONS = {
    preprocessed:
      "The 384 × 384 image the network actually receives, after cropping, contrast " +
      "enhancement, illumination correction and the circular mask.",
    original: "Your upload, resized for display only.",
    heatmap:
      "Warm regions pushed the severity estimate up. On a true positive these should " +
      "sit on haemorrhages and exudates, not on the optic disc or the image edge.",
    stages: "Each step of the preprocessing pipeline, in order.",
  };

  const state = {
    result: null,
    view: "preprocessed",
    grades: [],
    thresholds: {},
  };

  /* ─── status ───────────────────────────────────────────────────────── */

  async function loadStatus() {
    const pill = $("status-pill");
    try {
      const health = await API.health();
      if (health.demo_mode) {
        pill.textContent = "Demo mode";
        pill.className = "pill pill-demo";
        $("demo-banner").hidden = false;
      } else if (health.status === "ok") {
        const size = health.model.backbone_size_mb;
        pill.textContent = size ? `Model ready · ${size} MB` : "Model ready";
        pill.className = "pill pill-ok";
      } else {
        pill.textContent = "Model unavailable";
        pill.className = "pill pill-bad";
        showError(health.model.error || "The model could not be loaded.");
      }
      state.thresholds = health.thresholds || {};
    } catch (error) {
      pill.textContent = "Server unreachable";
      pill.className = "pill pill-bad";
    }
  }

  async function loadConfig() {
    try {
      const config = await API.config();
      state.grades = config.grades;
      $("kv-samples").textContent = config.mc_dropout_samples;
    } catch (_) {
      /* the readout falls back to the labels baked into the markup */
    }
  }

  async function loadSamples() {
    try {
      const { samples } = await API.samples();
      if (!samples.length) return;
      const list = $("samples-list");
      samples.forEach((sample) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "sample-btn";
        button.textContent = sample.label;
        button.addEventListener("click", () => runSample(sample.name));
        list.appendChild(button);
      });
      $("samples").hidden = false;
    } catch (_) {
      /* samples are optional */
    }
  }

  /* ─── upload flow ──────────────────────────────────────────────────── */

  function options() {
    return { stages: $("opt-stages").checked, heatmap: $("opt-heatmap").checked };
  }

  function showError(message) {
    const box = $("error");
    box.textContent = message;
    box.hidden = false;
  }

  function clearError() {
    $("error").hidden = true;
  }

  function setBusy(busy, message) {
    $("progress").hidden = !busy;
    if (message) $("progress-text").textContent = message;
    $("dropzone").style.opacity = busy ? 0.55 : 1;
    $("dropzone").style.pointerEvents = busy ? "none" : "auto";
  }

  async function run(promise, label) {
    clearError();
    setBusy(true, `Reading ${label}…`);
    try {
      const result = await promise;
      state.result = result;
      renderResult(result);
      $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      showError(error.message);
      $("result").hidden = true;
    } finally {
      setBusy(false);
    }
  }

  function runFile(file) {
    if (!file.type.startsWith("image/")) {
      showError(`'${file.name}' is not an image. Upload a JPEG, PNG or TIFF photograph.`);
      return;
    }
    run(API.predict(file, options()), file.name);
  }

  function runSample(name) {
    run(API.predictSample(name, options()), name);
  }

  /* ─── the severity ladder ──────────────────────────────────────────── */

  function renderLadder(prediction) {
    const ladder = $("ladder");
    ladder.innerHTML = "";

    const track = document.createElement("div");
    track.className = "ladder-track";
    Grades.COLOURS.forEach((colour, grade) => {
      const step = document.createElement("div");
      step.className = "ladder-step" + (grade === prediction.grade ? " is-active" : "");
      step.textContent = ["No DR", "Mild", "Moderate", "Severe", "PDR"][grade];
      if (grade === prediction.grade) {
        step.style.background = colour;
        step.style.borderBottomColor = colour;
      } else {
        // Unselected steps keep a dim trace of their hue, so the scale still
        // reads as a ramp rather than five grey boxes.
        step.style.borderBottomColor = `color-mix(in srgb, ${colour} 45%, transparent)`;
      }
      track.appendChild(step);
    });
    ladder.appendChild(track);

    // Position along the ladder is the continuous severity score, not the
    // rounded stage, so a case sitting on a boundary looks like one.
    const position = (prediction.expected_grade + 0.5) / 5;
    const spread = prediction.uncertainty / 5;

    if (spread > 0.002) {
      const band = document.createElement("div");
      band.className = "ladder-band";
      const left = Math.max(0, position - spread);
      const right = Math.min(1, position + spread);
      band.style.left = `${left * 100}%`;
      band.style.width = `${(right - left) * 100}%`;
      ladder.appendChild(band);
    }

    const marker = document.createElement("div");
    marker.className = "ladder-marker";
    marker.style.left = `${Math.min(Math.max(position, 0.02), 0.98) * 100}%`;
    marker.innerHTML =
      `<div class="ladder-marker-label">${prediction.expected_grade.toFixed(2)}</div>` +
      `<div class="ladder-marker-line"></div>`;
    ladder.appendChild(marker);

    const legend = document.createElement("p");
    legend.className = "ladder-legend";
    legend.textContent =
      spread > 0.002
        ? `The marker is the continuous severity score. The shaded band is how far ` +
          `${prediction.mc_samples} repeated passes disagreed with each other.`
        : `The marker is the continuous severity score. Repeated passes agreed closely.`;
    ladder.appendChild(legend);
  }

  /* ─── result rendering ─────────────────────────────────────────────── */

  function renderStageBars(prediction) {
    const container = $("stage-bars");
    container.innerHTML = "";
    prediction.grade_probabilities.forEach((entry) => {
      const row = document.createElement("div");
      row.className = "stage-bar" + (entry.grade === prediction.grade ? " is-top" : "");

      const name = document.createElement("span");
      name.className = "stage-bar-name";
      name.textContent = entry.name;

      const track = document.createElement("div");
      track.className = "stage-bar-track";
      const fill = document.createElement("div");
      fill.className = "stage-bar-fill";
      fill.style.width = `${Math.max(entry.probability * 100, 0.6)}%`;
      fill.style.background = Grades.COLOURS[entry.grade];
      fill.style.opacity = entry.grade === prediction.grade ? 1 : 0.42;
      track.appendChild(fill);

      const value = document.createElement("span");
      value.className = "stage-bar-value";
      value.textContent = `${(entry.probability * 100).toFixed(1)}%`;

      row.append(name, track, value);
      container.appendChild(row);
    });
  }

  function renderViewer() {
    const result = state.result;
    if (!result) return;

    const image = $("viewer-image");
    const strip = $("viewer-stages");
    const caption = $("viewer-caption");

    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.toggle("is-active", tab.dataset.view === state.view);
      tab.setAttribute("aria-selected", tab.dataset.view === state.view);
    });

    if (state.view === "stages") {
      image.hidden = true;
      strip.hidden = false;
      strip.innerHTML = "";
      const stages = result.images.stages || {};
      Object.entries(STAGE_CAPTIONS).forEach(([key, label]) => {
        if (!stages[key]) return;
        const figure = document.createElement("figure");
        figure.className = "stage-cell";
        figure.style.margin = "0";
        figure.innerHTML =
          `<img src="${stages[key]}" alt="${label}"><figcaption>${label}</figcaption>`;
        strip.appendChild(figure);
      });
      const final = document.createElement("figure");
      final.className = "stage-cell";
      final.style.margin = "0";
      final.innerHTML =
        `<img src="${result.images.preprocessed}" alt="Final input">` +
        `<figcaption>Masked — model input</figcaption>`;
      strip.appendChild(final);
    } else {
      strip.hidden = true;
      image.hidden = false;
      image.src = result.images[state.view] || result.images.preprocessed;
    }

    caption.textContent = VIEW_CAPTIONS[state.view] || "";
  }

  function renderResult(result) {
    const prediction = result.prediction;

    $("result").hidden = false;
    $("result-filename").textContent = result.filename;

    const total = Object.values(result.timing_ms).reduce((sum, value) => sum + value, 0);
    $("result-timing").textContent = `${Math.round(total)} ms on CPU`;

    $("ladder-grade").textContent = `${prediction.grade} · ${prediction.grade_name}`;
    $("ladder-grade").style.color = Grades.COLOURS[prediction.grade];
    $("ladder-description").textContent = prediction.grade_description;

    $("fig-severity").textContent = prediction.expected_grade.toFixed(2);
    $("fig-confidence").textContent = `${(prediction.confidence * 100).toFixed(0)}%`;
    $("fig-uncertainty").textContent = prediction.uncertainty.toFixed(2);

    renderLadder(prediction);
    renderStageBars(prediction);

    const triage = $("triage-card");
    triage.dataset.urgency = result.triage.urgency;
    $("triage-label").textContent = result.triage.label;
    $("triage-reason").textContent = result.triage.reason;
    $("triage-rule").textContent =
      `Decided by rule ${result.triage.rule} of 4 — quality, then certainty, then severity.`;

    $("kv-anydr").textContent = `${(prediction.probability_any_dr * 100).toFixed(1)}%`;
    $("kv-referable").textContent = `${(prediction.probability_referable * 100).toFixed(1)}%`;
    $("kv-ungradable").textContent = `${(prediction.probability_ungradable * 100).toFixed(1)}%`;
    $("kv-entropy").textContent = prediction.entropy.toFixed(3);

    const heatmapTab = document.querySelector('.tab[data-view="heatmap"]');
    heatmapTab.disabled = !result.images.heatmap;
    const stagesTab = document.querySelector('.tab[data-view="stages"]');
    stagesTab.disabled = !result.images.stages;

    if ((state.view === "heatmap" && !result.images.heatmap) ||
        (state.view === "stages" && !result.images.stages)) {
      state.view = "preprocessed";
    }
    renderViewer();

    $("evidence-cta").hidden = false;

    const warnings = $("warnings");
    warnings.innerHTML = "";
    warnings.hidden = !result.warnings.length;
    result.warnings.forEach((message) => {
      const node = document.createElement("p");
      node.className = "warning";
      node.textContent = message;
      warnings.appendChild(node);
    });
  }

  /* ─── wiring ───────────────────────────────────────────────────────── */

  function wire() {
    const dropzone = $("dropzone");
    const input = $("file-input");

    dropzone.addEventListener("click", () => input.click());
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });
    input.addEventListener("change", () => {
      if (input.files?.[0]) runFile(input.files[0]);
      input.value = "";
    });

    ["dragenter", "dragover"].forEach((name) =>
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-over");
      })
    );
    ["dragleave", "drop"].forEach((name) =>
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-over");
      })
    );
    dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file) runFile(file);
    });

    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        if (tab.disabled) return;
        state.view = tab.dataset.view;
        renderViewer();
      });
    });

    $("btn-reset").addEventListener("click", () => {
      $("result").hidden = true;
      $("evidence-cta").hidden = true;
      state.result = null;
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  wire();
  loadStatus();
  loadConfig();
  loadSamples();
})();
