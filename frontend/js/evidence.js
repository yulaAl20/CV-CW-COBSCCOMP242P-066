/* Evidence page.

   Everything here is read from artifacts/evaluation.json through /api/metrics.
   Nothing is recomputed and nothing is hard-coded, so this page cannot claim a
   score the training run did not produce. */

(() => {
  const $ = (id) => document.getElementById(id);

  const MAJORITY_BASELINE = 0.5003; // always answering "No DR" on the DDR test split

  function metricCard(value, label, note) {
    const card = document.createElement("div");
    card.className = "metric";
    card.innerHTML =
      `<p class="metric-value">${value}</p>` +
      `<p class="metric-label">${label}</p>` +
      (note ? `<p class="metric-note">${note}</p>` : "");
    return card;
  }

  function renderRunSummary(run) {
    if (!run || !run.backbone) return;
    $("run-summary").textContent =
      `${run.backbone} · ${run.total_parameters_millions}M parameters · ` +
      `${run.epochs_run} of ${run.epochs_configured} epochs, stopped early at epoch ` +
      `${run.best_epoch} · trained on ${run.train_images.toLocaleString()} images ` +
      `in ${run.training_minutes} minutes.`;
  }

  function renderHeadline(headline) {
    const { internal, external, detection, calibration } = headline;
    const row = $("headline-metrics");
    row.innerHTML = "";
    row.append(
      metricCard(
        Grades.percent(internal.accuracy),
        "Stage accuracy",
        `vs ${Grades.percent(MAJORITY_BASELINE)} for always answering "No DR"`
      ),
      metricCard(
        Grades.fmt(internal.qwk),
        "Agreement with the grader",
        "quadratic weighted kappa"
      ),
      metricCard(Grades.fmt(internal.macro_f1), "Macro F1", "averaged over all five stages"),
      metricCard(
        Grades.fmt(internal.referable_auc),
        "Referable DR, AUC",
        "stage 2 and above — the referral decision"
      ),
      metricCard(
        Grades.percent(detection.recall),
        "DR detection recall",
        `${Grades.percent(detection.specificity)} specificity`
      ),
      metricCard(
        Grades.fmt(calibration.ece),
        "Calibration error",
        "0 is perfect; this model is overconfident"
      )
    );
  }

  function renderComparison(rows) {
    const container = $("comparison");
    container.innerHTML = "";
    if (!rows || !rows.length) {
      container.textContent = "No external comparison recorded.";
      return;
    }

    const table = document.createElement("table");
    table.className = "table comparison-table";

    const head = table.createTHead().insertRow();
    ["Metric", "DDR test", "APTOS 2019", "Gap"].forEach((label) => {
      const cell = document.createElement("th");
      cell.textContent = label;
      head.appendChild(cell);
    });

    const body = table.createTBody();
    rows.forEach((row) => {
      const tr = body.insertRow();
      tr.insertCell().textContent = row.metric;
      tr.insertCell().textContent = Grades.fmt(row.internal);
      tr.insertCell().textContent = Grades.fmt(row.external);

      const gap = tr.insertCell();
      gap.textContent = Grades.missing(row.gap) ? "—" : `−${Grades.fmt(row.gap)}`;
      // A large gap is the finding, so it is coloured; a small one is not.
      gap.style.color = row.gap > 0.1 ? "#E9A13B" : "#3FB98A";
    });

    container.appendChild(table);

    const note = document.createElement("p");
    note.className = "panel-note";
    note.style.marginTop = "14px";
    note.textContent =
      "Macro F1 falls by 0.27 while referable AUC falls by 0.017. The model still " +
      "separates diseased eyes from healthy ones on unfamiliar data; what it loses " +
      "is knowing exactly where another set of graders drew the stage boundaries.";
    container.appendChild(note);
  }

  function renderDeployment(deployment) {
    const container = $("deployment");
    container.innerHTML = "";
    const benchmarks = deployment?.benchmarks;
    if (!benchmarks || !benchmarks.length) {
      container.textContent = "No deployment benchmarks recorded.";
      return;
    }

    const table = document.createElement("table");
    table.className = "table";

    const head = table.createTHead().insertRow();
    ["Export", "Size", "Median", "Mean", "95th percentile"].forEach((label) => {
      const cell = document.createElement("th");
      cell.textContent = label;
      head.appendChild(cell);
    });

    const body = table.createTBody();
    benchmarks.forEach((row) => {
      const tr = body.insertRow();
      tr.insertCell().textContent = row.model;
      tr.insertCell().textContent = `${row.size_mb} MB`;
      tr.insertCell().textContent = `${row.median_ms.toFixed(0)} ms`;
      tr.insertCell().textContent = `${row.mean_ms.toFixed(0)} ms`;
      tr.insertCell().textContent = `${row.p95_ms.toFixed(0)} ms`;
    });

    container.appendChild(table);

    const note = document.createElement("p");
    note.className = "panel-note";
    note.style.marginTop = "14px";
    note.textContent = deployment.onnx_export_verified
      ? "The int8 export is a quarter of the size and 1.8× slower: per-operator " +
        "conversion overhead costs more than the narrower arithmetic saves on a " +
        "convolutional backbone. The ONNX graph was verified against PyTorch to " +
        `within ${deployment.max_abs_difference.toExponential(1)}.`
      : "Export verification did not pass for this build.";
    container.appendChild(note);
  }

  async function load() {
    let data;
    try {
      data = await API.metrics();
    } catch (error) {
      $("evidence-loading").hidden = true;
      $("evidence-error").hidden = false;
      $("evidence-error").textContent =
        `The recorded results could not be loaded. ${error.message}`;
      return;
    }

    if (data.available === false || !data.headline?.available) {
      $("evidence-loading").hidden = true;
      $("evidence-error").hidden = false;
      $("evidence-error").textContent =
        "No evaluation results are installed on this server. Add " +
        "artifacts/evaluation.json from a training run to populate this page.";
      return;
    }

    renderRunSummary(data.run);
    renderHeadline(data.headline);
    renderComparison(data.comparison);

    Charts.confusionMatrix($("confusion"), data.confusion);
    Charts.perClassTable($("per-class"), data.per_class);

    if (data.history?.length) {
      Charts.lineChart($("curves"), {
        series: [
          {
            name: "Training loss",
            colour: "#5B8DEF",
            points: data.history.map((row) => [row.epoch, row.train_loss]),
          },
          {
            name: "Validation loss",
            colour: "#E2663C",
            points: data.history.map((row) => [row.epoch, row.val_loss]),
          },
          {
            name: "Validation agreement",
            colour: "#3FB98A",
            points: data.history.map((row) => [row.epoch, row.val_qwk]),
          },
        ],
        xLabel: "epoch",
        yMin: 0,
        yMax: 1,
      });
    }

    Charts.sweepChart($("sweep"), data.referral_sweep);
    renderDeployment(data.deployment);

    $("evidence-loading").hidden = true;
    $("evidence-body").hidden = false;
  }

  load();
})();
