/* Charts are drawn as inline SVG rather than pulled from a charting library.
   Four small plots do not justify a 200 KB dependency, and hand-drawn SVG
   inherits the page's own colour tokens for free. */

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";

  const GRADE_COLOURS = Grades.COLOURS;

  function el(name, attrs = {}) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) {
      node.setAttribute(key, value);
    }
    return node;
  }

  function svg(width, height) {
    const node = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      class: "chart",
      role: "img",
      preserveAspectRatio: "xMidYMid meet",
    });
    return node;
  }

  const fmt = Grades.fmt;
  const percent = Grades.percent;

  /* ─── confusion matrix ─────────────────────────────────────────────── */

  function confusionMatrix(container, { labels, matrix }) {
    container.innerHTML = "";
    if (!matrix || !matrix.length) {
      container.textContent = "No confusion matrix recorded.";
      return;
    }

    const table = document.createElement("table");
    table.className = "matrix";

    const head = table.createTHead().insertRow();
    head.appendChild(document.createElement("th"));
    labels.forEach((label) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      head.appendChild(cell);
    });

    const body = table.createTBody();
    matrix.forEach((row, rowIndex) => {
      const total = row.reduce((sum, value) => sum + value, 0) || 1;
      const tr = body.insertRow();
      const header = document.createElement("th");
      header.scope = "row";
      header.textContent = labels[rowIndex];
      tr.appendChild(header);

      row.forEach((count, columnIndex) => {
        const share = count / total;
        const cell = tr.insertCell();
        const onDiagonal = rowIndex === columnIndex;
        // Correct predictions take the stage's own colour; errors stay neutral,
        // so the eye lands on the diagonal first and the strays second.
        const hue = onDiagonal ? GRADE_COLOURS[rowIndex] : "#DDE7F1";
        cell.style.background = `color-mix(in srgb, ${hue} ${Math.round(share * 78)}%, transparent)`;
        cell.style.color = share > 0.55 ? "#0E1620" : "#97A9BD";
        cell.innerHTML =
          `<span class="cell-count">${count}</span>` +
          `<span class="cell-share">${(share * 100).toFixed(0)}%</span>`;
      });
    });

    container.appendChild(table);
    const caption = document.createElement("p");
    caption.className = "matrix-caption";
    caption.textContent = "Percentages are of the true stage, so each row sums to 100%.";
    container.appendChild(caption);
  }

  /* ─── per-class table ──────────────────────────────────────────────── */

  function perClassTable(container, rows) {
    container.innerHTML = "";
    const table = document.createElement("table");
    table.className = "table";

    const head = table.createTHead().insertRow();
    ["Stage", "Precision", "Recall", "F1", "Images"].forEach((label) => {
      const cell = document.createElement("th");
      cell.textContent = label;
      head.appendChild(cell);
    });

    const body = table.createTBody();
    rows.forEach((row) => {
      const tr = body.insertRow();
      tr.insertCell().innerHTML =
        `<span class="grade-chip" style="background:${GRADE_COLOURS[row.grade]}"></span>${row.name}`;
      tr.insertCell().textContent = fmt(row.precision);
      tr.insertCell().textContent = fmt(row.recall);
      tr.insertCell().textContent = fmt(row.f1);
      tr.insertCell().textContent = row.support ?? "—";
    });

    container.appendChild(table);
  }

  /* ─── line chart ───────────────────────────────────────────────────── */

  function lineChart(container, { series, xLabel, yLabel, yMax, yMin }) {
    container.innerHTML = "";
    const width = 420;
    const height = 220;
    const pad = { top: 12, right: 12, bottom: 30, left: 38 };

    const allX = series.flatMap((s) => s.points.map((p) => p[0]));
    const allY = series.flatMap((s) => s.points.map((p) => p[1]));
    if (!allX.length) {
      container.textContent = "No data recorded.";
      return;
    }

    const xLo = Math.min(...allX);
    const xHi = Math.max(...allX) || 1;
    const yLo = yMin ?? Math.min(...allY, 0);
    const yHi = yMax ?? Math.max(...allY);

    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;
    const sx = (x) => pad.left + ((x - xLo) / Math.max(xHi - xLo, 1e-9)) * plotWidth;
    const sy = (y) => pad.top + plotHeight - ((y - yLo) / Math.max(yHi - yLo, 1e-9)) * plotHeight;

    const node = svg(width, height);

    for (let step = 0; step <= 4; step += 1) {
      const value = yLo + ((yHi - yLo) * step) / 4;
      const y = sy(value);
      node.appendChild(el("line", { x1: pad.left, y1: y, x2: width - pad.right, y2: y, class: "grid" }));
      const text = el("text", { x: pad.left - 6, y: y + 3, "text-anchor": "end" });
      text.textContent = value.toFixed(2);
      node.appendChild(text);
    }

    node.appendChild(
      el("line", {
        x1: pad.left,
        y1: height - pad.bottom,
        x2: width - pad.right,
        y2: height - pad.bottom,
        class: "axis",
      })
    );

    const ticks = Math.min(6, xHi - xLo + 1);
    for (let index = 0; index < ticks; index += 1) {
      const value = Math.round(xLo + ((xHi - xLo) * index) / Math.max(ticks - 1, 1));
      const text = el("text", {
        x: sx(value),
        y: height - pad.bottom + 14,
        "text-anchor": "middle",
      });
      text.textContent = value;
      node.appendChild(text);
    }

    if (xLabel) {
      const text = el("text", { x: width / 2, y: height - 3, "text-anchor": "middle" });
      text.textContent = xLabel;
      node.appendChild(text);
    }

    series.forEach((s) => {
      const d = s.points
        .map((point, index) => `${index === 0 ? "M" : "L"}${sx(point[0]).toFixed(1)} ${sy(point[1]).toFixed(1)}`)
        .join(" ");
      node.appendChild(
        el("path", { d, fill: "none", stroke: s.colour, "stroke-width": 1.8, "stroke-linejoin": "round" })
      );
      s.points.forEach((point) => {
        node.appendChild(el("circle", { cx: sx(point[0]), cy: sy(point[1]), r: 2, fill: s.colour }));
      });
    });

    container.appendChild(node);

    const legend = document.createElement("div");
    legend.className = "chart-legend";
    series.forEach((s) => {
      const item = document.createElement("span");
      item.style.color = s.colour;
      item.textContent = s.name;
      legend.appendChild(item);
    });
    container.appendChild(legend);
  }

  /* ─── deferral sweep ───────────────────────────────────────────────── */

  function sweepChart(container, rows) {
    if (!rows || !rows.length) {
      container.textContent = "No deferral sweep recorded.";
      return;
    }
    lineChart(container, {
      series: [
        {
          name: "Accuracy on kept cases",
          colour: "#5B8DEF",
          points: rows.map((row) => [row.deferred_percent, row.accuracy]),
        },
        {
          name: "Referable sensitivity",
          colour: "#3FB98A",
          points: rows.map((row) => [row.deferred_percent, row.referable_sensitivity]),
        },
      ],
      xLabel: "% of cases handed to a human grader",
      yMin: 0.8,
      yMax: 1.0,
    });
  }

  return { confusionMatrix, perClassTable, lineChart, sweepChart, GRADE_COLOURS, fmt, percent };
})();
