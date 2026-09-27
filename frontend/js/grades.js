
const Grades = (() => {
  const COLOURS = ["#3FB98A", "#BFC94E", "#E9A13B", "#E2663C", "#CF3D57"];
  const SHORT = ["No DR", "Mild", "Moderate", "Severe", "PDR"];
  const FULL = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
  ];

  const missing = (value) =>
    value === null || value === undefined || Number.isNaN(Number(value));

  /* Fixed decimal places, because these are measurements sitting in columns
     and a ragged right edge makes them harder to compare. */
  const fmt = (value, digits = 3) =>
    missing(value) ? "—" : Number(value).toFixed(digits);

  const percent = (value, digits = 1) =>
    missing(value) ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;

  return { COLOURS, SHORT, FULL, fmt, percent, missing };
})();
