

from __future__ import annotations

import io
from datetime import datetime

from fpdf import FPDF

from .config import GRADE_DESCRIPTIONS, Settings

# Same severity colours as the interface (streamlit_ui.GRADE_COLOURS and
# frontend/js/grades.js), so a stage is the same colour on screen and on paper.
GRADE_COLOURS = ["#3FB98A", "#BFC94E", "#E9A13B", "#E2663C", "#CF3D57"]
GRADE_SHORT = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
URGENCY_COLOURS = {
    "routine": "#3FB98A",
    "soon": "#E9A13B",
    "urgent": "#CF3D57",
    "blocked": "#8298AE",
}

INK = "#16212E"
MUTED = "#5B6B7C"
RULE = "#D5DDE5"
PANEL = "#F3F6F9"

# The built-in PDF fonts only cover Latin-1. Rather than ship a font file, the
# handful of typographic characters the interface uses are mapped to plain ones.
_PLAIN = str.maketrans({
    "—": "-", "–": "-", "…": "...", "’": "'", "‘": "'", "“": '"', "”": '"',
    "≥": ">=", "≤": "<=", "→": "->", "　": " ",
})


def _text(value: object) -> str:
    return str(value).translate(_PLAIN).encode("latin-1", "replace").decode("latin-1")


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    hex_colour = hex_colour.lstrip("#")
    return tuple(int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))


class _ReportPDF(FPDF):
    footer_note = ""

    def footer(self) -> None:
        self.set_y(-14)
        self.set_draw_color(*_rgb(RULE))
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*_rgb(MUTED))
        self.cell(0, 4, _text(self.footer_note), align="L")
        self.cell(0, 4, f"Page {self.page_no()}", align="R")


def _heading(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*_rgb(INK))
    pdf.cell(0, 6, _text(text), new_x="LMARGIN", new_y="NEXT")


def _result_box(pdf: FPDF, *, colour: str, action_text: str, finding_text: str,
                scale_text: str) -> None:
    """The answer, in a tinted box with the urgency colour down its left edge."""
    left, width = pdf.l_margin, pdf.w - pdf.l_margin - pdf.r_margin
    top = pdf.get_y()

    # measure first, then draw the box behind the text
    pdf.set_font("Helvetica", "B", 15)
    action_lines = pdf.multi_cell(width - 12, 7, _text(action_text), dry_run=True,
                                  output="LINES")
    pdf.set_font("Helvetica", "", 10.5)
    finding_lines = pdf.multi_cell(width - 12, 5.4, _text(finding_text), dry_run=True,
                                   output="LINES")
    pdf.set_font("Helvetica", "", 8.5)
    scale_lines = (
        pdf.multi_cell(width - 12, 4.3, _text(scale_text), dry_run=True, output="LINES")
        if scale_text else []
    )
    height = 8 + 5 + len(action_lines) * 7 + 2 + len(finding_lines) * 5.4 + \
        (1.5 + len(scale_lines) * 4.3 if scale_lines else 0) + 5

    pdf.set_fill_color(*_rgb(PANEL))
    pdf.rect(left, top, width, height, style="F")
    pdf.set_fill_color(*_rgb(colour))
    pdf.rect(left, top, 2.2, height, style="F")

    pdf.set_xy(left + 7, top + 6)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*_rgb(MUTED))
    pdf.cell(0, 4, "NEXT STEP", new_x="LEFT", new_y="NEXT")
    pdf.ln(1)
    pdf.set_x(left + 7)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*_rgb(colour))
    pdf.multi_cell(width - 12, 7, _text(action_text), new_x="LEFT", new_y="NEXT")
    pdf.ln(2)
    pdf.set_x(left + 7)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*_rgb(INK))
    pdf.multi_cell(width - 12, 5.4, _text(finding_text), new_x="LEFT", new_y="NEXT")
    if scale_text:
        pdf.ln(1.5)
        pdf.set_x(left + 7)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*_rgb(MUTED))
        pdf.multi_cell(width - 12, 4.3, _text(scale_text), new_x="LEFT", new_y="NEXT")

    pdf.set_y(top + height + 5)


def _banner(pdf: FPDF, text: str, colour: str) -> None:
    width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_fill_color(*_rgb(colour))
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(width, 6, _text(text), fill=True, padding=(1.5, 3),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)


def _stage_bars(pdf: FPDF, x: float, y: float, width: float, prediction) -> float:
    """Five labelled bars. Returns the y below them."""
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*_rgb(INK))
    pdf.cell(width, 6, "How the five stages scored")
    y += 9

    label_width, value_width = 30, 13
    bar_width = width - label_width - value_width - 2
    for grade, probability in enumerate(prediction.grade_probabilities):
        winner = grade == prediction.grade
        pdf.set_xy(x, y)
        pdf.set_font("Helvetica", "B" if winner else "", 8.5)
        pdf.set_text_color(*_rgb(INK if winner else MUTED))
        pdf.cell(label_width, 5, _text(f"{grade}  {GRADE_SHORT[grade]}"))

        pdf.set_fill_color(*_rgb(RULE))
        pdf.rect(x + label_width, y + 1.3, bar_width, 2.6, style="F")
        filled = max(bar_width * float(probability), 0.4)
        pdf.set_fill_color(*_rgb(GRADE_COLOURS[grade]))
        pdf.rect(x + label_width, y + 1.3, filled, 2.6, style="F")

        pdf.set_xy(x + label_width + bar_width + 2, y)
        pdf.cell(value_width, 5, f"{float(probability) * 100:.1f}%", align="R")
        y += 6.5
    return y


def _figures(pdf: FPDF, x: float, y: float, width: float, rows: list[tuple[str, str]]) -> float:
    """Label / value pairs. Returns the y below them."""
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*_rgb(INK))
    pdf.cell(width, 6, "Key figures")
    y += 9
    for label, value in rows:
        pdf.set_xy(x, y)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*_rgb(MUTED))
        pdf.cell(width - 22, 5, _text(label))
        pdf.set_font("Courier", "B", 9)
        pdf.set_text_color(*_rgb(INK))
        pdf.cell(22, 5, _text(value), align="R")
        pdf.set_draw_color(*_rgb(RULE))
        pdf.line(x, y + 5.6, x + width, y + 5.6)
        y += 6.5
    return y


def build_report(
    *,
    filename: str,
    read_at: datetime,
    report_id: str,
    prediction,
    decision,
    action_text: str,
    finding_text: str,
    original_jpeg: bytes,
    second_jpeg: bytes | None,
    second_caption: str,
    warnings: list[str],
    demo: bool,
    settings: Settings,
    model_note: str = "",
) -> bytes:
    """Render one reading as a single A4 page. Returns the PDF bytes."""
    pdf = _ReportPDF(format="A4")
    pdf.set_margins(16, 16, 16)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.footer_note = (
        "Not a medical device. Student prototype for coursework; not clinically "
        "validated. Do not use for decisions about a real patient."
    )
    pdf.set_title(_text(f"RetinaTriage report - {filename}"))
    pdf.set_creator("RetinaTriage")
    pdf.add_page()

    content_width = pdf.w - pdf.l_margin - pdf.r_margin

    # --- header ----------------------------------------------------------
    pdf.set_font("Helvetica", "B", 19)
    pdf.set_text_color(*_rgb(INK))
    pdf.cell(content_width * 0.6, 9, "RetinaTriage")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*_rgb(MUTED))
    pdf.cell(content_width * 0.4, 9, _text(read_at.strftime("Read %d %b %Y, %H:%M")),
             align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(content_width * 0.6, 5, "Diabetic retinopathy screening report")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.cell(content_width * 0.4, 5, _text(f"Report {report_id}"), align="R",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.cell(0, 5, _text(f"Image: {filename}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_draw_color(*_rgb(RULE))
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    if demo:
        _banner(pdf, "SYNTHETIC DEMO RESULT - no trained model was loaded, so this is "
                     "not a real prediction.", "#CF3D57")

    # --- the answer ------------------------------------------------------
    scale_text = (
        ""
        if decision.action == "recapture"
        else f"Stage {prediction.grade} of 4 on the international diabetic retinopathy "
             f"scale - {GRADE_DESCRIPTIONS[prediction.grade]}"
    )
    _result_box(
        pdf,
        colour=URGENCY_COLOURS.get(decision.urgency, MUTED),
        action_text=action_text,
        finding_text=finding_text,
        scale_text=scale_text,
    )

    for message in warnings:
        _banner(pdf, message, "#B7791F")

    # --- images ----------------------------------------------------------
    gap = 6
    image_size = (content_width - gap) / 2
    top = pdf.get_y()
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*_rgb(INK))
    pdf.set_xy(pdf.l_margin, top)
    pdf.cell(image_size, 5, "The photograph")
    if second_jpeg:
        pdf.set_xy(pdf.l_margin + image_size + gap, top)
        pdf.cell(image_size, 5, _text(second_caption))
    top += 6
    pdf.image(io.BytesIO(original_jpeg), x=pdf.l_margin, y=top, w=image_size, h=image_size)
    if second_jpeg:
        pdf.image(io.BytesIO(second_jpeg), x=pdf.l_margin + image_size + gap, y=top,
                  w=image_size, h=image_size)
    pdf.set_y(top + image_size + 7)

    # --- numbers ---------------------------------------------------------
    column = (content_width - 10) / 2
    top = pdf.get_y()
    left_end = _stage_bars(pdf, pdf.l_margin, top, column, prediction)
    right_end = _figures(pdf, pdf.l_margin + column + 10, top, column, [
        ("Any diabetic retinopathy", f"{prediction.probability_any_dr * 100:.1f}%"),
        ("Referable disease, stage 2+", f"{prediction.probability_referable * 100:.1f}%"),
        ("Too poor to grade", f"{prediction.probability_ungradable * 100:.1f}%"),
        ("Continuous severity, 0-4", f"{prediction.expected_grade:.2f}"),
        (f"Spread across {prediction.mc_samples} passes", f"{prediction.uncertainty:.3f}"),
    ])
    pdf.set_y(max(left_end, right_end) + 5)

    # --- why -------------------------------------------------------------
    _heading(pdf, "Why this decision")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*_rgb(INK))
    pdf.multi_cell(0, 5, _text(decision.reason), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*_rgb(MUTED))
    pdf.multi_cell(
        0, 4.2,
        _text(
            f"Rule {decision.rule} of 4 - image quality first, then certainty, then "
            f"severity; the first rule that fires decides. Thresholds: re-capture above "
            f"{settings.ungradable_threshold:.2f} ungradable, human review above "
            f"{settings.uncertainty_threshold:.2f} spread or below "
            f"{settings.confidence_threshold:.2f} confidence, referral at stage "
            f"{settings.referral_grade}."
            + (f" {model_note}" if model_note else "")
        ),
        new_x="LMARGIN", new_y="NEXT",
    )

    return bytes(pdf.output())


def report_filename(filename: str, read_at: datetime) -> str:
    """`retinatriage_<image name>_<date>.pdf`, safe for any file system."""
    stem = filename.rsplit(".", 1)[0]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:40] or "image"
    return f"retinatriage_{safe}_{read_at:%Y%m%d_%H%M}.pdf"
