"""
rpps.report - Self-contained HTML report generation.

Loads the processed-output files written by `rpps.metrics.compute_all` (and,
when present, the Phase 3 break / regression / counterfactual outputs), builds
the corresponding figures via `rpps.visualization`, and emits a single
self-contained HTML file with figures embedded inline as base64 PNGs.

The output is deliberately a single static HTML file so it can be emailed,
opened from any browser without a local server, and printed to PDF.

Usage
-----
    from rpps.report import build_report
    out = build_report(
        processed_dir="data/processed",
        output_path="report.html",
    )
    print(f"Wrote: {out}")

Or from the command line:

    python -m rpps.report --processed-dir data/processed --output report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rpps import __version__ as RPPS_VERSION
from rpps import visualization as viz

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaded-data container
# ---------------------------------------------------------------------------

@dataclass
class ReportInputs:
    """All artifacts the report needs, loaded from data/processed."""

    # Phase 1 splice
    spliced_wages: pd.Series | None = None
    spliced_productivity: pd.Series | None = None
    splice_audit: dict | None = None

    # Phase 2 metrics
    rpph_composite: pd.Series | None = None
    rpph_by_item: pd.DataFrame | None = None
    rpph_audit: dict | None = None

    wicr_panel: pd.DataFrame | None = None
    wicr_audit: dict | None = None

    prwdi_panel: pd.DataFrame | None = None
    prwdi_audit: dict | None = None

    # Phase 3 (optional)
    breaks_regimes: pd.Series | None = None
    breaks_summary: pd.DataFrame | None = None
    breaks_audit: dict | None = None

    regression_audit: dict | None = None
    regression_cross_regime: pd.DataFrame | None = None
    regression_by_regime_coefs: pd.DataFrame | None = None

    counterfactual_panel: pd.DataFrame | None = None
    counterfactual_audit: dict | None = None

    # Run-level metadata
    metrics_run_summary: dict | None = None

    available_phases: list[str] = field(default_factory=list)


def _read_csv_series(path: Path) -> pd.Series | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.shape[1] == 0:
            return None
        return df.iloc[:, 0]
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _read_csv_df(path: Path, parse_dates: bool = True) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=parse_dates)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def load_inputs(processed_dir: str | Path) -> ReportInputs:
    """Load every available processed artifact from `processed_dir`."""
    p = Path(processed_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"processed_dir does not exist: {p.resolve()}")

    inp = ReportInputs()
    available: list[str] = []

    # Phase 1: splice
    inp.spliced_wages = _read_csv_series(p / "spliced_wages.csv")
    inp.spliced_productivity = _read_csv_series(p / "spliced_productivity.csv")
    inp.splice_audit = _read_json(p / "splice_audit.json")
    if inp.spliced_wages is not None or inp.spliced_productivity is not None:
        available.append("phase1")

    # Phase 2: metrics
    inp.rpph_composite = _read_csv_series(p / "rpph_composite.csv")
    inp.rpph_by_item = _read_csv_df(p / "rpph_by_item.csv")
    inp.rpph_audit = _read_json(p / "rpph_audit.json")

    inp.wicr_panel = _read_csv_df(p / "wicr_panel.csv")
    inp.wicr_audit = _read_json(p / "wicr_audit.json")

    inp.prwdi_panel = _read_csv_df(p / "prwdi_panel.csv")
    inp.prwdi_audit = _read_json(p / "prwdi_audit.json")

    if any(x is not None for x in
           (inp.rpph_composite, inp.wicr_panel, inp.prwdi_panel)):
        available.append("phase2")

    # Phase 3 (optional)
    inp.breaks_regimes = _read_csv_series(p / "breaks_regimes.csv")
    inp.breaks_summary = _read_csv_df(p / "breaks_summary.csv", parse_dates=False)
    inp.breaks_audit = _read_json(p / "breaks_audit.json")

    inp.regression_audit = _read_json(p / "regression_audit.json")
    inp.regression_cross_regime = _read_csv_df(p / "regression_cross_regime.csv",
                                                parse_dates=False)
    inp.regression_by_regime_coefs = _read_csv_df(
        p / "regression_by_regime_coefs.csv", parse_dates=False)

    inp.counterfactual_panel = _read_csv_df(p / "counterfactual_panel.csv")
    inp.counterfactual_audit = _read_json(p / "counterfactual_audit.json")

    if any(x is not None for x in
           (inp.breaks_audit, inp.regression_audit,
            inp.counterfactual_audit)):
        available.append("phase3")

    inp.metrics_run_summary = _read_json(p / "metrics_run_summary.json")

    inp.available_phases = available
    return inp


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; }
html, body {
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: #1f2933;
    background: #fafbfc;
    line-height: 1.55;
}
.container {
    max-width: 920px;
    margin: 0 auto;
    padding: 48px 32px 96px;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
}
header {
    border-bottom: 2px solid #1f4e79;
    padding-bottom: 24px;
    margin-bottom: 36px;
}
h1 { color: #1f4e79; font-weight: 700; font-size: 28px; margin: 0 0 4px; }
.subtitle { color: #52606d; font-size: 14px; }
.meta-row {
    display: flex; flex-wrap: wrap; gap: 16px 32px;
    color: #7f8a96; font-size: 12px; margin-top: 12px;
}
h2 {
    color: #1f4e79; font-weight: 600; font-size: 20px;
    margin: 36px 0 12px; padding-bottom: 6px;
    border-bottom: 1px solid #dfe5eb;
}
h3 { color: #1f2933; font-weight: 600; font-size: 16px; margin: 18px 0 8px; }
p { margin: 0 0 12px; }
.lead { font-size: 15px; color: #3e4c59; }
.metric-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 16px; margin: 16px 0 24px;
}
.metric {
    background: #f5f7fa; border-left: 3px solid #1f4e79;
    padding: 14px 18px; border-radius: 0 4px 4px 0;
}
.metric .label { color: #7f8a96; font-size: 11px; letter-spacing: 0.06em;
                 text-transform: uppercase; }
.metric .value { color: #1f2933; font-size: 22px; font-weight: 600;
                 margin-top: 4px; }
.metric .sub { color: #7f8a96; font-size: 11px; margin-top: 4px; }
figure { margin: 18px 0 24px; }
figure img { display: block; max-width: 100%; height: auto;
             border: 1px solid #eaedf2; border-radius: 3px; }
figcaption { color: #7f8a96; font-size: 12px; margin-top: 6px;
             font-style: italic; }
table {
    border-collapse: collapse; width: 100%; margin: 16px 0 24px;
    font-size: 13px;
}
th {
    text-align: left; padding: 10px 12px;
    background: #f5f7fa; color: #3e4c59; font-weight: 600;
    border-bottom: 2px solid #c2cad3;
}
td { padding: 8px 12px; border-bottom: 1px solid #eef1f5; }
tr:nth-child(even) td { background: #fafbfc; }
.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.warning {
    background: #fff8ed; border-left: 3px solid #e0a458;
    padding: 12px 16px; border-radius: 0 3px 3px 0;
    margin: 14px 0; font-size: 13px;
}
footer {
    margin-top: 56px; padding-top: 16px; border-top: 1px solid #dfe5eb;
    color: #7f8a96; font-size: 12px;
}
code { background: #f5f7fa; padding: 1px 5px; border-radius: 2px;
       font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
       font-size: 0.92em; }
.kv { display: grid; grid-template-columns: max-content 1fr;
      gap: 4px 16px; font-size: 12px; }
.kv dt { color: #7f8a96; font-variant-numeric: tabular-nums; }
.kv dd { margin: 0; color: #3e4c59; }
@media print {
    body { background: white; }
    .container { box-shadow: none; max-width: none; padding: 0; }
}
"""


def _img_tag(fig, alt: str) -> str:
    b64 = viz.figure_to_base64(fig)
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" />'


def _df_to_table(df: pd.DataFrame, max_rows: int = 20,
                 numeric_format: str = "{:,.4g}") -> str:
    """Render a DataFrame to an HTML table with light styling."""
    if df is None or df.empty:
        return '<p><em>(no data)</em></p>'
    if len(df) > max_rows:
        head_n = max_rows // 2
        tail_n = max_rows - head_n
        shown = pd.concat([df.head(head_n), df.tail(tail_n)])
        elided = True
    else:
        shown = df
        elided = False

    cols = list(shown.columns)
    html = ['<table>', '<thead><tr><th>&nbsp;</th>']
    for c in cols:
        html.append(f'<th>{c}</th>')
    html.append('</tr></thead><tbody>')
    for idx, row in shown.iterrows():
        idx_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
        html.append(f'<tr><td><strong>{idx_str}</strong></td>')
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cell = numeric_format.format(val) if pd.notna(val) else "&mdash;"
                html.append(f'<td class="numeric">{cell}</td>')
            else:
                html.append(f'<td>{val if pd.notna(val) else "&mdash;"}</td>')
        html.append('</tr>')
    html.append('</tbody></table>')
    if elided:
        html.append(f'<p style="color:#7f8a96;font-size:11px;">'
                    f'(Showing {max_rows} of {len(df)} rows; full data in CSV.)</p>')
    return "\n".join(html)


def _kv_block(d: dict) -> str:
    """Render a flat dict as a key-value definition list."""
    items = []
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, default=str)
        items.append(f'<dt>{k}</dt><dd>{v}</dd>')
    return f'<dl class="kv">{"".join(items)}</dl>'


def _executive_metrics(inp: ReportInputs) -> list[tuple[str, str, str]]:
    """Return a list of (label, value, sub) tuples for the headline grid."""
    rows: list[tuple[str, str, str]] = []

    if inp.rpph_composite is not None and not inp.rpph_composite.dropna().empty:
        valid = inp.rpph_composite.dropna()
        rows.append((
            "RPPH composite (latest)",
            f"{valid.iloc[-1]:,.1f} hrs",
            f"as of {valid.index[-1].date()}",
        ))
        rows.append((
            "RPPH composite (start)",
            f"{valid.iloc[0]:,.1f} hrs",
            f"as of {valid.index[0].date()}",
        ))

    if inp.wicr_audit is not None:
        n_high = inp.wicr_audit.get("n_high_wicr_periods", "n/a")
        rows.append((
            "Sustained-high WICR periods",
            str(n_high),
            "smoothed WICR > 0.80 for >=8 periods",
        ))

    if inp.prwdi_audit is not None:
        end_val = inp.prwdi_audit.get("prwdi_at_end")
        base = inp.prwdi_audit.get("base_year", "?")
        if end_val is not None:
            rows.append((
                "PRWDI (latest)",
                f"{end_val:.2f}x",
                f"productivity vs compensation, base {base}",
            ))

    if inp.counterfactual_audit is not None:
        gap = inp.counterfactual_audit.get("final_pct_gap")
        ci_lo = inp.counterfactual_audit.get("final_pct_gap_ci_low")
        ci_hi = inp.counterfactual_audit.get("final_pct_gap_ci_high")
        if gap is not None:
            sub = "1948-71 distribution counterfactual"
            if ci_lo is not None and ci_hi is not None:
                sub = f"95% CI [{ci_lo:.1%}, {ci_hi:.1%}]"
            rows.append(("Counterfactual gap", f"{gap:+.1%}", sub))

    if inp.spliced_wages is not None:
        n = len(inp.spliced_wages.dropna())
        first = inp.spliced_wages.dropna().index.min()
        last = inp.spliced_wages.dropna().index.max()
        rows.append((
            "Spliced wage series",
            f"{n:,} obs",
            f"{first.year}-{last.year}",
        ))

    return rows


def render_html(inp: ReportInputs) -> str:
    """Build the full HTML report from loaded inputs."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = []

    parts.append(f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<title>Real Purchasing Power Simulator - Analysis Report</title>
<style>{CSS}</style>
</head>
<body>
<div class='container'>
<header>
  <h1>Real Purchasing Power Simulator</h1>
  <div class='subtitle'>MTS Pillar 6 - Welfare Measurement Architecture, 1920-present</div>
  <div class='meta-row'>
    <span>Generated <strong>{now}</strong></span>
    <span>rpps version <strong>{RPPS_VERSION}</strong></span>
    <span>Phases included: <strong>{', '.join(inp.available_phases) or '(none detected)'}</strong></span>
  </div>
</header>
""")

    # Executive summary
    parts.append("<h2>Executive summary</h2>")
    parts.append(
        "<p class='lead'>This report summarizes the spliced data, "
        "derived welfare metrics, and (when available) regime-structure "
        "and counterfactual analyses produced by the Real Purchasing Power "
        "Simulator. All figures and tables are derived from the artifacts "
        "in <code>data/processed</code> at the time of report generation.</p>")

    metrics = _executive_metrics(inp)
    if metrics:
        parts.append("<div class='metric-grid'>")
        for label, value, sub in metrics:
            parts.append(
                f"<div class='metric'>"
                f"<div class='label'>{label}</div>"
                f"<div class='value'>{value}</div>"
                f"<div class='sub'>{sub}</div>"
                f"</div>"
            )
        parts.append("</div>")

    # Phase 1: splice
    if "phase1" in inp.available_phases:
        parts.append("<h2>1. Splice (1920-present)</h2>")
        parts.append(
            "<p>The wage and productivity series are reconstructed from the "
            "BLS / FRED post-1939 (post-1947 for productivity) leg and the "
            "NBER Macrohistory / Kendrick (1961) pre-1939 (pre-1947) leg, "
            "joined via a multiplicative geometric-mean adjustment computed "
            "over the overlap window.</p>")

        if inp.spliced_wages is not None:
            parts.append("<h3>Wage splice (manufacturing production workers)</h3>")
            fig = viz.figure_spliced_wages(inp.spliced_wages)
            parts.append(f"<figure>{_img_tag(fig, 'Spliced wage series')}"
                         f"<figcaption>Source: NBER M08142USM055NNBR + BLS AHEMAN, "
                         f"spliced 1939-01-01.</figcaption></figure>")
            if inp.splice_audit:
                wage_audit = inp.splice_audit.get("splices", {}).get("wages", {})
                if wage_audit:
                    parts.append("<h3>Wage-splice audit</h3>")
                    parts.append(_kv_block(wage_audit))

        if inp.spliced_productivity is not None:
            parts.append("<h3>Productivity splice</h3>")
            fig = viz.figure_spliced_productivity(inp.spliced_productivity)
            parts.append(f"<figure>{_img_tag(fig, 'Spliced productivity')}"
                         f"<figcaption>Source: Kendrick (1961) + BLS OPHNFB, "
                         f"spliced 1947-01-01.</figcaption></figure>")
            if inp.splice_audit:
                prod_audit = inp.splice_audit.get("splices", {}).get("productivity", {})
                if prod_audit:
                    parts.append("<h3>Productivity-splice audit</h3>")
                    parts.append(_kv_block(prod_audit))

    # Phase 2: derived metrics
    if "phase2" in inp.available_phases:
        parts.append("<h2>2. Derived welfare metrics</h2>")

        if inp.rpph_composite is not None:
            parts.append("<h3>2.1 Real Purchasing Power Hours (RPPH)</h3>")
            parts.append(
                "<p>Hours of labor required at the prevailing manufacturing "
                "production-worker wage to purchase the fixed reference "
                "basket. Falling RPPH = rising real purchasing power; rising "
                "RPPH = falling real purchasing power. The metric is "
                "denominated in hours, not dollars, deliberately.</p>")
            fig = viz.figure_rpph_composite(inp.rpph_composite)
            parts.append(f"<figure>{_img_tag(fig, 'RPPH composite')}"
                         f"<figcaption>Composite basket: total hours of labor.</figcaption></figure>")

            if inp.rpph_by_item is not None and not inp.rpph_by_item.empty:
                fig = viz.figure_rpph_by_item(inp.rpph_by_item)
                parts.append(f"<figure>{_img_tag(fig, 'RPPH by item')}"
                             f"<figcaption>Per-item RPPH on log scale; one line per basket item.</figcaption></figure>")

        if inp.wicr_panel is not None:
            parts.append("<h3>2.2 Wage-Inflation Capture Ratio (WICR)</h3>")
            parts.append(
                "<p>WICR = year-over-year inflation / year-over-year wage "
                "growth. WICR=0 means full nominal-wage gains convert to real "
                "gains; WICR=1 means real wages flat; WICR>1 means real wages "
                "declining. The 0.50 and 0.80 thresholds (low/medium/high) "
                "are pre-registered priors per the Stage 1 working paper "
                "(MTS Pillar 6, &sect;6.1). Shaded regions mark sustained "
                "high-WICR runs (smoothed WICR > 0.80 for &ge;8 periods).</p>")
            fig = viz.figure_wicr(inp.wicr_panel)
            parts.append(f"<figure>{_img_tag(fig, 'WICR')}"
                         f"<figcaption>Smoothed WICR with regime thresholds and sustained-high shading.</figcaption></figure>")

        if inp.prwdi_panel is not None:
            parts.append("<h3>2.3 Productivity-Real-Wage Decoupling Index (PRWDI)</h3>")
            base = (inp.prwdi_audit or {}).get("base_year", 1947)
            parts.append(
                f"<p>PRWDI<sub>t</sub> = (Q<sub>t</sub>/Q<sub>{base}</sub>) "
                f"/ (C<sub>t</sub>/C<sub>{base}</sub>). PRWDI=1 at the base "
                f"year by construction. Values above 1 indicate productivity "
                f"outpacing real compensation (cumulative decoupling); values "
                f"below 1 indicate the reverse.</p>")
            fig = viz.figure_prwdi(inp.prwdi_panel, base_year=base)
            parts.append(f"<figure>{_img_tag(fig, 'PRWDI')}"
                         f"<figcaption>Top: productivity and compensation indices "
                         f"(both = 1.0 at base year). Bottom: PRWDI ratio.</figcaption></figure>")

    # Phase 3 (optional)
    if "phase3" in inp.available_phases:
        parts.append("<h2>3. Regime structure and counterfactual</h2>")

        if inp.breaks_audit is not None and inp.breaks_summary is not None:
            parts.append("<h3>3.1 Detected structural breaks</h3>")
            parts.append(
                f"<p>Method: {inp.breaks_audit.get('method', 'unspecified')}. "
                f"Detected {inp.breaks_audit.get('n_breaks', 0)} break(s), "
                f"yielding {inp.breaks_audit.get('n_regimes', 0)} regime(s).</p>")
            parts.append(_df_to_table(inp.breaks_summary))

        if inp.regression_cross_regime is not None:
            parts.append("<h3>3.2 Cross-regime coefficient tests</h3>")
            parts.append(_df_to_table(inp.regression_cross_regime))

        if inp.counterfactual_panel is not None:
            parts.append("<h3>3.3 Counterfactual real-compensation trajectory</h3>")
            cf_audit = inp.counterfactual_audit or {}
            ref = cf_audit.get("reference_window_years", "?")
            parts.append(
                f"<p>Reference window: {ref}. The counterfactual applies the "
                f"reference-period productivity-distribution coefficients to "
                f"the realized post-reference productivity path.</p>")
            fig = viz.figure_counterfactual(inp.counterfactual_panel)
            parts.append(f"<figure>{_img_tag(fig, 'Counterfactual gap')}"
                         f"<figcaption>Actual real compensation versus the "
                         f"reference-window-distribution counterfactual.</figcaption></figure>")

    # Diagnostics / audit
    parts.append("<h2>Audit and diagnostics</h2>")

    if inp.metrics_run_summary:
        parts.append("<h3>Metrics run summary</h3>")
        parts.append(_kv_block({
            k: v for k, v in inp.metrics_run_summary.items()
            if k not in ("results",)
        }))

    if inp.splice_audit:
        boundary_warnings = []
        for splice_name, splice_data in inp.splice_audit.get("splices", {}).items():
            if not splice_data.get("boundary_continuity_ok", True):
                boundary_warnings.append((splice_name, splice_data))
        if boundary_warnings:
            parts.append('<div class="warning"><strong>Boundary continuity flag:</strong> '
                         'the following splice(s) reported a residual level discontinuity '
                         'at the boundary even after multiplicative adjustment. This is '
                         'expected when the legacy and modern series have different '
                         'methodological underpinnings; the geometric-mean adjustment '
                         'minimizes total log-distance over the overlap but does not '
                         'force exact boundary equality.<ul>')
            for name, data in boundary_warnings:
                parts.append(
                    f'<li><strong>{name}</strong>: adjustment factor '
                    f'{data.get("adjustment_factor", "n/a"):.4f}, '
                    f'overlap {data.get("overlap_start", "?")} to '
                    f'{data.get("overlap_end", "?")} '
                    f'({data.get("n_overlap_obs", "?")} obs)</li>'
                )
            parts.append('</ul></div>')

    parts.append("""<footer>
  <p>Real Purchasing Power Simulator. MTS doctrine, Pillar 6.</p>
  <p>Companion code to: Green, R. J. (2026). The Inflationary Yardstick: A
     Mutual Threshold Saturation Critique of Nominal Wage Growth as a
     Welfare Metric, 1925-2025.</p>
  <p>This report was generated from the artifacts in <code>data/processed/</code>.
     Re-running <code>make data &amp;&amp; make metrics &amp;&amp; make report</code>
     reproduces it from source.</p>
</footer>
</div>
</body>
</html>""")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_report(
    processed_dir: str | Path = "data/processed",
    output_path: str | Path = "report.html",
) -> dict[str, Any]:
    """Load processed artifacts, render the HTML report, write it to disk.

    Returns
    -------
    dict with keys:
        output_path : str
        n_phases    : int
        size_bytes  : int
    """
    inp = load_inputs(processed_dir)
    if not inp.available_phases:
        raise FileNotFoundError(
            f"No processed artifacts found under {Path(processed_dir).resolve()}. "
            "Run `make data` then `make metrics` first."
        )
    html = render_html(inp)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size
    logger.info("Wrote report to %s (%d bytes)", out.resolve(), size)
    return {
        "output_path": str(out.resolve()),
        "n_phases": len(inp.available_phases),
        "available_phases": inp.available_phases,
        "size_bytes": size,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rpps.report",
        description="Generate a self-contained HTML analysis report from "
                    "processed simulator outputs.",
    )
    parser.add_argument(
        "--processed-dir", default="data/processed",
        help="Directory containing processed CSV/JSON artifacts (default: data/processed).",
    )
    parser.add_argument(
        "--output", default="report.html",
        help="Output HTML file path (default: report.html).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        result = build_report(processed_dir=args.processed_dir,
                              output_path=args.output)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {result['output_path']} "
          f"({result['size_bytes']:,} bytes; "
          f"phases: {', '.join(result['available_phases'])})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
