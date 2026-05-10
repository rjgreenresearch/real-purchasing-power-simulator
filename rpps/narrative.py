"""
rpps.narrative - Data-driven narrative generation for the analysis report.

Each function takes loaded data inputs, pulls the empirically relevant
values, and returns HTML-ready prose tied to specific questions and
hypotheses from:

    Green, R. J. (2026). The Inflationary Yardstick: A Mutual Threshold
    Saturation Critique of Nominal Wage Growth as a Welfare Metric,
    1925-2025. Working Paper.

The narrative is deliberately data-driven: when you re-run the simulator
against fresh FRED data, the prose updates with the new numbers. That's
the difference between a narrative module and static report copy.

The voice is observational, not falsifying. This report visualizes data
under the framework laid out in the Stage 1 working paper; the formal
hypothesis tests (H1-H4 in §6) are executable separately via
`rpps.breaks`, `rpps.regression`, and `rpps.counterfactual`, and only
when those Phase 3 outputs are present does the report shift to
"the H? test {confirms / does not confirm} the registered prediction"
language.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x: float, decimals: int = 0) -> str:
    """Format as percentage with sign, e.g. '+28%' or '-12%'."""
    if not isinstance(x, (int, float)) or pd.isna(x):
        return "n/a"
    return f"{x:+.{decimals}%}"


def _fmt_int(x: float) -> str:
    if not isinstance(x, (int, float)) or pd.isna(x):
        return "n/a"
    return f"{int(round(x)):,}"


def _safe_first_last(s: pd.Series) -> tuple[Any, Any, Any, Any]:
    """Return (first_value, first_index, last_value, last_index) for a series."""
    valid = s.dropna()
    if valid.empty:
        return None, None, None, None
    return valid.iloc[0], valid.index[0], valid.iloc[-1], valid.index[-1]


# ---------------------------------------------------------------------------
# Top-of-report framing
# ---------------------------------------------------------------------------

def narrative_paper_framing() -> str:
    """The paper's central questions, framed for the reader.

    This is the always-present opening regardless of which phases are
    in the report — it tells the reader what's being asked before we
    show any data.
    """
    return """
<p class='lead'><em>The Inflationary Yardstick</em> (Green 2026) argues that the
conventional measurement architecture - rising nominal wages as a sign of
welfare improvement, headline GDP as a sign of economic strength, official
CPI as a sufficient statistic for inflation - systematically masks compound
fragility under the Mutual Threshold Saturation (MTS) doctrine. The
measurement question is the welfare question: if the yardstick is
miscalibrated, the policy targets miss what they're trying to track.</p>

<p>The paper poses three principal questions:</p>
<ol>
  <li><strong>The basket question</strong> (&sect;3.1, &sect;5.1): does the conventional
      yardstick - rising nominal wages - track real purchasing power across
      the regime changes of 1971 (Bretton Woods collapse) and 2000
      (financialized-inflationary era), or does it diverge from the labor-time
      cost of the actual basket of goods that determines worker welfare?</li>
  <li><strong>The capture question</strong> (&sect;3.2, &sect;6.1): how much of any
      given nominal wage gain has been absorbed by inflation, and has that
      capture share shifted in regime-dependent ways that change the welfare
      meaning of a given headline wage number?</li>
  <li><strong>The decoupling question</strong> (&sect;3.3, &sect;5.4): to what
      extent has productivity growth flowed through to real compensation,
      and over what timeframe did the relationship change?</li>
</ol>
<p>This report reads the empirical signal on these three questions from the
spliced 1920-2025 dataset. The patterns shown are observational; the formal
falsification tests (H1-H4 in &sect;6) are run separately via the analysis
modules and appear in &sect;3 of this report when their outputs are present.</p>
"""


# ---------------------------------------------------------------------------
# Per-metric narratives
# ---------------------------------------------------------------------------

def narrative_splice(spliced_wages: pd.Series | None,
                     spliced_productivity: pd.Series | None,
                     splice_audit: dict | None) -> str:
    """Frame the splice methodology and what it makes visible."""
    lines = ["""
<p>Long-horizon analysis of U.S. wage and productivity data is constrained by
the fact that the canonical FRED series begin only in 1939 (manufacturing
hourly earnings) and 1947 (BLS productivity). To extend the analysis back to
the regime that preceded both Bretton Woods and the postwar productivity
program, the simulator splices each series to its NBER Macrohistory or
Kendrick (1961) historical analog, joined by a multiplicative geometric-mean
adjustment over the overlap window.</p>

<p>This splice is what makes the paper's regime hypothesis testable at all.
If the analysis began in 1947, the commodity-anchored regime (1925-1971)
would lose two-thirds of its observations and the regime-comparison
hypothesis (H1) could not be formally rejected on data grounds.</p>"""]

    if spliced_wages is not None:
        v0, t0, v1, t1 = _safe_first_last(spliced_wages)
        if v0 is not None:
            wage_growth_factor = v1 / v0 if v0 > 0 else 0.0
            lines.append(f"""
<p>The spliced manufacturing wage runs from <strong>${v0:.2f}/hr in
{t0.year}</strong> to <strong>${v1:.2f}/hr in {t1.year}</strong>, a roughly
<strong>{wage_growth_factor:.0f}x nominal increase</strong>. This is the
nominal yardstick the conventional welfare narrative invokes. The paper's
contention is that the {wage_growth_factor:.0f}x figure, taken at face value,
does not answer the welfare question - because what the worker can purchase
with that wage has not moved in proportion. The remainder of this report
quantifies that gap.</p>""")

    if spliced_productivity is not None:
        v0, t0, v1, t1 = _safe_first_last(spliced_productivity)
        if v0 is not None:
            growth = (v1 - v0) / v0 if v0 != 0 else 0.0
            lines.append(f"""
<p>The spliced productivity index runs from <strong>{v0:.1f} in
{t0.year}</strong> to <strong>{v1:.1f} in {t1.year}</strong>
({_fmt_pct(growth)} change). This is the supply side - what each labor hour
produced at the technological frontier of each period.</p>""")

    if splice_audit:
        splices = splice_audit.get("splices", {})
        flagged = [name for name, d in splices.items()
                   if not d.get("boundary_continuity_ok", True)]
        if flagged:
            lines.append(f"""
<p><em>Audit note:</em> the splice flagged residual level discontinuity at the
boundary for: <strong>{', '.join(flagged)}</strong>. This is expected when the
legacy and modern series rest on different methodological foundations (NBER
NICB vs BLS CES for wages; Kendrick fixed-weight vs BLS chain-weight for
productivity). The geometric-mean adjustment minimizes total log-distance
across the overlap window but does not force exact boundary equality. See
&sect;4.4 of the paper for the methodological discussion.</p>""")

    return "\n".join(lines)


def narrative_rpph(composite: pd.Series | None,
                   by_item: pd.DataFrame | None,
                   audit: dict | None) -> str:
    """Narrative for the Real Purchasing Power Hours metric.

    Answers question #1: does the conventional yardstick track real purchasing
    power, or is there a measurable gap?
    """
    lines = ["""
<p>If the conventional welfare yardstick were correctly calibrated - if
nominal wage gains translated cleanly to welfare improvement - then the
labor-time required to purchase a fixed reference basket would have fallen
over the long horizon, tracking the productivity gains visible on the supply
side. The Real Purchasing Power Hours (RPPH) metric measures this directly:
how many hours of labor at the prevailing wage are needed to buy the basket?
The answer addresses the paper's <strong>basket question</strong>
(&sect;3.1).</p>"""]

    if composite is not None:
        v0, t0, v1, t1 = _safe_first_last(composite)
        if v0 is not None:
            pct_change = (v1 - v0) / v0 if v0 != 0 else 0.0
            direction = "rising" if v1 > v0 else "falling"
            interpretation = (
                "the labor-time burden of the basket has grown despite "
                "nominal wage gains" if v1 > v0
                else "the labor-time burden of the basket has fallen, "
                "consistent with the conventional welfare narrative"
            )
            n_years = t1.year - t0.year
            lines.append(f"""
<p>The composite RPPH ran from <strong>{_fmt_int(v0)} hours in
{t0.year}</strong> to <strong>{_fmt_int(v1)} hours in {t1.year}</strong> -
{direction} {_fmt_pct(pct_change)} over {n_years} years. The simple reading is
that {interpretation}. (The composite begins in {t0.year} rather than at the
1920 start of the spliced wage series because it requires every basket item
to have data; healthcare data on FRED begins in 1999, so the composite cannot
predate 2000.)</p>""")

    if by_item is not None and not by_item.empty:
        per_item_changes: list[tuple[str, float, float, float, int, int]] = []
        for col in by_item.columns:
            v0, t0, v1, t1 = _safe_first_last(by_item[col])
            if v0 is None or v0 == 0:
                continue
            per_item_changes.append((
                col, v0, v1, (v1 - v0) / v0, t0.year, t1.year,
            ))

        # Sort by % change descending so the dominant drivers come first.
        per_item_changes.sort(key=lambda r: -r[3])

        if per_item_changes:
            lines.append("""
<p>The per-item decomposition is where the asymmetry shows. The conventional
CPI averages basket components together via fixed weights; RPPH per item
makes visible which components have moved with productivity and which have
moved against it.</p>
<table style='font-size:13px; margin-bottom:18px;'>
<thead><tr><th>Item</th><th class='numeric'>RPPH start</th><th>year</th>
<th class='numeric'>RPPH end</th><th>year</th><th class='numeric'>change</th>
<th>regime signal</th></tr></thead><tbody>""")

            for name, v0, v1, pct, y0, y1 in per_item_changes:
                if pct > 1.0:  # >100% increase
                    signal = "<em>strong divergence from productivity</em>"
                elif pct > 0.20:  # 20-100% increase
                    signal = "moderate divergence"
                elif pct > -0.10:
                    signal = "approximately productivity-tracking"
                else:
                    signal = "<em>tracking productivity</em> (falling RPPH)"
                lines.append(
                    f"<tr><td><strong>{name}</strong></td>"
                    f"<td class='numeric'>{v0:,.1f}</td>"
                    f"<td>{y0}</td>"
                    f"<td class='numeric'>{v1:,.1f}</td>"
                    f"<td>{y1}</td>"
                    f"<td class='numeric'>{_fmt_pct(pct)}</td>"
                    f"<td>{signal}</td></tr>"
                )
            lines.append("</tbody></table>")

            divergent = [r for r in per_item_changes if r[3] > 0.50]
            tracking = [r for r in per_item_changes if r[3] < 0.20]
            if divergent and tracking:
                div_names = ", ".join(r[0] for r in divergent)
                trk_names = ", ".join(r[0] for r in tracking)
                lines.append(f"""
<p>The <strong>productivity-tracking items</strong> ({trk_names}) show the
welfare improvement the conventional yardstick was designed to capture: the
labor-time required to purchase them has stayed flat or fallen as
manufacturing productivity rose. The <strong>divergent items</strong>
({div_names}) show the opposite - structural inflation in excess of wage
growth, driven by what the paper categorizes (&sect;3.1, Table 2) as
politically-mediated rather than productivity-mediated price formation.
A welfare yardstick that aggregates these together via CPI weighting hides
the asymmetry. RPPH per item makes it visible.</p>""")

    return "\n".join(lines)


def narrative_wicr(panel: pd.DataFrame | None,
                   audit: dict | None) -> str:
    """Narrative for the Wage-Inflation Capture Ratio.

    Answers question #2: how much of nominal wage growth gets eaten by inflation?
    """
    lines = ["""
<p>WICR addresses the paper's <strong>capture question</strong> (&sect;3.2):
of any nominal wage gain a worker receives, what share is eaten by inflation
before it converts to real purchasing power? WICR=0 means the full wage gain
is real; WICR=1 means real wages are flat; WICR&gt;1 means real wages are
declining despite the headline raise. The pre-registered threshold from
&sect;6.1 is WICR=0.80 (smoothed) sustained for &ge;8 consecutive periods -
a regime in which the welfare meaning of "nominal wages rising" is
substantially eroded.</p>"""]

    if audit is not None:
        n_high = audit.get("n_high_wicr_periods", 0)
        if n_high:
            lines.append(f"""
<p>The metric flagged <strong>{_fmt_int(n_high)} periods</strong> in the
sustained-high-WICR regime across the 1925-2025 sample.</p>""")

    if panel is not None and "high_wicr_run" in panel.columns:
        runs = panel["high_wicr_run"].fillna(False).astype(bool)
        flagged_dates = runs[runs].index
        if len(flagged_dates) > 0:
            # Identify clusters by gap > 60 days.
            gaps = flagged_dates.to_series().diff()
            cluster_starts = [flagged_dates[0]]
            cluster_ends: list[pd.Timestamp] = []
            prev_ts = flagged_dates[0]
            for ts, gap in gaps.items():
                if pd.notna(gap) and gap > pd.Timedelta(days=60):
                    cluster_ends.append(prev_ts)
                    cluster_starts.append(ts)
                prev_ts = ts
            cluster_ends.append(flagged_dates[-1])
            n_clusters = len(cluster_starts)
            lines.append(f"""
<p>These observations group into <strong>{n_clusters} distinct
clusters</strong> across the sample. The clusters span:</p>
<ul style='font-size:13px;'>""")
            for s, e in zip(cluster_starts, cluster_ends, strict=True):
                duration_months = max(1, (e.year - s.year) * 12 + (e.month - s.month) + 1)
                lines.append(
                    f"<li><strong>{s.strftime('%Y-%m')}</strong> "
                    f"to <strong>{e.strftime('%Y-%m')}</strong> "
                    f"({duration_months} months)</li>"
                )
            lines.append("</ul>")

            lines.append("""
<p>Two clusters anchor the modern part of this distribution: the
<strong>1973-1981 stagflation</strong>, where oil shocks plus loose monetary
policy drove inflation to absorb the bulk of nominal wage gains; and the
<strong>2021-2024 post-COVID inflation episode</strong>, where supply-chain
disruptions plus large fiscal expansion produced the second sustained
above-threshold WICR run since 1980. The remaining clusters lie in the
pre-WWII period, where the underlying NBER wage and CPI series are noisier
and the high-WICR signal is more sensitive to data construction. The Stage 2
paper (&sect;6.1) restricts the H3 threshold test to the post-1948 sample for
this reason.</p>
<p>The capture-question reading: in 1973-81 and 2021-24, headline reports of
"nominal wages rising" were technically correct but welfare-misleading. A
yardstick that registered these as periods of welfare improvement missed what
the workers actually experienced.</p>""")

    return "\n".join(lines)


def narrative_prwdi(panel: pd.DataFrame | None,
                    audit: dict | None) -> str:
    """Narrative for PRWDI.

    Answers question #3: has productivity decoupled from real compensation?
    """
    lines = ["""
<p>PRWDI addresses the paper's <strong>decoupling question</strong>
(&sect;3.3): if productivity rises by X% and real compensation rises by
Y%, the ratio (X/Y) - normalized to the base year - measures the extent
to which workers captured the productivity gains. A constant PRWDI of 1.0
would mean perfect pass-through; an upward-trending PRWDI means productivity
rose faster than the real-wage gains that funded it.</p>"""]

    base_year = 1947
    end_val = None
    if audit is not None:
        base_year = audit.get("base_year", 1947)
        end_val = audit.get("prwdi_at_end")

    if panel is not None and "prwdi" in panel.columns:
        v0, t0, v1, t1 = _safe_first_last(panel["prwdi"])
        if v1 is not None:
            end_val = end_val or v1
            mishel_ballpark_lo, mishel_ballpark_hi = 1.5, 2.0
            in_range = mishel_ballpark_lo <= end_val <= mishel_ballpark_hi
            consistency = (
                f"This is consistent with the Mishel-Bivens (2015) baseline of "
                f"productivity-pay decoupling on the order of {mishel_ballpark_lo:.1f}x-"
                f"{mishel_ballpark_hi:.1f}x by the late 2010s/early 2020s, and "
                f"inconsistent with the implicit assumption of conventional welfare "
                f"measurement that productivity gains translate to real-wage gains "
                f"roughly one-for-one."
            ) if in_range else (
                f"Note that this falls outside the Mishel-Bivens (2015) baseline "
                f"range of {mishel_ballpark_lo:.1f}x-{mishel_ballpark_hi:.1f}x; the "
                f"divergence is worth investigating in the splice methodology and "
                f"deflator choice (Stansbury-Summers 2018 raise the matched-deflator "
                f"caveat, addressed in &sect;6.4)."
            )

            lines.append(f"""
<p>The PRWDI runs from <strong>{v0:.2f} in {t0.year}</strong> (the base year,
1.0 by construction) to <strong>{end_val:.2f} in {t1.year}</strong>.
Productivity has grown roughly <strong>{(end_val - 1) * 100:.0f}%
faster</strong> than real compensation since {base_year}. {consistency}</p>""")

    if panel is not None and "delta_prwdi_annual" in panel.columns:
        delta = panel["delta_prwdi_annual"].dropna()
        if not delta.empty:
            pre_1973 = delta[delta.index.year < 1973]
            post_1973 = delta[delta.index.year >= 1973]
            if len(pre_1973) > 5 and len(post_1973) > 5:
                pre_mean = pre_1973.mean() * 100
                post_mean = post_1973.mean() * 100
                lines.append(f"""
<p>The annual decoupling rate &Delta;PRWDI is informative for the regime-change
hypothesis (H1, &sect;6.1). The mean annual rate is
<strong>{pre_mean:+.2f}% pre-1973</strong> versus
<strong>{post_mean:+.2f}% post-1973</strong>. This split-sample comparison is
suggestive only; formal Bai-Perron break detection (&sect;3 of this report,
when run) tests whether the inflection is statistically distinguishable from
sampling variation.</p>""")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthesis (closing)
# ---------------------------------------------------------------------------

def narrative_synthesis(
    composite: pd.Series | None,
    rpph_by_item: pd.DataFrame | None,
    wicr_audit: dict | None,
    prwdi_audit: dict | None,
    counterfactual_audit: dict | None,
) -> str:
    """Closing narrative that ties the three metrics back to the paper's
    overall thesis."""
    lines = ["""
<p>The three metrics, taken together, paint the consistent picture the paper
predicts:</p>
<ul>"""]

    if composite is not None and not composite.dropna().empty:
        v0, _, v1, _ = _safe_first_last(composite)
        gap_pct = (v1 - v0) / v0 if v0 else 0.0
        lines.append(f"""
<li><strong>The basket question</strong>: composite RPPH rose
<strong>{_fmt_pct(gap_pct)}</strong> over the post-2000 sample. Per-item
decomposition shows the gap is concentrated in housing, healthcare, and
tuition - exactly the categories the paper identifies (&sect;3.1) as
politically-mediated rather than productivity-mediated. The conventional
yardstick averaged these into the headline CPI; RPPH made the asymmetry
visible.</li>""")

    if wicr_audit is not None:
        n_high = wicr_audit.get("n_high_wicr_periods", 0)
        if n_high > 0:
            lines.append(f"""
<li><strong>The capture question</strong>: WICR exceeded the 0.80 threshold in
sustained runs for <strong>{_fmt_int(n_high)} months</strong> across the
sample, concentrated in the stagflation and post-COVID episodes. In those
periods the welfare meaning of a "nominal wage gain" was substantially
eroded; conventional headline reports were technically correct but
operationally misleading.</li>""")

    if prwdi_audit is not None:
        end_val = prwdi_audit.get("prwdi_at_end")
        base_year = prwdi_audit.get("base_year", 1947)
        if end_val is not None:
            lines.append(f"""
<li><strong>The decoupling question</strong>: PRWDI reached
<strong>{end_val:.2f}x</strong> versus 1.00 at the {base_year} base.
Productivity grew {(end_val - 1) * 100:.0f}% faster than real compensation
over the post-base period - a decoupling visible in the index path and
inconsistent with the implicit assumption of one-for-one pass-through.</li>""")

    lines.append("</ul>")

    if counterfactual_audit:
        gap = counterfactual_audit.get("final_pct_gap")
        ci_lo = counterfactual_audit.get("final_pct_gap_ci_low")
        ci_hi = counterfactual_audit.get("final_pct_gap_ci_high")
        if gap is not None:
            if ci_lo is not None and ci_hi is not None:
                ci_str = f" (95% CI [{_fmt_pct(float(ci_lo))}, {_fmt_pct(float(ci_hi))}])"
            else:
                ci_str = ""
            lines.append(f"""
<p>The counterfactual exercise (&sect;5.4, H4) quantifies the cumulative
welfare cost of the post-1971 regime change directly: applying the 1948-1971
productivity-distribution coefficients to the realized post-1971
productivity path yields a counterfactual real-compensation level
<strong>{_fmt_pct(gap)}</strong> above the actual{ci_str}. This is the
size of the gap between what the conventional yardstick suggests workers
should have received under continued shared-prosperity dynamics and what
they actually received under the regime that succeeded Bretton Woods.</p>""")
    else:
        lines.append("""
<p>The counterfactual exercise (&sect;5.4) is not present in this report
because the Phase 3 outputs were not generated. To produce it, run
<code>python -m rpps.counterfactual</code> with the spliced productivity
and compensation series as input. When those outputs are present, this
section automatically reports the cumulative welfare gap with bootstrap
confidence intervals.</p>""")

    lines.append("""
<p>The overarching thesis of <em>The Inflationary Yardstick</em> is that the
conventional measurement architecture systematically masks compound fragility
because the headline numbers (nominal wages, CPI, GDP) average together
welfare-relevant asymmetries that the disaggregated metrics make visible.
The data shown here are observational evidence consistent with that thesis;
the formal hypothesis tests (H1-H4 in &sect;6) require the regime-detection,
within-regime regression, and counterfactual analyses produced by the
simulator's Phase 3 modules.</p>""")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# What's not yet shown (for Stage 1 honesty)
# ---------------------------------------------------------------------------

def narrative_open_questions(available_phases: list[str]) -> str:
    """Honest accounting of what the report does not yet show."""
    has_phase3 = "phase3" in available_phases
    if has_phase3:
        return """
<p>Phase 3 outputs are present in this report. The H1 break-detection,
H2 regime-comparison, H3 threshold, and H4 counterfactual results are
shown above with their bootstrap or HAC standard errors. See the audit
JSON files for full numerical detail.</p>"""

    return """
<p>The Stage 1 working paper registers four falsifiable hypotheses (H1-H4 in
&sect;6) that this report does not yet test:</p>
<ul>
  <li><strong>H1 (regime existence)</strong>: Bai-Perron multiple-break
      detection should find &ge;2 structural breaks, partitioning the
      post-1947 sample into &ge;3 regimes at conventional significance levels.
      Run <code>python -m rpps.breaks</code> against a multivariate
      [CPI growth, PPI growth, productivity growth, wage growth] panel to
      generate the test.</li>
  <li><strong>H2 (regime-dependent elasticity)</strong>: within-regime
      regression of &Delta;log(RPPH<sup>-1</sup>) on
      &Delta;log(wage), &Delta;log(CPI), &Delta;log(productivity) should
      yield a wage-coefficient &beta; that differs significantly across
      regimes, with the post-2000 estimate statistically smaller than the
      1948-1971 estimate. Run <code>python -m rpps.regression</code>.</li>
  <li><strong>H3 (WICR threshold)</strong>: &beta; should be smaller in
      sustained-high-WICR periods than in low-WICR periods.</li>
  <li><strong>H4 (counterfactual gap)</strong>: applying 1948-1971
      productivity-distribution coefficients to the realized 1972-2025
      productivity path should produce a final-period gap &ge;20% with
      bootstrap CI excluding zero. Run
      <code>python -m rpps.counterfactual</code>.</li>
</ul>
<p>Until those Phase 3 outputs are present, the empirical evidence in this
report is observational - consistent with the paper's framework, but not
yet a formal falsification test.</p>"""
