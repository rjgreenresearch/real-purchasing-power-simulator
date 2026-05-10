"""Tests for rpps.narrative.

The narrative functions are data-driven: each takes loaded inputs and
returns HTML-ready prose. The tests verify that:
- Real numbers from the inputs land in the prose (not just static text)
- The prose responds correctly to different data scenarios
  (rising vs falling RPPH, with/without Phase 3, etc.)
- The output is HTML-safe and stable
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rpps import narrative

# ---------------------------------------------------------------------------
# narrative_paper_framing
# ---------------------------------------------------------------------------

class TestPaperFraming:
    def test_returns_html_with_three_questions(self):
        html = narrative.narrative_paper_framing()
        assert "basket question" in html
        assert "capture question" in html
        assert "decoupling question" in html

    def test_references_paper_sections(self):
        html = narrative.narrative_paper_framing()
        assert "&sect;3.1" in html
        assert "&sect;6" in html

    def test_references_paper_title(self):
        html = narrative.narrative_paper_framing()
        # Either italics or quoted form
        assert "Inflationary Yardstick" in html


# ---------------------------------------------------------------------------
# narrative_splice
# ---------------------------------------------------------------------------

class TestSpliceNarrative:
    def test_quotes_real_wage_endpoints(self):
        idx = pd.date_range("1920-01-01", "2024-12-01", freq="MS")
        wages = pd.Series(np.linspace(0.40, 30.0, len(idx)), index=idx)
        html = narrative.narrative_splice(wages, None, None)
        # Specific values from the actual data should appear
        assert "$0.40/hr" in html
        assert "$30.00/hr" in html
        assert "1920" in html
        assert "2024" in html

    def test_quotes_real_productivity_endpoints(self):
        idx = pd.DatetimeIndex([f"{y}-12-31" for y in range(1925, 2025)])
        prod = pd.Series(np.linspace(12.0, 119.0, len(idx)), index=idx)
        html = narrative.narrative_splice(None, prod, None)
        assert "12.0" in html
        assert "119.0" in html

    def test_flags_boundary_discontinuity(self):
        audit = {
            "splices": {
                "wages": {"boundary_continuity_ok": False, "adjustment_factor": 0.73},
                "productivity": {"boundary_continuity_ok": False},
            }
        }
        html = narrative.narrative_splice(None, None, audit)
        assert "wages" in html
        assert "productivity" in html
        assert "&sect;4.4" in html

    def test_no_flag_when_clean(self):
        audit = {
            "splices": {"wages": {"boundary_continuity_ok": True}},
        }
        html = narrative.narrative_splice(None, None, audit)
        assert "Audit note" not in html

    def test_handles_no_inputs_gracefully(self):
        html = narrative.narrative_splice(None, None, None)
        assert isinstance(html, str)
        assert len(html) > 0


# ---------------------------------------------------------------------------
# narrative_rpph
# ---------------------------------------------------------------------------

class TestRpphNarrative:
    def test_quotes_composite_endpoints(self):
        idx = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
        composite = pd.Series(np.linspace(12000, 16000, len(idx)), index=idx)
        html = narrative.narrative_rpph(composite, None, None)
        assert "12,000 hours" in html
        assert "16,000 hours" in html

    def test_recognises_rising_rpph(self):
        idx = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
        composite = pd.Series(np.linspace(10000, 16000, len(idx)), index=idx)
        html = narrative.narrative_rpph(composite, None, None)
        assert "rising" in html.lower() or "+60" in html

    def test_recognises_falling_rpph(self):
        idx = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
        composite = pd.Series(np.linspace(20000, 10000, len(idx)), index=idx)
        html = narrative.narrative_rpph(composite, None, None)
        # When RPPH falls, prose should describe the welfare-positive interpretation.
        assert "falling" in html.lower()
        assert "consistent with the conventional welfare narrative" in html

    def test_decomposes_by_item_in_html_table(self):
        idx = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
        by_item = pd.DataFrame({
            "gasoline": np.linspace(2.0, 1.5, len(idx)),    # falling: tracks productivity
            "housing":  np.linspace(5000, 12000, len(idx)),  # rising sharply: divergent
            "tuition":  np.linspace(100, 400, len(idx)),     # rising: divergent
        }, index=idx)
        html = narrative.narrative_rpph(None, by_item, None)
        # Item names appear in the table
        assert "gasoline" in html
        assert "housing" in html
        assert "tuition" in html
        # The table is rendered
        assert "<table" in html

    def test_classifies_divergent_vs_tracking(self):
        idx = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
        by_item = pd.DataFrame({
            "gasoline": np.linspace(2.0, 1.0, len(idx)),    # tracking
            "housing":  np.linspace(5000, 15000, len(idx)),  # divergent
        }, index=idx)
        html = narrative.narrative_rpph(None, by_item, None)
        assert "productivity-tracking" in html
        assert "divergent" in html

    def test_handles_empty_inputs(self):
        html = narrative.narrative_rpph(None, None, None)
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# narrative_wicr
# ---------------------------------------------------------------------------

class TestWicrNarrative:
    def test_no_high_periods_omits_cluster_text(self):
        idx = pd.date_range("1990-01-01", "2024-12-01", freq="MS")
        panel = pd.DataFrame({
            "wicr_smoothed": np.full(len(idx), 0.4),
            "high_wicr_run": [False] * len(idx),
        }, index=idx)
        html = narrative.narrative_wicr(panel, {"n_high_wicr_periods": 0})
        # No cluster bullets when there are no high periods.
        assert "<ul" not in html or "1973-1981 stagflation" not in html

    def test_high_periods_emit_cluster_bullets(self):
        # Two distinct high-WICR clusters.
        idx = pd.date_range("1970-01-01", "2024-12-01", freq="MS")
        run = pd.Series(False, index=idx)
        # Cluster 1: 1973-1981 (Jul-1973 to Dec-1981)
        run.loc["1973-07-01":"1981-12-01"] = True
        # Cluster 2: 2021-2024
        run.loc["2021-01-01":"2024-12-01"] = True
        panel = pd.DataFrame({
            "wicr_smoothed": np.full(len(idx), 0.6),
            "high_wicr_run": run,
        }, index=idx)
        html = narrative.narrative_wicr(panel, {"n_high_wicr_periods": int(run.sum())})
        assert "2 distinct" in html
        assert "1973" in html
        assert "2021" in html

    def test_audit_count_appears(self):
        idx = pd.date_range("1990-01-01", "2024-12-01", freq="MS")
        panel = pd.DataFrame({
            "wicr_smoothed": np.full(len(idx), 0.4),
            "high_wicr_run": [False] * len(idx),
        }, index=idx)
        html = narrative.narrative_wicr(panel, {"n_high_wicr_periods": 537})
        assert "537" in html

    def test_handles_no_panel(self):
        html = narrative.narrative_wicr(None, None)
        # Just the framing paragraph.
        assert "capture question" in html


# ---------------------------------------------------------------------------
# narrative_prwdi
# ---------------------------------------------------------------------------

class TestPrwdiNarrative:
    def test_quotes_endpoint_and_base(self):
        idx = pd.DatetimeIndex([f"{y}-12-31" for y in range(1947, 2025)])
        n = len(idx)
        panel = pd.DataFrame({
            "prwdi": np.linspace(1.0, 1.66, n),
            "delta_prwdi_annual": np.full(n, 0.005),
        }, index=idx)
        html = narrative.narrative_prwdi(panel, {"base_year": 1947, "prwdi_at_end": 1.66})
        assert "1.66" in html
        assert "1947" in html

    def test_recognises_mishel_bivens_range(self):
        idx = pd.DatetimeIndex([f"{y}-12-31" for y in range(1947, 2025)])
        n = len(idx)
        panel = pd.DataFrame({
            "prwdi": np.linspace(1.0, 1.66, n),
            "delta_prwdi_annual": np.full(n, 0.005),
        }, index=idx)
        html = narrative.narrative_prwdi(panel, {"base_year": 1947, "prwdi_at_end": 1.66})
        # In-range: should call out consistency
        assert "Mishel-Bivens" in html
        assert "consistent with" in html

    def test_flags_out_of_range_prwdi(self):
        idx = pd.DatetimeIndex([f"{y}-12-31" for y in range(1947, 2025)])
        n = len(idx)
        # PRWDI = 5.0 is well outside the Mishel-Bivens range.
        panel = pd.DataFrame({
            "prwdi": np.linspace(1.0, 5.0, n),
            "delta_prwdi_annual": np.full(n, 0.02),
        }, index=idx)
        html = narrative.narrative_prwdi(panel, {"base_year": 1947, "prwdi_at_end": 5.0})
        assert "outside" in html or "Stansbury-Summers" in html

    def test_split_sample_pre_post_1973(self):
        idx = pd.DatetimeIndex([f"{y}-12-31" for y in range(1947, 2025)])
        # ΔPRWDI low pre-1973, high post-1973
        delta = np.where(np.array([d.year for d in idx]) < 1973, 0.001, 0.012)
        panel = pd.DataFrame({
            "prwdi": np.cumprod(1 + delta),
            "delta_prwdi_annual": delta,
        }, index=idx)
        html = narrative.narrative_prwdi(panel, {"base_year": 1947})
        assert "pre-1973" in html
        assert "post-1973" in html


# ---------------------------------------------------------------------------
# narrative_synthesis
# ---------------------------------------------------------------------------

class TestSynthesis:
    def test_full_phase2_synthesis(self):
        idx = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
        composite = pd.Series(np.linspace(12000, 16000, len(idx)), index=idx)
        html = narrative.narrative_synthesis(
            composite=composite,
            rpph_by_item=None,
            wicr_audit={"n_high_wicr_periods": 537},
            prwdi_audit={"base_year": 1947, "prwdi_at_end": 1.66},
            counterfactual_audit=None,
        )
        # All three questions referenced
        assert "basket question" in html
        assert "capture question" in html
        assert "decoupling question" in html
        # Headline numbers from the inputs appear
        assert "537" in html
        assert "1.66" in html
        # Phase 3 not present yet, so it's noted as absent
        assert "counterfactual" in html.lower()

    def test_with_counterfactual(self):
        html = narrative.narrative_synthesis(
            composite=None, rpph_by_item=None,
            wicr_audit=None, prwdi_audit=None,
            counterfactual_audit={
                "final_pct_gap": 0.32,
                "final_pct_gap_ci_low": 0.21,
                "final_pct_gap_ci_high": 0.43,
            },
        )
        assert "+32%" in html
        assert "21%" in html or "+21%" in html

    def test_thesis_recap_present(self):
        html = narrative.narrative_synthesis(
            composite=None, rpph_by_item=None,
            wicr_audit=None, prwdi_audit=None,
            counterfactual_audit=None,
        )
        assert "compound fragility" in html
        assert "H1-H4" in html or "Phase 3" in html


# ---------------------------------------------------------------------------
# narrative_open_questions
# ---------------------------------------------------------------------------

class TestOpenQuestions:
    def test_lists_h1_through_h4_when_phase3_absent(self):
        html = narrative.narrative_open_questions(["phase1", "phase2"])
        assert "H1" in html
        assert "H2" in html
        assert "H3" in html
        assert "H4" in html
        assert "rpps.breaks" in html
        assert "rpps.regression" in html
        assert "rpps.counterfactual" in html

    def test_acknowledges_when_phase3_present(self):
        html = narrative.narrative_open_questions(["phase1", "phase2", "phase3"])
        assert "Phase 3 outputs are present" in html
        # No "registered hypotheses ... do not yet test" language
        assert "does not yet test" not in html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestFormatHelpers:
    def test_fmt_pct_signs(self):
        assert narrative._fmt_pct(0.28) == "+28%"
        assert narrative._fmt_pct(-0.12) == "-12%"
        assert narrative._fmt_pct(0.0) == "+0%"

    def test_fmt_pct_decimals(self):
        assert narrative._fmt_pct(0.0234, decimals=2) == "+2.34%"

    def test_fmt_int_thousands_sep(self):
        assert narrative._fmt_int(1234567) == "1,234,567"

    def test_safe_first_last_returns_none_for_empty(self):
        s = pd.Series([], dtype=float)
        assert narrative._safe_first_last(s) == (None, None, None, None)

    def test_safe_first_last_drops_nans(self):
        idx = pd.date_range("2000-01-01", periods=5, freq="MS")
        s = pd.Series([np.nan, 10.0, 20.0, 30.0, np.nan], index=idx)
        v0, t0, v1, t1 = narrative._safe_first_last(s)
        assert v0 == 10.0
        assert v1 == 30.0
        assert t0 == idx[1]
        assert t1 == idx[3]
