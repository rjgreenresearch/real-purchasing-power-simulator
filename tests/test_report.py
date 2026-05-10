"""Tests for rpps.report — HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rpps import report

# ---------------------------------------------------------------------------
# Fixtures: write a fake "processed" dir to tmp_path
# ---------------------------------------------------------------------------

def _build_minimal_processed_dir(tmp_path: Path) -> Path:
    """Build a minimal-but-realistic data/processed/ for the report to read."""
    p = tmp_path / "processed"
    p.mkdir()

    # Wage splice
    wage_idx = pd.date_range("1920-01-01", periods=1200, freq="MS")
    pd.Series(
        np.linspace(0.40, 30.0, len(wage_idx)),
        index=wage_idx, name="spliced_AHEMAN",
    ).to_csv(p / "spliced_wages.csv", header=True)

    # Productivity splice
    prod_idx = pd.DatetimeIndex([f"{y}-12-31" for y in range(1925, 2025)])
    pd.Series(
        np.linspace(12.0, 119.0, len(prod_idx)),
        index=prod_idx, name="spliced_OPHNFB",
    ).to_csv(p / "spliced_productivity.csv", header=True)

    # Splice audit
    splice_audit = {
        "built_at": "2026-05-10T00:00:00+00:00",
        "splices": {
            "wages": {
                "legacy_series_id": "M08142USM055NNBR",
                "modern_series_id": "AHEMAN",
                "adjustment_factor": 0.7340,
                "overlap_start": "1939-01-01",
                "overlap_end": "1942-12-31",
                "n_overlap_obs": 48,
                "boundary_continuity_ok": False,
                "first_obs_date": "1920-01-01",
                "last_obs_date": "2026-04-01",
                "n_observations": 1276,
            },
            "productivity": {
                "legacy_series_id": "kendrick_productivity",
                "modern_series_id": "OPHNFB",
                "adjustment_factor": 0.1540,
                "overlap_start": "1947-01-01",
                "overlap_end": "1957-12-31",
                "n_overlap_obs": 11,
                "boundary_continuity_ok": False,
                "first_obs_date": "1925-12-31",
                "last_obs_date": "2026-12-31",
                "n_observations": 102,
            },
        },
    }
    (p / "splice_audit.json").write_text(json.dumps(splice_audit, indent=2))

    # RPPH composite + by_item
    rpph_idx = pd.date_range("1990-01-01", periods=400, freq="MS")
    pd.Series(np.linspace(150, 80, len(rpph_idx)),
              index=rpph_idx, name="rpph_composite").to_csv(
        p / "rpph_composite.csv", header=True)
    pd.DataFrame({
        "gasoline": np.linspace(2.5, 1.4, len(rpph_idx)),
        "beef": np.linspace(8.0, 4.5, len(rpph_idx)),
        "tuition": np.linspace(40.0, 65.0, len(rpph_idx)),
    }, index=rpph_idx).to_csv(p / "rpph_by_item.csv")
    (p / "rpph_audit.json").write_text(json.dumps({
        "wage_series_id": "AHEMAN+M08142USM055NNBR_spliced",
        "items_used": ["gasoline", "beef", "tuition"],
        "n_observations": 400,
    }))

    # WICR panel
    pd.DataFrame({
        "wage_growth_yoy": np.full(len(rpph_idx), 0.04),
        "inflation_yoy": np.full(len(rpph_idx), 0.025),
        "wicr_yoy": np.full(len(rpph_idx), 0.6),
        "wicr_smoothed": np.full(len(rpph_idx), 0.6),
        "regime_label": ["medium"] * len(rpph_idx),
        "high_wicr_run": [False] * len(rpph_idx),
    }, index=rpph_idx).to_csv(p / "wicr_panel.csv")
    (p / "wicr_audit.json").write_text(json.dumps({
        "n_observations": len(rpph_idx),
        "n_high_wicr_periods": 0,
    }))

    # PRWDI panel
    pd.DataFrame({
        "productivity_index": np.linspace(1.0, 3.0, len(prod_idx)),
        "compensation_index": np.linspace(1.0, 1.7, len(prod_idx)),
        "prwdi": np.linspace(1.0, 3.0, len(prod_idx)) /
                 np.linspace(1.0, 1.7, len(prod_idx)),
        "delta_prwdi_annual": np.full(len(prod_idx), 0.01),
    }, index=prod_idx).to_csv(p / "prwdi_panel.csv")
    (p / "prwdi_audit.json").write_text(json.dumps({
        "base_year": 1947,
        "n_observations": len(prod_idx),
        "prwdi_at_end": 1.76,
    }))

    # Run summary
    (p / "metrics_run_summary.json").write_text(json.dumps({
        "rpps_version": "0.3.7",
        "started_at_utc": "2026-05-10T00:00:00+00:00",
        "finished_at_utc": "2026-05-10T00:00:30+00:00",
        "frequency": "M",
        "base_year": 1947,
        "n_ok": 3,
        "n_error": 0,
    }))

    return p


# ---------------------------------------------------------------------------
# load_inputs
# ---------------------------------------------------------------------------

class TestLoadInputs:
    def test_phase1_and_phase2_detected(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        assert "phase1" in inp.available_phases
        assert "phase2" in inp.available_phases
        assert "phase3" not in inp.available_phases

    def test_loads_spliced_wages_as_series(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        assert isinstance(inp.spliced_wages, pd.Series)
        assert len(inp.spliced_wages) == 1200

    def test_loads_rpph_by_item_as_dataframe(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        assert isinstance(inp.rpph_by_item, pd.DataFrame)
        assert set(inp.rpph_by_item.columns) == {"gasoline", "beef", "tuition"}

    def test_loads_audit_json(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        assert inp.splice_audit is not None
        assert "splices" in inp.splice_audit

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            report.load_inputs(tmp_path / "doesnt-exist")

    def test_partial_processed_dir(self, tmp_path):
        # Only RPPH outputs present, nothing else.
        p = tmp_path / "partial"
        p.mkdir()
        idx = pd.date_range("2000-01-01", periods=12, freq="MS")
        pd.Series(np.arange(12.0), index=idx).to_csv(
            p / "rpph_composite.csv", header=True)
        inp = report.load_inputs(p)
        assert "phase1" not in inp.available_phases
        assert "phase2" in inp.available_phases
        assert inp.spliced_wages is None


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------

class TestRenderHtml:
    def test_returns_html_string(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        html = report.render_html(inp)
        assert isinstance(html, str)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_key_sections(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        html = report.render_html(inp)
        assert "Executive summary" in html
        assert "Splice" in html
        assert "Real Purchasing Power Hours" in html
        assert "Wage-Inflation Capture Ratio" in html
        assert "Productivity-Real-Wage Decoupling" in html

    def test_embeds_base64_images(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        html = report.render_html(inp)
        # Multiple <img src="data:image/png;base64,..."> tags.
        n_imgs = html.count('src="data:image/png;base64,')
        assert n_imgs >= 4  # spliced wages, productivity, RPPH composite, etc.

    def test_embeds_css(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        html = report.render_html(inp)
        assert "<style>" in html
        assert ".container" in html

    def test_records_rpps_version(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        html = report.render_html(inp)
        from rpps import __version__
        assert __version__ in html

    def test_warns_on_boundary_discontinuity(self, tmp_path):
        # Our fixture has boundary_continuity_ok = False on both splices.
        p = _build_minimal_processed_dir(tmp_path)
        inp = report.load_inputs(p)
        html = report.render_html(inp)
        assert "Boundary continuity flag" in html


# ---------------------------------------------------------------------------
# build_report (end-to-end)
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_writes_file(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        out = tmp_path / "report.html"
        result = report.build_report(processed_dir=p, output_path=out)
        assert out.exists()
        assert result["size_bytes"] > 50_000  # large because of embedded PNGs
        assert "phase1" in result["available_phases"]
        assert "phase2" in result["available_phases"]

    def test_empty_processed_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No processed artifacts"):
            report.build_report(processed_dir=empty, output_path=tmp_path / "x.html")

    def test_creates_output_parent_dirs(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        deep_out = tmp_path / "deep" / "nested" / "dir" / "report.html"
        report.build_report(processed_dir=p, output_path=deep_out)
        assert deep_out.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestReportCLI:
    def test_main_writes_report(self, tmp_path, capsys):
        p = _build_minimal_processed_dir(tmp_path)
        out = tmp_path / "cli-report.html"
        rc = report._main(["--processed-dir", str(p), "--output", str(out)])
        assert rc == 0
        assert out.exists()
        captured = capsys.readouterr()
        assert "Wrote" in captured.out

    def test_main_returns_nonzero_on_missing_dir(self, tmp_path, capsys):
        rc = report._main([
            "--processed-dir", str(tmp_path / "nonexistent"),
            "--output", str(tmp_path / "x.html"),
        ])
        assert rc != 0
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_module_runs_as_script_with_help(self):
        import os
        import subprocess
        import sys
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, "-m", "rpps.report", "--help"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30, env=env,
        )
        assert result.returncode == 0
        assert "--processed-dir" in result.stdout
        assert "--output" in result.stdout
