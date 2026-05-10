"""Tests for rpps.visualization.

We assert on figure structure (number of axes, line counts, label content)
rather than rendered pixels. The rendered output is non-deterministic across
matplotlib versions; the structure is stable and is what users care about.
"""

from __future__ import annotations

import base64

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from rpps import visualization as viz  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_wage_series():
    idx = pd.date_range("1920-01-01", "2024-12-01", freq="MS")
    n = len(idx)
    # Roughly geometric growth from $0.40 to $30.
    log_growth = np.linspace(np.log(0.40), np.log(30.0), n)
    return pd.Series(np.exp(log_growth), index=idx, name="spliced_wage")


@pytest.fixture
def synthetic_productivity_series():
    idx = pd.DatetimeIndex([f"{y}-12-31" for y in range(1925, 2025)])
    n = len(idx)
    log_growth = np.linspace(np.log(12.0), np.log(120.0), n)
    return pd.Series(np.exp(log_growth), index=idx, name="spliced_productivity")


@pytest.fixture
def synthetic_rpph_composite():
    idx = pd.date_range("1990-01-01", "2024-12-01", freq="MS")
    n = len(idx)
    rng = np.random.default_rng(42)
    base = np.linspace(150.0, 80.0, n) + rng.normal(0, 5, n)
    return pd.Series(np.maximum(base, 1.0), index=idx, name="rpph_composite")


@pytest.fixture
def synthetic_rpph_by_item():
    idx = pd.date_range("1990-01-01", "2024-12-01", freq="MS")
    n = len(idx)
    return pd.DataFrame({
        "gasoline": np.linspace(2.5, 1.4, n),
        "beef": np.linspace(8.0, 4.5, n),
        "tuition": np.linspace(40.0, 65.0, n),
    }, index=idx)


@pytest.fixture
def synthetic_wicr_panel():
    idx = pd.date_range("1990-01-01", "2024-12-01", freq="MS")
    n = len(idx)
    rng = np.random.default_rng(7)
    smoothed = np.clip(0.5 + 0.3 * np.sin(np.arange(n) / 24) + rng.normal(0, 0.05, n),
                       -0.2, 1.5)
    high_run = pd.Series(smoothed > 0.80, index=idx)
    return pd.DataFrame({
        "wage_growth_yoy": np.full(n, 0.04),
        "inflation_yoy": np.full(n, 0.025),
        "wicr_yoy": smoothed + rng.normal(0, 0.05, n),
        "wicr_smoothed": smoothed,
        "regime_label": pd.cut(smoothed, bins=[-np.inf, 0.5, 0.8, np.inf],
                                labels=["low", "medium", "high"]),
        "high_wicr_run": high_run,
    }, index=idx)


@pytest.fixture
def synthetic_prwdi_panel():
    idx = pd.DatetimeIndex([f"{y}-12-31" for y in range(1947, 2025)])
    n = len(idx)
    prod = np.linspace(1.0, 3.5, n)  # productivity grows 3.5x
    comp = np.linspace(1.0, 1.8, n)  # compensation grows 1.8x
    return pd.DataFrame({
        "productivity_index": prod,
        "compensation_index": comp,
        "prwdi": prod / comp,
        "delta_prwdi_annual": np.gradient(prod / comp),
    }, index=idx)


# ---------------------------------------------------------------------------
# figure_spliced_wages
# ---------------------------------------------------------------------------

class TestFigureSplicedWages:
    def test_returns_figure(self, synthetic_wage_series):
        fig = viz.figure_spliced_wages(synthetic_wage_series)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_one_axes(self, synthetic_wage_series):
        fig = viz.figure_spliced_wages(synthetic_wage_series)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_contains_two_lines_pre_and_post_boundary(self, synthetic_wage_series):
        # Series spans 1920-2024, boundary is 1939: should produce both legacy
        # and modern segments.
        fig = viz.figure_spliced_wages(synthetic_wage_series, splice_boundary="1939-01-01")
        ax = fig.axes[0]
        assert len(ax.get_lines()) >= 2
        plt.close(fig)

    def test_uses_log_scale(self, synthetic_wage_series):
        fig = viz.figure_spliced_wages(synthetic_wage_series)
        assert fig.axes[0].get_yscale() == "log"
        plt.close(fig)


# ---------------------------------------------------------------------------
# figure_spliced_productivity
# ---------------------------------------------------------------------------

class TestFigureSplicedProductivity:
    def test_basic(self, synthetic_productivity_series):
        fig = viz.figure_spliced_productivity(synthetic_productivity_series)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.get_lines()) >= 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# RPPH figures
# ---------------------------------------------------------------------------

class TestFigureRpphComposite:
    def test_basic(self, synthetic_rpph_composite):
        fig = viz.figure_rpph_composite(synthetic_rpph_composite)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_data_line(self, synthetic_rpph_composite):
        fig = viz.figure_rpph_composite(synthetic_rpph_composite)
        # At least one Line2D drawn.
        assert len(fig.axes[0].get_lines()) >= 1
        plt.close(fig)


class TestFigureRpphByItem:
    def test_one_line_per_item(self, synthetic_rpph_by_item):
        fig = viz.figure_rpph_by_item(synthetic_rpph_by_item)
        assert len(fig.axes[0].get_lines()) == 3
        plt.close(fig)

    def test_filtered_items(self, synthetic_rpph_by_item):
        fig = viz.figure_rpph_by_item(
            synthetic_rpph_by_item, items_to_show=["gasoline", "beef"])
        assert len(fig.axes[0].get_lines()) == 2
        plt.close(fig)

    def test_uses_log_scale(self, synthetic_rpph_by_item):
        fig = viz.figure_rpph_by_item(synthetic_rpph_by_item)
        assert fig.axes[0].get_yscale() == "log"
        plt.close(fig)


# ---------------------------------------------------------------------------
# WICR figure
# ---------------------------------------------------------------------------

class TestFigureWicr:
    def test_basic(self, synthetic_wicr_panel):
        fig = viz.figure_wicr(synthetic_wicr_panel)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_includes_threshold_lines(self, synthetic_wicr_panel):
        fig = viz.figure_wicr(synthetic_wicr_panel,
                              low_threshold=0.50, high_threshold=0.80)
        ax = fig.axes[0]
        # Find the horizontal threshold lines.
        hlines = [line for line in ax.get_lines()
                  if line.get_xdata() is not None and len(line.get_ydata()) > 0
                  and len(set(line.get_ydata())) == 1]
        # We expect at least 2 threshold lines (0.50 and 0.80).
        assert len(hlines) >= 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# PRWDI figure
# ---------------------------------------------------------------------------

class TestFigurePrwdi:
    def test_two_axes(self, synthetic_prwdi_panel):
        # Top: productivity + compensation indices. Bottom: PRWDI ratio.
        fig = viz.figure_prwdi(synthetic_prwdi_panel)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_top_axes_has_two_index_lines(self, synthetic_prwdi_panel):
        fig = viz.figure_prwdi(synthetic_prwdi_panel)
        # Top axes: productivity + compensation + horizontal at 1.0.
        top_lines = fig.axes[0].get_lines()
        assert len(top_lines) >= 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 3 figures
# ---------------------------------------------------------------------------

class TestFigureRegimes:
    def test_basic(self, synthetic_wage_series):
        # Three regimes: pre-1971, 1972-1999, 2000+.
        regimes = pd.Series(0, index=synthetic_wage_series.index, dtype=int)
        regimes[synthetic_wage_series.index >= "1972-01-01"] = 1
        regimes[synthetic_wage_series.index >= "2000-01-01"] = 2
        fig = viz.figure_regimes(synthetic_wage_series, regimes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestFigureCounterfactual:
    def test_basic(self):
        idx = pd.date_range("1972-01-01", periods=200, freq="QE")
        rng = np.random.default_rng(0)
        actual = pd.Series(np.cumprod(1 + rng.normal(0.005, 0.01, 200)) * 100, index=idx)
        cf = pd.Series(np.cumprod(1 + rng.normal(0.008, 0.01, 200)) * 100, index=idx)
        panel = pd.DataFrame({
            "actual_compensation": actual,
            "counterfactual_compensation": cf,
            "gap": cf - actual,
            "pct_gap": (cf - actual) / actual,
        })
        fig = viz.figure_counterfactual(panel)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_figure_to_png_bytes_returns_png(self, synthetic_wage_series):
        fig = viz.figure_spliced_wages(synthetic_wage_series)
        b = viz.figure_to_png_bytes(fig)
        # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
        assert b[:8] == b"\x89PNG\r\n\x1a\n"

    def test_figure_to_base64_round_trip(self, synthetic_wage_series):
        fig = viz.figure_spliced_wages(synthetic_wage_series)
        s = viz.figure_to_base64(fig)
        # Should be a base64 string that decodes to PNG bytes.
        decoded = base64.b64decode(s)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
