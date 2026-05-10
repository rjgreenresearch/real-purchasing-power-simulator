"""
rpps.visualization - Matplotlib figure generation for the report module.

Pure functions: each takes data (DataFrames / Series / dicts) and returns a
matplotlib Figure. None of these write to disk. The report module decides what
to do with the figures (PNG embed, savefig, etc.).

Style philosophy
----------------
- Tufte-leaning: minimal chartjunk, light gridlines, no top/right spines
- System font stack for cross-platform professional appearance
- Color palette: muted blues for primary signals, ambers for thresholds,
  greys for reference lines
- Title hierarchy: figure suptitle for the main claim, axis title for context
- Date axes: pandas-style major ticks (every 10 years for long series)
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # Headless backend; no display required.
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Muted, print-friendly palette.
COLOR_PRIMARY = "#1f4e79"      # deep blue
COLOR_SECONDARY = "#c2553f"    # muted red-orange
COLOR_TERTIARY = "#6d8c4a"     # muted green
COLOR_GRID = "#dcdcdc"
COLOR_THRESHOLD_LO = "#e0a458"  # amber
COLOR_THRESHOLD_HI = "#a83232"  # dark red
COLOR_REFERENCE = "#7f7f7f"    # neutral grey
COLOR_REGIME_HIGH = "#fff1d6"  # very pale amber for shading
COLOR_REGIME_HIGH_BOUNDARY = "#9e7a1f"


def _apply_style(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply the project visual style to a (Figure, Axes) pair."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_REFERENCE)
    ax.spines["bottom"].set_color(COLOR_REFERENCE)
    ax.tick_params(colors=COLOR_REFERENCE, which="both")
    ax.yaxis.label.set_color(COLOR_REFERENCE)
    ax.xaxis.label.set_color(COLOR_REFERENCE)
    ax.title.set_color("black")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6, alpha=0.8)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")


def _format_date_axis(ax: plt.Axes, span_years: float) -> None:
    """Pick reasonable major/minor tick locators based on time span."""
    if span_years > 80:
        major = mdates.YearLocator(20)
        minor = mdates.YearLocator(5)
    elif span_years > 30:
        major = mdates.YearLocator(10)
        minor = mdates.YearLocator(2)
    elif span_years > 10:
        major = mdates.YearLocator(5)
        minor = mdates.YearLocator(1)
    else:
        major = mdates.YearLocator(2)
        minor = mdates.YearLocator(1)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_minor_locator(minor)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _years_in_span(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 1.0
    return (idx.max() - idx.min()).days / 365.25


# ---------------------------------------------------------------------------
# Splice figures
# ---------------------------------------------------------------------------

def figure_spliced_wages(
    spliced: pd.Series,
    splice_boundary: str | pd.Timestamp = "1939-01-01",
    title: str = "Spliced manufacturing wage, 1920-present",
) -> plt.Figure:
    """Plot the spliced wage series with a marker at the splice boundary.

    Parameters
    ----------
    spliced : pd.Series
        Output of `rpps.nber_splice.build_wage_splice().spliced`.
    splice_boundary : str | pd.Timestamp
        Date of the legacy/modern boundary, drawn as a vertical reference line.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    _apply_style(fig, ax)

    boundary = pd.Timestamp(splice_boundary)
    pre = spliced[spliced.index < boundary]
    post = spliced[spliced.index >= boundary]

    if len(pre):
        ax.plot(pre.index, pre.values, color=COLOR_SECONDARY, linewidth=1.0,
                label=f"Legacy (NBER M08142, pre-{boundary.year})")
    if len(post):
        ax.plot(post.index, post.values, color=COLOR_PRIMARY, linewidth=1.0,
                label=f"Modern (BLS AHEMAN, {boundary.year}+)")
    ax.axvline(boundary, color=COLOR_REFERENCE, linewidth=0.7,
               linestyle="--", alpha=0.7)

    ax.set_yscale("log")
    ax.set_ylabel("Hourly wage, USD (log scale)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    _format_date_axis(ax, _years_in_span(spliced.index))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def figure_spliced_productivity(
    spliced: pd.Series,
    splice_boundary: str | pd.Timestamp = "1947-01-01",
    title: str = "Spliced productivity index, 1925-present",
) -> plt.Figure:
    """Plot the spliced productivity index (Kendrick legacy + OPHNFB modern)."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    _apply_style(fig, ax)

    boundary = pd.Timestamp(splice_boundary)
    pre = spliced[spliced.index < boundary]
    post = spliced[spliced.index >= boundary]

    if len(pre):
        ax.plot(pre.index, pre.values, color=COLOR_SECONDARY, linewidth=1.2,
                marker="o", markersize=3,
                label=f"Legacy (Kendrick 1961, pre-{boundary.year})")
    if len(post):
        ax.plot(post.index, post.values, color=COLOR_PRIMARY, linewidth=1.2,
                label=f"Modern (BLS OPHNFB, {boundary.year}+)")
    ax.axvline(boundary, color=COLOR_REFERENCE, linewidth=0.7,
               linestyle="--", alpha=0.7)

    ax.set_ylabel("Productivity index")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    _format_date_axis(ax, _years_in_span(spliced.index))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# RPPH figures
# ---------------------------------------------------------------------------

def figure_rpph_composite(
    composite: pd.Series,
    title: str = "Real Purchasing Power Hours - composite basket",
) -> plt.Figure:
    """Plot the composite RPPH series (hours-of-labor to buy the basket)."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    _apply_style(fig, ax)

    valid = composite.dropna()
    ax.plot(valid.index, valid.values, color=COLOR_PRIMARY, linewidth=1.2)
    ax.fill_between(valid.index, valid.values, alpha=0.10, color=COLOR_PRIMARY)

    ax.set_ylabel("Hours of labor required")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    _format_date_axis(ax, _years_in_span(valid.index))
    fig.tight_layout()
    return fig


def figure_rpph_by_item(
    by_item: pd.DataFrame,
    title: str = "Real Purchasing Power Hours - per-item",
    items_to_show: list[str] | None = None,
) -> plt.Figure:
    """Plot per-item RPPH, one line per item, on a shared axis (log scale)."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    _apply_style(fig, ax)

    cols = items_to_show if items_to_show else list(by_item.columns)
    palette = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY,
               "#7c4ba0", "#446e7f", "#a06450"]
    for i, col in enumerate(cols):
        if col not in by_item.columns:
            continue
        s = by_item[col].dropna()
        if s.empty:
            continue
        color = palette[i % len(palette)]
        ax.plot(s.index, s.values, label=col, color=color, linewidth=1.1)

    ax.set_yscale("log")
    ax.set_ylabel("Hours of labor (log scale)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    _format_date_axis(ax, _years_in_span(by_item.index))
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# WICR figures
# ---------------------------------------------------------------------------

def figure_wicr(
    wicr_panel: pd.DataFrame,
    low_threshold: float = 0.50,
    high_threshold: float = 0.80,
    title: str = "Wage-Inflation Capture Ratio",
) -> plt.Figure:
    """Plot WICR with threshold lines and shaded sustained-high regions.

    Parameters
    ----------
    wicr_panel : pd.DataFrame
        Output panel with columns including 'wicr_smoothed' and 'high_wicr_run'.
    low_threshold, high_threshold : float
        The two regime cutoffs. Default 0.50 and 0.80.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    _apply_style(fig, ax)

    smoothed = wicr_panel["wicr_smoothed"].dropna()
    if not smoothed.empty:
        ax.plot(smoothed.index, smoothed.values,
                color=COLOR_PRIMARY, linewidth=1.1, label="Smoothed WICR")

    if "high_wicr_run" in wicr_panel.columns:
        runs = wicr_panel["high_wicr_run"].fillna(False).astype(bool)
        # Shade contiguous True runs.
        in_run = False
        run_start: pd.Timestamp | None = None
        for ts, val in runs.items():
            if val and not in_run:
                run_start = ts
                in_run = True
            elif not val and in_run:
                assert run_start is not None  # for type-narrowing
                ax.axvspan(run_start, ts, facecolor=COLOR_REGIME_HIGH, alpha=0.55,
                           edgecolor=COLOR_REGIME_HIGH_BOUNDARY,
                           linewidth=0.4, zorder=0)
                in_run = False
        if in_run:
            assert run_start is not None
            ax.axvspan(run_start, runs.index[-1], facecolor=COLOR_REGIME_HIGH,
                       alpha=0.55, edgecolor=COLOR_REGIME_HIGH_BOUNDARY,
                       linewidth=0.4, zorder=0)

    ax.axhline(low_threshold, color=COLOR_THRESHOLD_LO, linestyle=":",
               linewidth=1.0, label=f"Low/medium threshold ({low_threshold:.2f})")
    ax.axhline(high_threshold, color=COLOR_THRESHOLD_HI, linestyle="--",
               linewidth=1.1, label=f"High threshold ({high_threshold:.2f})")
    ax.axhline(0.0, color=COLOR_REFERENCE, linewidth=0.5, alpha=0.5)
    ax.axhline(1.0, color=COLOR_REFERENCE, linewidth=0.5, alpha=0.5,
               linestyle=":")

    ax.set_ylabel("WICR = inflation_yoy / wage_growth_yoy")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    if not smoothed.empty:
        _format_date_axis(ax, _years_in_span(smoothed.index))
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_ylim(-0.5, 2.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PRWDI figures
# ---------------------------------------------------------------------------

def figure_prwdi(
    prwdi_panel: pd.DataFrame,
    base_year: int = 1947,
    title: str = "Productivity-Real-Wage Decoupling Index",
) -> plt.Figure:
    """Plot PRWDI with productivity and compensation index overlays."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.5), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    _apply_style(fig, ax1)
    _apply_style(fig, ax2)

    if "productivity_index" in prwdi_panel.columns:
        prod = prwdi_panel["productivity_index"].dropna()
        ax1.plot(prod.index, prod.values, color=COLOR_PRIMARY,
                 linewidth=1.2, label=f"Productivity (={base_year}=1.0)")
    if "compensation_index" in prwdi_panel.columns:
        comp = prwdi_panel["compensation_index"].dropna()
        ax1.plot(comp.index, comp.values, color=COLOR_SECONDARY,
                 linewidth=1.2, label=f"Real compensation (={base_year}=1.0)")
    ax1.axhline(1.0, color=COLOR_REFERENCE, linewidth=0.5, linestyle=":")
    ax1.set_ylabel(f"Index ({base_year} = 1.0)")
    ax1.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    if "prwdi" in prwdi_panel.columns:
        prwdi = prwdi_panel["prwdi"].dropna()
        ax2.plot(prwdi.index, prwdi.values, color=COLOR_TERTIARY,
                 linewidth=1.2, label="PRWDI")
        ax2.fill_between(prwdi.index, 1.0, prwdi.values,
                         where=(prwdi.values >= 1.0),
                         alpha=0.20, color=COLOR_TERTIARY,
                         label="Productivity > compensation")
    ax2.axhline(1.0, color=COLOR_REFERENCE, linewidth=0.6, linestyle="--",
                label="Coupled baseline")
    ax2.set_ylabel("PRWDI ratio")
    ax2.set_xlabel("Year")
    ax2.legend(loc="upper left", frameon=False, fontsize=8)

    span = _years_in_span(prwdi_panel.index) if len(prwdi_panel) > 0 else 50.0
    _format_date_axis(ax2, span)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Phase 3 figures (used when break/regression/counterfactual results exist)
# ---------------------------------------------------------------------------

def figure_regimes(
    series: pd.Series,
    regime_assignments: pd.Series,
    title: str = "Regime structure",
) -> plt.Figure:
    """Plot a series with regime boundaries shaded."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    _apply_style(fig, ax)

    valid = series.dropna()
    ax.plot(valid.index, valid.values, color=COLOR_PRIMARY, linewidth=1.0)

    palette = ["#fff1d6", "#e6e9f2", "#e8f0e8", "#f5e1d6", "#e1e8ee"]
    if not regime_assignments.empty:
        regimes = regime_assignments.dropna()
        unique = sorted(regimes.unique())
        for r in unique:
            mask = regimes == r
            if not mask.any():
                continue
            r_dates = regimes.index[mask]
            ax.axvspan(r_dates.min(), r_dates.max(),
                       color=palette[int(r) % len(palette)],
                       alpha=0.45, zorder=0)

    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    if not valid.empty:
        _format_date_axis(ax, _years_in_span(valid.index))
    fig.tight_layout()
    return fig


def figure_counterfactual(
    counterfactual_panel: pd.DataFrame,
    title: str = "Counterfactual real-compensation gap",
) -> plt.Figure:
    """Plot actual vs counterfactual compensation, shading the gap."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    _apply_style(fig, ax)

    actual = counterfactual_panel["actual_compensation"].dropna()
    cf = counterfactual_panel["counterfactual_compensation"].dropna()

    ax.plot(actual.index, actual.values, color=COLOR_SECONDARY,
            linewidth=1.4, label="Actual real compensation")
    ax.plot(cf.index, cf.values, color=COLOR_PRIMARY, linewidth=1.4,
            linestyle="--", label="Counterfactual (1948-71 distribution)")
    ax.fill_between(cf.index, cf.values, actual.reindex(cf.index).values,
                    where=(cf.values >= actual.reindex(cf.index).values),
                    alpha=0.18, color=COLOR_PRIMARY, label="Gap")

    ax.set_ylabel("Real compensation, indexed")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    _format_date_axis(ax, _years_in_span(actual.index))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helpers for the report module
# ---------------------------------------------------------------------------

def figure_to_png_bytes(fig: plt.Figure) -> bytes:
    """Serialize a Figure to PNG bytes (for HTML embed via base64)."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def figure_to_base64(fig: plt.Figure) -> str:
    """Serialize a Figure to a base64 PNG string suitable for <img src='data:'>."""
    import base64
    return base64.b64encode(figure_to_png_bytes(fig)).decode("ascii")
