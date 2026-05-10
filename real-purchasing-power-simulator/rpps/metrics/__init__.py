"""
rpps.metrics — derived welfare metrics.

Implements the three derived metrics specified in §5.1 of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

    rpph   :  Real Purchasing Power Hours
    wicr   :  Wage-Inflation Capture Ratio
    prwdi  :  Productivity-Real-Wage Decoupling Index
"""

from rpps.metrics.compute_all import run as compute_all_metrics
from rpps.metrics.prwdi import (
    DEFAULT_BASE_YEAR,
    PrwdiResult,
    compute_prwdi,
    save_prwdi_result,
)
from rpps.metrics.rpph import (
    RpphResult,
    compute_rpph,
    labor_hours_for_item,
    save_rpph_result,
)
from rpps.metrics.wicr import (
    WICR_HIGH_THRESHOLD,
    WICR_LOW_THRESHOLD,
    WicrResult,
    compute_wicr,
    save_wicr_result,
)

__all__ = [
    "RpphResult",
    "compute_rpph",
    "labor_hours_for_item",
    "save_rpph_result",
    "WICR_LOW_THRESHOLD",
    "WICR_HIGH_THRESHOLD",
    "WicrResult",
    "compute_wicr",
    "save_wicr_result",
    "DEFAULT_BASE_YEAR",
    "PrwdiResult",
    "compute_prwdi",
    "save_prwdi_result",
    "compute_all_metrics",
]
