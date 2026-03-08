from dataclasses import dataclass
from typing import Any
import numpy as np

from spc_monitor.objects.limits import Limits


@dataclass
class QCCResult:
    chart: str
    phase: str
    limits: Limits
    statistics: np.ndarray
    violations: np.ndarray
    meta: dict[str, Any] | None = None

    @property
    def violations_idx(self) -> np.ndarray:
        return np.where(self.violations)[0]

    def summary(self) -> str:
        n = int(self.statistics.size)
        v = int(self.violations.sum())
        ucl = self.limits.ucl
        lcl = self.limits.lcl
        return (
            f"QCCResult(chart={self.chart}, phase={self.phase})\n"
            f"- n_points: {n}\n"
            f"- violations: {v}\n"
            f"- lcl: {lcl}\n"
            f"- ucl: {ucl}\n"
        )

    def plot(self):
        from spc_monitor.plotting.matplotlib import plot_result
        return plot_result(self)