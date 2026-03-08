from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spc_monitor.objects.limits import Limits
from spc_monitor.objects.result import QCCResult
from spc_monitor.utils.validation import to_numpy_2d
from spc_monitor.limits.calibration import calibrate_ucl_arl0


@dataclass
class MWD2LocationChart:
    """
    Carta MWD2 (Location) - implementación base (MVP).

    - fit(...)        : Phase I (estima parámetros + define/calibra límites)
    - monitor(...)    : Phase II batch (estadístico + violaciones)
    - monitor_one(...) : streaming (una observación)
    - summary(), plot() : utilidades
    - save/load       : persistencia para operación
    """

    center: np.ndarray
    limits: Limits
    feature_cols: Optional[list[str]] = None
    meta: dict[str, Any] | None = None

    # Guardamos stats Phase I para plot/calibración (opcional)
    _ref_statistics: Optional[np.ndarray] = None

    @classmethod
    def fit(
        cls,
        reference,
        feature_cols: Optional[list[str]] = None,
        limits: str = "calibrate",
        calibrate: str = "arl0",
        ucl: Optional[float] = None,
        lcl: Optional[float] = None,
        cl: Optional[float] = None,
        target_arl0: float = 200.0,
        iters: int = 10000,
        seed: int = 123,
        **kwargs,
    ) -> "MWD2LocationChart":
        X = to_numpy_2d(reference, feature_cols=feature_cols)

        # Phase I: estimación base (placeholder; aquí irá tu estimación real)
        center = X.mean(axis=0)

        # Calcula estadísticos sobre referencia (Phase I)
        stats_ic = np.array([cls._statistic_static(x, center) for x in X], dtype=float)

        limits_mode = str(limits).lower().strip()
        calibrate_mode = str(calibrate).lower().strip()

        if limits_mode == "fixed":
            if ucl is None and lcl is None:
                raise ValueError("For limits='fixed', provide at least one of ucl or lcl.")
            lim = Limits(
                lcl=float(lcl) if lcl is not None else None,
                ucl=float(ucl) if ucl is not None else None,
                cl=float(cl) if cl is not None else None,
                meta={"mode": "fixed"},
            )

        elif limits_mode == "calibrate":
            if calibrate_mode != "arl0":
                raise ValueError("MVP supports calibrate='arl0' only (for now).")

            # Stub: usamos cuantil alto para obtener un UCL inicial.
            # Más adelante aquí conectamos bisección + simulación RL real.
            ucl_star = calibrate_ucl_arl0(stats_ic, target_arl0=target_arl0)

            lim = Limits(
                lcl=float(lcl) if lcl is not None else None,
                ucl=float(ucl_star),
                cl=float(cl) if cl is not None else None,
                meta={
                    "mode": "calibrate",
                    "calibrate": "arl0",
                    "target_arl0": float(target_arl0),
                    "iters": int(iters),
                    "seed": int(seed),
                },
            )
        else:
            raise ValueError("limits must be 'fixed' or 'calibrate'.")

        meta = {
            "n_ref": int(X.shape[0]),
            "p": int(X.shape[1]),
        }
        meta.update(kwargs or {})

        return cls(
            center=center,
            limits=lim,
            feature_cols=feature_cols,
            meta=meta,
            _ref_statistics=stats_ic,
        )

    @staticmethod
    def _statistic_static(x_t: np.ndarray, center: np.ndarray) -> float:
        # Placeholder: reemplazar por tu estadístico MWD2 real.
        d = x_t - center
        return float(np.sum(d * d))

    def statistic(self, x_t: np.ndarray) -> float:
        return self._statistic_static(x_t, self.center)

    def monitor(self, stream, feature_cols: Optional[list[str]] = None) -> QCCResult:
        cols = feature_cols if feature_cols is not None else self.feature_cols
        Xs = to_numpy_2d(stream, feature_cols=cols)

        stats = np.array([self.statistic(x) for x in Xs], dtype=float)

        viol = np.zeros(stats.shape[0], dtype=bool)
        if self.limits.ucl is not None:
            viol |= stats > float(self.limits.ucl)
        if self.limits.lcl is not None:
            viol |= stats < float(self.limits.lcl)

        return QCCResult(
            chart="mwd2_location",
            phase="phase2",
            limits=self.limits,
            statistics=stats,
            violations=viol,
            meta={"feature_cols": cols, **(self.meta or {})},
        )

    def monitor_one(self, x_t) -> dict[str, Any]:
        x = to_numpy_2d(x_t).reshape(-1)  # convierte a (p,)
        stat = self.statistic(x)

        alarm = False
        if self.limits.ucl is not None and stat > float(self.limits.ucl):
            alarm = True
        if self.limits.lcl is not None and stat < float(self.limits.lcl):
            alarm = True

        return {"statistic": float(stat), "alarm": bool(alarm), "lcl": self.limits.lcl, "ucl": self.limits.ucl}

    def summary(self) -> str:
        n_ref = None if self.meta is None else self.meta.get("n_ref")
        p = None if self.meta is None else self.meta.get("p")
        return (
            "MWD2LocationChart\n"
            f"- n_ref: {n_ref}\n"
            f"- p: {p}\n"
            f"- lcl: {self.limits.lcl}\n"
            f"- ucl: {self.limits.ucl}\n"
            f"- mode: {None if self.limits.meta is None else self.limits.meta.get('mode')}\n"
        )

    def plot(self, reference: bool = False):
        if reference and self._ref_statistics is not None:
            res = QCCResult(
                chart="mwd2_location",
                phase="phase1",
                limits=self.limits,
                statistics=self._ref_statistics,
                violations=np.zeros(self._ref_statistics.shape[0], dtype=bool),
                meta={"feature_cols": self.feature_cols, **(self.meta or {})},
            )
            return res.plot()

        # Si no pediste reference, solo plot de límites sin datos
        # (puedes cambiar esto luego)
        return None

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MWD2LocationChart":
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}.")
        return obj