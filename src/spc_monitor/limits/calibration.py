import numpy as np

from spc_monitor.sim.rl import estimate_arl


def calibrate_ucl_arl0(
    statistics_ic: np.ndarray,
    *,
    target_arl0: float,
    iters: int = 3000,
    seed: int = 123,
    max_steps: int = 100000,
    tol: float = 5.0,
    max_bisect_iter: int = 30,
    lo_q: float = 0.90,
    hi_q: float = 0.9999,
) -> float:
    """
    Calibra UCL para aproximar ARL0 usando:
    - Simulación de RL (sampleando el estadístico in-control con reemplazo)
    - Bisección sobre UCL

    Nota (MVP):
    - Aquí modelamos el estadístico IC con su distribución empírica (statistics_ic).
    - En una versión avanzada, sample_fn podría generar X_t y stat_fn sería tu estadístico real.
    """
    stats = np.asarray(statistics_ic, dtype=float)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        raise ValueError("statistics_ic is empty or non-finite.")
    if target_arl0 <= 1:
        raise ValueError("target_arl0 must be > 1.")

    # sample_fn: devuelve un estadístico IC (con reemplazo)
    def sample_fn(rng: np.random.Generator):
        idx = rng.integers(0, stats.size)
        return stats[idx]

    # stat_fn: identidad (ya estamos sampleando el estadístico)
    def stat_fn(s):
        return float(s)

    # Bracket inicial (UCL bajo y alto)
    lo = float(np.quantile(stats, lo_q))
    hi = float(np.quantile(stats, hi_q))
    if hi <= lo:
        hi = float(stats.max()) + 1e-9

    # Función: ARL(ucl)
    def arl_of(ucl: float, local_seed: int) -> float:
        return estimate_arl(
            n_iters=int(iters),
            seed=int(local_seed),
            sample_fn=sample_fn,
            stat_fn=stat_fn,
            ucl=float(ucl),
            lcl=None,
            max_steps=int(max_steps),
        )

    # Asegurar que el bracket cubra el target:
    # ARL crece con UCL, así que queremos:
    # arl(lo) <= target <= arl(hi)
    arl_lo = arl_of(lo, seed + 1)
    arl_hi = arl_of(hi, seed + 2)

    expand = 0
    while arl_lo > target_arl0 and expand < 10:
        # si ARL ya es demasiado alto, baja lo
        lo = max(0.0, lo * 0.8)
        arl_lo = arl_of(lo, seed + 10 + expand)
        expand += 1

    expand = 0
    while arl_hi < target_arl0 and expand < 10:
        # si ARL aún es bajo, sube hi
        hi = hi * 1.5 + 1e-9
        arl_hi = arl_of(hi, seed + 100 + expand)
        expand += 1

    # Bisección
    for k in range(int(max_bisect_iter)):
        mid = 0.5 * (lo + hi)
        arl_mid = arl_of(mid, seed + 1000 + k)

        if abs(arl_mid - target_arl0) <= tol:
            return float(mid)

        if arl_mid < target_arl0:
            # UCL muy bajo -> ARL bajo -> subir UCL
            lo = mid
        else:
            # UCL muy alto -> ARL alto -> bajar UCL
            hi = mid

    return float(0.5 * (lo + hi))