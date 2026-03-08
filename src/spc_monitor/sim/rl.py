import numpy as np


def simulate_run_length(
    *,
    rng: np.random.Generator,
    sample_fn,
    stat_fn,
    ucl: float,
    lcl: float | None = None,
    max_steps: int = 100000,
) -> int:
    """
    Simula un Run Length (RL): cuántas muestras hasta que hay señal.

    - sample_fn(rng) -> devuelve una "observación" (puede ser vector o escalar)
    - stat_fn(x) -> devuelve el estadístico (escalar)
    - señal si stat > ucl o (si lcl) stat < lcl
    - truncamiento en max_steps
    """
    for t in range(1, max_steps + 1):
        x_t = sample_fn(rng)
        s = float(stat_fn(x_t))

        if s > float(ucl):
            return t
        if lcl is not None and s < float(lcl):
            return t

    return max_steps


def estimate_arl(
    *,
    n_iters: int,
    seed: int,
    sample_fn,
    stat_fn,
    ucl: float,
    lcl: float | None = None,
    max_steps: int = 100000,
) -> float:
    """
    Estima ARL = E[RL] por Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    rls = np.empty(n_iters, dtype=int)

    for i in range(n_iters):
        rls[i] = simulate_run_length(
            rng=rng,
            sample_fn=sample_fn,
            stat_fn=stat_fn,
            ucl=ucl,
            lcl=lcl,
            max_steps=max_steps,
        )

    return float(rls.mean())