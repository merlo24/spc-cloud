import numpy as np
import matplotlib.pyplot as plt


def plot_result(res):
    t = np.arange(res.statistics.size)

    fig, ax = plt.subplots()
    ax.plot(t, res.statistics)

    if res.limits.ucl is not None:
        ax.axhline(res.limits.ucl, linestyle="--")
    if res.limits.lcl is not None:
        ax.axhline(res.limits.lcl, linestyle="--")
    if res.limits.cl is not None:
        ax.axhline(res.limits.cl, linestyle=":")

    ax.set_title(f"{res.chart} ({res.phase})")
    ax.set_xlabel("t")
    ax.set_ylabel("Statistic")

    return fig, ax