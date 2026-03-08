import numpy as np
import pandas as pd

from spc_monitor import control_chart, MWD2LocationChart


def test_control_chart_build_and_monitor_dataframe():
    df_ref = pd.DataFrame(
        {
            "x1": np.random.normal(size=200),
            "x2": np.random.normal(size=200),
            "x3": np.random.normal(size=200),
        }
    )

    chart = control_chart(
        df_ref,
        chart="mwd2_location",
        feature_cols=["x1", "x2", "x3"],
        limits="calibrate",
        calibrate="arl0",
        target_arl0=50,
        iters=500,
        seed=123,
    )

    df_new = pd.DataFrame(
        {
            "x1": np.random.normal(size=50),
            "x2": np.random.normal(size=50),
            "x3": np.random.normal(size=50),
        }
    )

    res = chart.monitor(df_new, feature_cols=["x1", "x2", "x3"])

    assert res.statistics.shape == (50,)
    assert res.violations.shape == (50,)


def test_class_fit_and_monitor_ndarray():
    X_ref = np.random.normal(size=(200, 2))
    chart = MWD2LocationChart.fit(reference=X_ref, limits="fixed", ucl=10.0)

    X_new = np.random.normal(size=(40, 2))
    res = chart.monitor(X_new)

    assert res.statistics.shape == (40,)
    assert res.violations.shape == (40,)


def test_monitor_one():
    X_ref = np.random.normal(size=(100, 3))
    chart = MWD2LocationChart.fit(reference=X_ref, limits="fixed", ucl=10.0)

    out = chart.monitor_one([0.0, 0.0, 0.0])
    assert "statistic" in out
    assert "alarm" in out