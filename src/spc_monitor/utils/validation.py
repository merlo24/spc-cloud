import numpy as np


def to_numpy_2d(data, feature_cols=None):
    # Acepta: DataFrame, ndarray, lista
    # Devuelve: ndarray 2D float (n, p)

    if hasattr(data, "to_numpy"):
        # pandas DataFrame/Series
        if feature_cols is not None:
            data = data[feature_cols]
        X = data.to_numpy()
    else:
        X = np.asarray(data)

    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("Input data must be 1D or 2D.")

    return X