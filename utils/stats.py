import numpy as np


def get_stats(vals: list[float], unbiased_std: bool = False) -> dict[str, float]:
    ddof = 1 if unbiased_std else 0  # whether to divide by N or N-1 the in the formula of std
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=ddof)),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }
