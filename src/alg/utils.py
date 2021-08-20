import cvxpy as cp
import numpy as np


def FDP_coef(delta: float, a: float) -> float:
    """Returns the coefficient to multiply the FDP to create FDP hat."""
    return -np.log(delta) / (a * np.log(1 + -np.log(delta) / a))


def slope_fn(delta: float, a: float) -> float:
    """Slope of the coefficient function w.r.t a.

    This is a strictly decreasing function, which allows us to binary
    search for 0.
    """
    x = a / np.log(1 / delta)
    return np.log(1 + 1 / x) - 1 / (x + 1)


def find_default_a(gamma: float, delta: float, threshold: int,
                   tolerance: float) -> float:
    """Finds the canonical a using binary search."""

    # optimal value for slope to equal to optimize for a
    target = np.log(1 / delta) / (gamma * threshold)

    # binary search
    low = tolerance
    while low > 0 and target > slope_fn(delta, low):
        low /= 2

    high = tolerance
    while target < slope_fn(delta, high):
        high *= 2

    mid = low + (high - low) / 2
    slope_mid = slope_fn(delta, mid)
    while high - low > tolerance:
        if slope_mid > target:
            low = mid
        else:
            high = mid
        mid = low + (high - low) / 2
        slope_mid = slope_fn(delta, mid)
    return mid
