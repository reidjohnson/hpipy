"""Stineman interpolation functions."""

from typing import Optional

import numpy as np


def stineman_slope(x: np.ndarray, y: np.ndarray, scale: bool = False) -> np.ndarray:
    """Estimate the slope of an interpolating function with Stineman's method.

    This function estimates the slope of a function that passes through a set
    of points in xy-plane. The slopes are calculated based on the tangent of
    circles passing through every three consecutive points as proposed by [1].

    Args:
        x (np.ndarray): Array of x-coordinates of points.
        y (np.ndarray): Array of y-coordinates of points.
        scale (bool, optional): If True, x and y values are normalized before
            the slope calculation.
            Defaults to False.

    Returns:
        np.ndarray: Array of estimated slopes of the interpolant at (x, y).

    References:
        [1] Stineman, Russell W. "A Consistently Well-Behaved Method of
            Interpolation." Creative Computing 6.7 (1980): 54-57.
    """
    m = len(x)
    if m == 2:
        yp = np.repeat(np.diff(y) / np.diff(x), 2)
    else:
        if scale:
            sx = np.diff(np.array([np.min(x), np.max(x)]))
            sy = np.diff(np.array([np.min(y), np.max(y)]))
            if sy <= 0:
                sy = 1  # type: ignore
            x = (x + 1) / sx
            y = y / sy
        dx = np.diff(x)
        dy = np.diff(y)
        yp = np.repeat(np.nan, m)
        dx2dy2p = dx[1:] ** 2 + dy[1:] ** 2
        dx2dy2m = dx[:-1] ** 2 + dy[:-1] ** 2
        yp[1:-1] = (dy[:-1] * dx2dy2p + dy[1:] * dx2dy2m) / (dx[:-1] * dx2dy2p + dx[1:] * dx2dy2m)
        s = dy[0] / dx[0]
        if ((s >= 0) & (s >= yp[1])) | ((s <= 0) & (s <= yp[1])):
            yp[0] = 2 * s - yp[1]
        else:
            yp[0] = s + np.abs(s) * (s - yp[1]) / (np.abs(s) + np.abs(s - yp[1]))
        s = dy[-1] / dx[-1]
        if ((s >= 0) & (s >= yp[-2])) | ((s <= 0) & (s <= yp[-2])):
            yp[-1] = 2 * s - yp[-2]
        else:
            yp[-1] = s + np.abs(s) * (s - yp[-2]) / (np.abs(s) + np.abs(s - yp[-2]))
        if scale:
            yp = yp * sy / sx
    return yp


def parabola_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate the slope of an interpolating parabola.

    This function uses the slope of a parabola passing through points to
    estimate the slope of an interpolant at these points.

    Args:
        x (np.ndarray): Array of x-coordinates of points.
        y (np.ndarray): Array of y-coordinates of points.

    Returns:
        np.ndarray: Estimate of the slope of the interpolant at (x, y).
    """
    m = len(x)
    dx = np.diff(x)
    dy = np.diff(y)
    dydx = dy / dx
    if m == 2:
        yp = np.repeat(dydx, 2)
    else:
        yp = [  # type: ignore
            (dydx[0] * (2 * dx[0] + dx[1]) - dydx[1] * dx[0]) / (dx[0] + dx[1]),
            (dydx[:-1] * dx[-1] + dydx[-1] * dx[:-1]) / (dx[-1] + dx[:-1]),
            (dydx[-1] * (2 * dx[-1] + dx[-2]) - dydx[-2] * dx[-1]) / (dx[-1] + dx[-2]),
        ]
    return yp


def calculate_stineman_interpolant(
    x: np.ndarray,
    y: np.ndarray,
    xout: np.ndarray,
    yp: Optional[np.ndarray] = None,
    method: Optional[str] = None,
) -> np.ndarray:
    """Calculate Stineman interpolation.

    The function calculates the interpolated values based on the method of
    Stineman [1]. The user can specify the method to compute the slope at
    given points if the slope is not known.

    Args:
        x (np.ndarray): Array of x-coordinates of points.
        y (np.ndarray): Array of y-coordinates of points.
        xout (np.ndarray): x-coordinate where the interpolant is to be found.
        yp (np.ndarray, optional): Slopes of the interpolating function at x.
            Optional: only given if they are known, otherwise the argument is
            not used.
            Defaults to None.
        method (str, optional): Method for computing the slope at the given
            points if the slope is not known. With method="stineman",
            Stineman's original method based on an interpolating circle is
            used. Use method="scaledstineman" if scaling of x and y is to be
            carried out before Stineman's method is applied, and use
            method="parabola" to calculate the slopes from a parabola through
            every three points.
            Defaults to None.

    Returns:
        np.ndarray: Array with components 'x' and 'y' with the coordinates of
            the interpolant at the points specified by `xout`.

    References:
        [1] Stineman, Russell W. "A Consistently Well-Behaved Method of
            Interpolation." Creative Computing 6.7 (1980): 54-57.
    """
    if x is None or y is None or xout is None:
        raise ValueError("Wrong number of input arguments: `x`, `y` and `xout` must be specified.")
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(xout, np.ndarray)
        or not issubclass(x.dtype.type, np.number)
        or not issubclass(y.dtype.type, np.number)
        or not issubclass(xout.dtype.type, np.number)
    ):
        raise ValueError("`x`, `y` and `xout` must be numeric vectors.")
    if len(x) < 2:
        raise ValueError("`x` must have 2 or more elements.")
    if len(x) != len(y):
        raise ValueError("`x` must have the same number of elements as `y`.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(xout)):
        raise ValueError("NaNs in `x`, `y` or xout are not allowed.")
    if yp is not None:
        if not isinstance(yp, np.ndarray) or not issubclass(yp.dtype.type, np.number):
            raise ValueError("`yp` must be a numeric vector.")
        if len(y) != len(yp):
            raise ValueError("When specified, `yp` must have the same number of elements as `y`.")
        if np.any(np.isnan(yp)):
            raise ValueError("NaNs in `yp` are not allowed.")
        if method is not None:
            raise ValueError("Method should not be specified if `yp` is given.")

    dx = np.diff(x)
    dy = np.diff(y)

    if np.any(dx <= 0):
        raise ValueError("The values of `x` must strictly increasing.")

    # Calculation of slopes if needed.
    if yp is None:
        if (method is None) or (method == "scaledstineman"):
            yp = stineman_slope(x, y, scale=True)
        elif method == "stineman":
            yp = stineman_slope(x, y, scale=False)
        elif method == "parabola":
            yp = parabola_slope(x, y)

    # Preparations.
    m = len(x)
    s = dy / dx
    k = len(xout)

    ix = np.searchsorted(x, xout, side="right") - 1

    # For edgepoints, allow extrapolation within a tiny range.
    epx = 5 * (np.finfo(float).eps) * np.diff(np.array([np.min(x), np.max(x)]))
    ix[((np.min(x) - epx) <= xout) & (xout <= np.min(x))] = 0
    ix[(np.max(x) <= xout) & (xout <= (np.max(x) + epx))] = m - 2
    idx = (0 <= ix) & (ix <= (m - 2))
    ix1 = ix[idx]
    ix2 = ix1 + 1

    # Computation of the interpolant.
    # Three cases: dyo1dyo2 ==, > and < 0.
    dxo1 = xout[idx] - x[ix1]
    dxo2 = xout[idx] - x[ix2]
    y0o = y[ix1] + s[ix1] * dxo1
    dyo1 = (yp[ix1] - s[ix1]) * dxo1  # type: ignore
    dyo2 = (yp[ix2] - s[ix1]) * dxo2  # type: ignore
    dyo1dyo2 = dyo1 * dyo2
    yo = y0o

    # Skipped for m=2, in which case linear interpolation is sufficient,
    # unless slopes are given.
    if (m > 2) or (yp is not None):
        id = dyo1dyo2 > 0
        yo[id] = y0o[id] + dyo1dyo2[id] / (dyo1[id] + dyo2[id])
        id = dyo1dyo2 < 0
        yo[id] = y0o[id] + dyo1dyo2[id] * (dxo1[id] + dxo2[id]) / (dyo1[id] - dyo2[id]) / (
            dx[ix1][id]
        )

    yout = np.repeat(np.nan, k)
    yout[idx] = yo

    return yout


def fill_missing(arr: np.ndarray, option: str = "locf") -> np.ndarray:
    """Fill missing values.

    Missing values can be replaced with the the most recent present value
    prior to it in the series, known as the Last Observation Carried Forward
    (LOCF), or by the most recent present next value starting from the end
    of the series, known as the Next Observation Carried Backward (NOCB).

    Args:
        arr (np.ndarray): Array with missing values.
        option (str, optional): Algorithm to be used.
            Accepts the following input:
                - "locf": Last Observation Carried Forward
                - "nocb": Next Observation Carried Backward
            Defaults to "locf".

    Returns:
        np.ndarray: Array with missing values filled.
    """
    if option == "nocb":
        arr = np.flip(arr)
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    val = np.maximum.accumulate(idx, axis=0)
    out = arr[val]
    if option == "nocb":
        out = np.flip(out)
    return out


def interpolate_stineman(x: np.ndarray) -> np.ndarray:
    """Interpolate missing values in an array using Stineman's method.

    To avoid NaNs at the beginning and end of the array, any remaining NaNs
    are forward filled (to eliminate end NaNs) and then backward filled (to
    eliminate beginning NaNs).

    Args:
        x (np.ndarray): Array with possible missing values.

    Returns:
        np.ndarray: Array with missing values replaced.
    """
    x_isnan = np.isnan(x)
    if x_isnan.sum() > 0:
        # Perform Stineman interpolation.
        x = calculate_stineman_interpolant(np.where(~x_isnan)[0], x[~x_isnan], np.arange(len(x)))
    if np.any(np.isnan(x)):
        # Fill to avoid NaNs at the beginning and end of array.
        x = fill_missing(fill_missing(x), option="nocb")
    return x
