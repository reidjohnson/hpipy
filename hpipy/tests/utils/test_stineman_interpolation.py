import numpy as np
import pytest

from hpipy.utils.stineman_interpolation import (
    calculate_stineman_interpolant,
    fill_missing,
    interpolate_stineman,
    parabola_slope,
    stineman_slope,
)


def test_stineman_slope_basic() -> None:
    """Test basic functionality of stineman_slope."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 4.0, 9.0])  # y = x^2
    slopes = stineman_slope(x, y)
    assert len(slopes) == len(x)
    assert np.all(np.isfinite(slopes))
    # For y = x^2, slope should be approximately 2x.
    np.testing.assert_allclose(slopes, [2.0, 4.0, 6.0], rtol=0.3)


def test_stineman_slope_two_points() -> None:
    """Test stineman_slope with only two points."""
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])  # linear function
    slopes = stineman_slope(x, y)
    assert len(slopes) == 2
    np.testing.assert_allclose(slopes, [1.0, 1.0])


def test_stineman_slope_with_scaling() -> None:
    """Test stineman_slope with scaling option."""
    x = np.array([1.0, 2.0, 3.0]) * 1000  # large x values
    y = np.array([1.0, 4.0, 9.0]) * 0.001  # small y values
    slopes = stineman_slope(x, y, scale=True)
    assert len(slopes) == len(x)
    assert np.all(np.isfinite(slopes))


def test_parabola_slope_two_points() -> None:
    """Test parabola_slope with only two points."""
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    slopes = parabola_slope(x, y)
    assert len(slopes) == 2
    np.testing.assert_allclose(slopes, [1.0, 1.0])


def test_calculate_stineman_interpolant_basic() -> None:
    """Test basic functionality of calculate_stineman_interpolant."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 4.0, 9.0])  # y = x^2
    xout = np.array([1.5, 2.5])
    yout = calculate_stineman_interpolant(x, y, xout)
    assert len(yout) == len(xout)
    assert np.all(np.isfinite(yout))
    # Output should be close to x^2.
    np.testing.assert_allclose(yout, xout**2, rtol=0.2)


def test_calculate_stineman_interpolant_with_slopes() -> None:
    """Test calculate_stineman_interpolant with provided slopes."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 4.0, 9.0])
    yp = np.array([2.0, 4.0, 6.0])  # exact slopes for y = x^2
    xout = np.array([1.5, 2.5])
    yout = calculate_stineman_interpolant(x, y, xout, yp=yp)
    assert len(yout) == len(xout)
    assert np.all(np.isfinite(yout))


def test_calculate_stineman_interpolant_errors() -> None:
    """Test error handling in calculate_stineman_interpolant."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 4.0, 9.0])
    xout = np.array([1.5, 2.5])

    # Test non-increasing x.
    with pytest.raises(ValueError):
        calculate_stineman_interpolant(np.array([3.0, 2.0, 1.0]), y, xout)

    # Test mismatched lengths.
    with pytest.raises(ValueError):
        calculate_stineman_interpolant(x, y[:-1], xout)

    # Test NaN values.
    with pytest.raises(ValueError):
        calculate_stineman_interpolant(x, np.array([1.0, np.nan, 9.0]), xout)


def test_fill_missing_locf() -> None:
    """Test Last Observation Carried Forward."""
    x = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
    filled = fill_missing(x, option="locf")
    expected = np.array([1.0, 1.0, 1.0, 4.0, 4.0])
    np.testing.assert_array_equal(filled, expected)


def test_fill_missing_nocb() -> None:
    """Test Next Observation Carried Backward."""
    x = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
    filled = fill_missing(x, option="nocb")
    expected = np.array([2.0, 2.0, 4.0, 4.0, np.nan])
    np.testing.assert_array_equal(filled, expected)


def test_interpolate_stineman_basic() -> None:
    """Test basic functionality of interpolate_stineman."""
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    interpolated = interpolate_stineman(x)
    assert len(interpolated) == len(x)
    assert not np.any(np.isnan(interpolated))
    # Values at known points should remain unchanged.
    np.testing.assert_allclose(interpolated[::2], [1.0, 3.0, 5.0])


def test_interpolate_stineman_no_nans() -> None:
    """Test interpolate_stineman with no missing values."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    interpolated = interpolate_stineman(x)
    np.testing.assert_array_equal(interpolated, x)
