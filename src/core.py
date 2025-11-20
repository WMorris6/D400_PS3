import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp1d


def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)


def plot_with_parameters(a, b, c):
    # Define time range
    t = np.linspace(0, 1, 100)

    # Calculate f(t) for these parameters
    f_values = f(t, a, b, c)

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(t, f_values, label=r"$f(t; a, b, c)$", color="blue")
    plt.xlabel("Time $t$")
    plt.ylabel(r"$f(t)$")
    plt.title(r"Time Series $f(t; a, b, c)$")
    plt.legend()
    plt.grid(True)
    plt.show()


def time_interpolator(a, b, c, grid_n=100):
    """Build an interpolator over t for fixed a, b, c."""
    t = np.linspace(0, 1, grid_n)
    y = f(t, a, b, c)
    # linear interpolation is enough for Q1
    return interp1d(t, y, kind="linear"), t, y


def ratio_interpolated_to_true(interp_fn, t_values, a, b, c, zero_tol=1e-8):
    """Return ratio of interpolated to true values on a grid."""
    y_true = f(t_values, a, b, c)
    y_interp = interp_fn(t_values)
    safe_mask = np.abs(y_true) > zero_tol
    ratio = np.full_like(y_true, np.nan, dtype=float)
    ratio[safe_mask] = y_interp[safe_mask] / y_true[safe_mask]
    return ratio, y_interp, y_true


def plot_ratio(t_values, ratio, ax=None):
    """Plot ratio of interpolated to true values on the provided grid."""
    ax = ax or plt.gca()
    ax.plot(t_values, ratio, label="Interpolated / True")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Perfect agreement")
    ax.set_xlabel("t")
    ax.set_ylabel("Ratio")
    ax.set_title("Ratio of interpolated to true values")
    ax.grid(True)
    ax.legend()
    return ax


def generate_a_sweep_dataframe(b, c, t_grid=None, num_samples=10):
    """Return DataFrame with columns indexed by linearly-spaced values of a in [0, 1]."""
    t_grid = t_grid if t_grid is not None else np.linspace(0, 1, 100)
    a_values = np.linspace(0, 1, num_samples)
    # stack each sample as a column for convenient plotting later
    data = {f"a={a_val:.2f}": f(t_grid, a_val, b, c) for a_val in a_values}
    df = pd.DataFrame(data, index=t_grid)
    df.index.name = "t"
    return df, a_values, t_grid


def build_at_interpolator(b, c, a_grid=None, t_grid=None):
    """Create a 2D interpolator over (a, t)."""
    a_grid = a_grid if a_grid is not None else np.linspace(0, 1, 10)
    t_grid = t_grid if t_grid is not None else np.linspace(0, 1, 100)
    # shape (len(a_grid), len(t_grid))
    values = np.array([f(t_grid, a_val, b, c) for a_val in a_grid])
    interpolator = RegularGridInterpolator(
        (a_grid, t_grid), values, bounds_error=False, fill_value=None
    )
    return interpolator, a_grid, t_grid, values


def eval_at_interpolator(interp_fn, a_value, t_values):
    """Evaluate 2D interpolator for single a over vector of t."""
    points = np.column_stack((np.full_like(t_values, a_value, dtype=float), t_values))
    return interp_fn(points)


def ratio_at_interpolator(interp_fn, a_value, t_values, b, c, zero_tol=1e-8):
    """Compute ratio between 2D interpolator output and true function."""
    y_true = f(t_values, a_value, b, c)
    y_interp = eval_at_interpolator(interp_fn, a_value, t_values)
    safe_mask = np.abs(y_true) > zero_tol
    ratio = np.full_like(y_true, np.nan, dtype=float)
    ratio[safe_mask] = y_interp[safe_mask] / y_true[safe_mask]
    return ratio, y_interp, y_true
