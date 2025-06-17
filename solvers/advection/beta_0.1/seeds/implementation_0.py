
import numpy as np

def _fractional_periodic_shift(u, shift):
    """
    Periodically shifts the last axis of an array by a (possibly non-integer)
    number of grid points using linear interpolation.

    Parameters
    ----------
    u : np.ndarray
        Input array of shape (..., N).
    shift : float
        Positive shift (to the right) measured in grid points.
    Returns
    -------
    shifted : np.ndarray
        Array of the same shape as `u`, shifted periodically.
    """
    N = u.shape[-1]
    # Bring the shift back to the canonical interval [0, N)
    shift = shift % N

    # Integer and fractional parts of the shift
    k = int(np.floor(shift))
    f = shift - k          # 0 <= f < 1

    if f < 1.0e-12:        # pure integer shift – avoid extra work
        return np.roll(u, k, axis=-1)

    # Values needed for linear interpolation
    u_k   = np.roll(u,  k,     axis=-1)   # u[j - k]
    u_k1  = np.roll(u,  k + 1, axis=-1)   # u[j - k - 1]

    return (1.0 - f) * u_k + f * u_k1


def solver(u0_batch, t_coordinate, beta):
    """Solves the 1-D periodic advection equation u_t + beta * u_x = 0.

    The method is *exact*: the initial profile is merely shifted by
    beta * t for each requested time instant.

    Parameters
    ----------
    u0_batch : np.ndarray
        Initial data with shape [batch_size, N]
    t_coordinate : np.ndarray
        Time stamps (T+1,) beginning with 0.
    beta : float
        Constant advection speed.

    Returns
    -------
    solutions : np.ndarray
        Array of shape [batch_size, T+1, N] containing u(t_i, x_j).
    """
    # --------------------------- sanity checks ---------------------------
    if t_coordinate.ndim != 1:
        raise ValueError("t_coordinate must be one-dimensional")
    if abs(t_coordinate[0]) > 1e-12:
        raise ValueError("t_coordinate[0] must be 0.0")

    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1          # number of future time frames

    # Spatial step Δx assuming domain length L = 1
    dx = 1.0 / N

    # ---------- allocate result tensor and copy initial condition -------
    solutions = np.empty((batch_size, T + 1, N), dtype=u0_batch.dtype)
    solutions[:, 0, :] = u0_batch

    # ----------------------- march through required times ---------------
    for i, t in enumerate(t_coordinate[1:], start=1):
        shift_flow = beta * t / dx     # shift in *grid points*
        if i == 1:                     # tiny diagnostic once per run
            print(f"[solver] beta={beta:.4g}, t={t:.4g}, "
                  f"shift={shift_flow:.4g} grid points")
        solutions[:, i, :] = _fractional_periodic_shift(u0_batch,
                                                        shift_flow)

    return solutions
