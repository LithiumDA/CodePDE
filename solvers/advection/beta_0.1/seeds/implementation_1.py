
import numpy as np

# Optional acceleration with PyTorch if it is installed and a GPU is visible.
try:
    import torch
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:                # pragma: no cover
    _TORCH_AVAILABLE = False


def _select_backend(array):
    """
    Decide whether to run on numpy (CPU) or torch (CPU/GPU) based on the
    type of `array` and on the availability of torch + CUDA.
    """
    if _TORCH_AVAILABLE and isinstance(array, torch.Tensor):
        return "torch"
    return "numpy"


def _rfft(a, backend):
    """Real-to-complex FFT along the last axis for both back-ends."""
    if backend == "torch":
        return torch.fft.rfft(a, dim=-1)
    return np.fft.rfft(a, axis=-1)


def _irfft(a, n, backend, dtype):
    """
    Inverse rFFT that returns a real array of length `n` and is cast back to
    `dtype` for the NumPy backend (torch keeps dtype automatically).
    """
    if backend == "torch":
        return torch.fft.irfft(a, n=n, dim=-1)
    arr = np.fft.irfft(a, n=n, axis=-1)
    return arr.astype(dtype, copy=False)


def solver(u0_batch, t_coordinate, beta):
    """Solves the 1-D periodic advection equation ∂_t u + β ∂_x u = 0.

    Exact spectral shift method:
        u(t,x) = u0(x − β t)   (periodic on [0,1))

    Parameters
    ----------
    u0_batch : np.ndarray | torch.Tensor, shape (B, N)
        Batch of initial conditions.
    t_coordinate : np.ndarray | torch.Tensor, shape (T+1,)
        Time stamps, starting with 0.0.
    beta : float
        Constant advection speed.

    Returns
    -------
    solutions : same backend as `u0_batch`, shape (B, T+1, N)
        Numerical solution for all requested times.
    """
    backend = _select_backend(u0_batch)
    B, N = u0_batch.shape
    T_plus_1 = t_coordinate.shape[0]

    # Convert t_coordinate and wavenumbers to the active backend
    if backend == "torch":
        device = u0_batch.device
        dtype = u0_batch.dtype
        t_arr = torch.as_tensor(t_coordinate, dtype=dtype, device=device)
        k = torch.arange(0, N // 2 + 1, dtype=dtype, device=device)
    else:
        dtype = u0_batch.dtype
        t_arr = np.asarray(t_coordinate, dtype=dtype)
        k = np.arange(0, N // 2 + 1, dtype=dtype)

    # Allocate output container
    if backend == "torch":
        solutions = torch.empty((B, T_plus_1, N), dtype=dtype, device=u0_batch.device)
    else:
        solutions = np.empty((B, T_plus_1, N), dtype=dtype)

    # Store initial condition
    solutions[:, 0, :] = u0_batch

    # Forward FFT of the initial condition
    U0 = _rfft(u0_batch, backend=backend)          # shape (B, N//2+1)

    # Build complex exponent  -i 2π k β t  (vectorised over all times)
    if backend == "torch":
        exponent = -2 * torch.pi * 1j * (beta * t_arr)[:, None] * k[None, :]
        phase = torch.exp(exponent)                # shape (T+1, N//2+1)
    else:
        exponent = -2 * np.pi * 1j * (beta * t_arr)[:, None] * k[None, :]
        phase = np.exp(exponent)                   # shape (T+1, N//2+1)

    # Fill all requested time levels except t=0 (already done)
    for idx in range(1, T_plus_1):
        Ut = U0 * phase[idx]                       # broadcasting over batch
        solutions[:, idx, :] = _irfft(Ut, n=N, backend=backend, dtype=dtype)

    # Concise diagnostic
    print(f"[solver] batch={B}, N={N}, times={T_plus_1-1}, "
          f"beta={beta}, t_min={t_arr[0]:.3g}, t_max={t_arr[-1]:.3g}")

    return solutions
