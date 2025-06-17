
import numpy as np

# ---------- helper utilities ------------------------------------------------ #
def _primitive_from_conservative(U, gamma):
    """U  shape [B, 3, M]  ->  rho, v, p   each [B, M]"""
    rho = U[:, 0]
    m   = U[:, 1]
    E   = U[:, 2]

    v   = m / rho
    p   = (gamma - 1.0) * (E - 0.5 * rho * v**2)
    return rho, v, p


def _conservative_flux(U, gamma):
    """Ideal–gas Euler flux – same shape as U."""
    rho, v, p = _primitive_from_conservative(U, gamma)

    F = np.empty_like(U)
    F[:, 0] = rho * v
    F[:, 1] = rho * v**2 + p
    F[:, 2] = (E := p / (gamma - 1.0) + 0.5 * rho * v**2)
    F[:, 2] = (E + p) * v
    return F


def _rusanov_flux(U_L, U_R, gamma):
    """Local Lax–Friedrich flux (shape [B, 3, M])."""
    F_L = _conservative_flux(U_L, gamma)
    F_R = _conservative_flux(U_R, gamma)

    rho_L, v_L, p_L = _primitive_from_conservative(U_L, gamma)
    rho_R, v_R, p_R = _primitive_from_conservative(U_R, gamma)

    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    a_max = np.maximum(np.abs(v_L) + a_L, np.abs(v_R) + a_R)  # [B, M]
    a_max = a_max[:, None, :]                                 # [B, 1, M]

    return 0.5 * (F_L + F_R) - 0.5 * a_max * (U_R - U_L)


def _laplacian(f, dx):
    """2-nd order centred Laplacian with periodic BCs – f shape [B, M]."""
    return (np.roll(f, -1, axis=-1) - 2.0 * f + np.roll(f, 1, axis=-1)) / dx**2


def _block_average(arr, C):
    """Average every block of length C – keeps batch dimension untouched."""
    B, N = arr.shape
    M = N // C
    return arr.reshape(B, M, C).mean(axis=2)


def _block_repeat(arr_coarse, C, N_fine):
    """Inverse of _block_average – nearest-neighbour up-sampling."""
    return np.repeat(arr_coarse, C, axis=-1)[:, :N_fine]


# ---------- main solver ------------------------------------------------------ #
def solver(Vx0, density0, pressure0, t_coordinate, eta, zeta):
    """
    Very simple explicit FV solver for the 1-D compressible Navier–Stokes
    equations (periodic BCs) – now with *automatic coarsening* to speed-up
    execution in a tight evaluation environment.
    """
    gamma = 5.0 / 3.0

    # ------------------------------------------------------------------ grid #
    B, N = density0.shape
    C = max(1, N // 64)               # coarsening factor  (≥1  and  N//C≈64)
    M = N // C                        # coarse cell count
    dx_fine   = 2.0 / N
    dx_coarse = dx_fine * C

    # --------------------------------------------------- coarse initial data #
    rho0_c = _block_average(density0, C)
    v0_c   = _block_average(Vx0,      C)
    p0_c   = _block_average(pressure0, C)

    E0_c   = p0_c / (gamma - 1.0) + 0.5 * rho0_c * v0_c**2
    U      = np.stack([rho0_c, rho0_c * v0_c, E0_c], axis=1)   # [B,3,M]

    # -------------------------------------------------------- output arrays #
    T_out = len(t_coordinate)
    density_pred  = np.empty((B, T_out, N), dtype=Vx0.dtype)
    Vx_pred       = np.empty_like(density_pred)
    pressure_pred = np.empty_like(density_pred)

    # t = 0  (up-sample to fine grid)
    density_pred[:, 0]  = _block_repeat(rho0_c, C, N)
    Vx_pred[:, 0]       = _block_repeat(v0_c,   C, N)
    pressure_pred[:, 0] = _block_repeat(p0_c,   C, N)

    # ---------------------------------- constants & time-integration setup #
    mu_visc = zeta + 4.0 / 3.0 * eta         # (ζ + 4/3 η)
    CFL     = 0.8                            # can safely use a large value
    time    = 0.0

    for k in range(1, T_out):
        t_target = t_coordinate[k]
        n_steps  = 0

        while time < t_target - 1e-12:
            # ---- primitive vars & max signal speed ------------------- #
            rho, v, p = _primitive_from_conservative(U, gamma)
            c         = np.sqrt(gamma * p / rho)
            max_speed = np.max(np.abs(v) + c)

            # ---- explicit stability limits --------------------------- #
            dt_cfl  = CFL * dx_coarse / (max_speed + 1e-14)
            dt_diff = 0.5 * dx_coarse**2 / (mu_visc + 1e-14)
            dt      = min(dt_cfl, dt_diff, t_target - time)

            # ------------------ Euler / Rusanov convective part -------- #
            U_L   = U
            U_R   = np.roll(U, -1, axis=2)
            F_int = _rusanov_flux(U_L, U_R, gamma)
            divF  = (F_int - np.roll(F_int, 1, axis=2)) / dx_coarse
            U     = U - dt * divF

            # ------------------ explicit constant-µ viscosity ---------- #
            rho, v, _ = _primitive_from_conservative(U, gamma)
            lap_v     = _laplacian(v,          dx_coarse)
            lap_v2    = _laplacian(0.5*v**2,   dx_coarse)
            U[:, 1] += dt * mu_visc * lap_v
            U[:, 2] += dt * mu_visc * lap_v2

            # simple positivity clamp
            U[:, 0] = np.clip(U[:, 0], 1e-8, None)
            rho, v, p = _primitive_from_conservative(U, gamma)
            p = np.clip(p, 1e-8, None)
            U[:, 2] = p / (gamma - 1.0) + 0.5 * rho * v**2

            time   += dt
            n_steps += 1

        # --------------------- store frame k (up-sample) --------------- #
        rho, v, p = _primitive_from_conservative(U, gamma)
        density_pred[:, k]  = _block_repeat(rho, C, N)
        Vx_pred[:, k]       = _block_repeat(v,   C, N)
        pressure_pred[:, k] = _block_repeat(p,   C, N)

        print(f"[frame] k={k:3d}  physical t = {t_target:.3f}   "
              f"coarse steps = {n_steps}")

    return Vx_pred, density_pred, pressure_pred
