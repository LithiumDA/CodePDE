
import numpy as np


def _primitive_from_conservative(rho, mom, E, gamma):
    """
    Converts conservative to primitive variables.
    Returns v, p and sound speed c.
    """
    v = mom / rho
    kinetic = 0.5 * rho * v ** 2
    p = (gamma - 1.0) * (E - kinetic)
    # avoid negative pressure due to numerical noise
    p = np.maximum(p, 1.0e-8)
    c = np.sqrt(gamma * p / rho)
    return v, p, c


def _flux(rho, mom, E, p):
    """
    Euler flux for 1-D compressible flow.
    """
    v = mom / rho
    F_rho = mom
    F_mom = mom * v + p
    F_E = (E + p) * v
    return np.stack([F_rho, F_mom, F_E], axis=0)  # shape [3, batch, N]


def solver(Vx0, density0, pressure0, t_coordinate, eta, zeta):
    """Solves 1-D compressible Navier–Stokes on a periodic domain [-1,1].

    Parameters
    ----------
    Vx0        : ndarray, shape [B, N]   – initial velocity
    density0   : ndarray, shape [B, N]   – initial density
    pressure0  : ndarray, shape [B, N]   – initial pressure
    t_coordinate : ndarray, shape [T+1]  – output times, first one must be 0
    eta, zeta  : floats                  – (constant) viscosities

    Returns
    -------
    Vx_pred, density_pred, pressure_pred : ndarrays with shapes
        [B, T+1, N]
    """
    gamma = 5.0 / 3.0
    cfl = 0.5

    batch_size, N = Vx0.shape
    dx = 2.0 / N  # domain is [-1,1]
    mu_tot = zeta + 4.0 * eta / 3.0

    # Conservative variables U = [rho, mom, E]
    rho = density0.copy()
    mom = density0 * Vx0
    E = pressure0 / (gamma - 1.0) + 0.5 * density0 * Vx0 ** 2

    # Allocate output arrays
    num_frames = len(t_coordinate)
    Vx_pred = np.zeros((batch_size, num_frames, N), dtype=Vx0.dtype)
    density_pred = np.zeros_like(Vx_pred)
    pressure_pred = np.zeros_like(Vx_pred)

    # Store initial frame
    Vx_pred[:, 0] = Vx0
    density_pred[:, 0] = density0
    pressure_pred[:, 0] = pressure0

    t_now = 0.0
    frame_idx = 1  # next frame to record

    # Helper lambdas for periodic shift
    roll_plus = lambda a: np.roll(a, -1, axis=-1)
    roll_minus = lambda a: np.roll(a, 1, axis=-1)

    while frame_idx < num_frames:
        t_target = float(t_coordinate[frame_idx])

        # ===================================================================
        # inner sub-cycling loop
        # ===================================================================
        while t_now + 1.0e-12 < t_target:
            # ---------- primitives & wave speed for current state ----------
            v, p, c = _primitive_from_conservative(rho, mom, E, gamma)

            # time-step restriction
            max_speed = np.max(np.abs(v) + c)  # over batches & grid
            dt_cfl = cfl * dx / max_speed if max_speed > 0 else 1e-6

            # viscous restriction: dt <= dx^2 / (2 ν)
            nu = mu_tot / rho  # kinematic viscosity
            dt_visc = 0.5 * dx ** 2 / np.max(nu)

            dt = min(dt_cfl, dt_visc, t_target - t_now)

            # ---------- Euler (hyperbolic) update via Lax-Friedrichs ----------
            # neighbour states (periodic)
            rho_r = roll_plus(rho)
            rho_l = roll_minus(rho)

            mom_r = roll_plus(mom)
            mom_l = roll_minus(mom)

            E_r = roll_plus(E)
            E_l = roll_minus(E)

            # primitive of neighbours for flux
            _, p_r, _ = _primitive_from_conservative(rho_r, mom_r, E_r, gamma)
            _, p_l, _ = _primitive_from_conservative(rho_l, mom_l, E_l, gamma)

            F_r = _flux(rho_r, mom_r, E_r, p_r)
            F_l = _flux(rho_l, mom_l, E_l, p_l)

            U_half = 0.5 * (np.stack([rho_r, mom_r, E_r], axis=0) +
                            np.stack([rho_l, mom_l, E_l], axis=0)) \
                     - dt / (2.0 * dx) * (F_r - F_l)

            rho_new = U_half[0]
            mom_new = U_half[1]
            E_new = U_half[2]

            # ---------- viscous source term (explicit) ----------
            # central differences for derivatives
            v_center = mom_new / rho_new

            dv_dx = (roll_plus(v_center) - roll_minus(v_center)) / (2.0 * dx)
            d2v_dx2 = (roll_plus(v_center) - 2.0 * v_center +
                       roll_minus(v_center)) / dx ** 2

            # momentum source
            S_mom = mu_tot * d2v_dx2

            # energy source
            sigma = mu_tot * dv_dx
            v_sigma = v_center * sigma
            dv_sigma_dx = (roll_plus(v_sigma) - roll_minus(v_sigma)) / (2.0 * dx)
            S_E = dv_sigma_dx

            # apply explicit Euler for source
            mom_new = mom_new + dt * S_mom
            E_new = E_new + dt * S_E

            # ensure positivity (numerical safety)
            rho_new = np.maximum(rho_new, 1e-8)

            # next state becomes current
            rho, mom, E = rho_new, mom_new, E_new
            t_now += dt
        # ===================================================================

        # Record the state exactly at t_target
        v_rec, p_rec, _ = _primitive_from_conservative(rho, mom, E, gamma)

        Vx_pred[:, frame_idx] = v_rec
        density_pred[:, frame_idx] = rho
        pressure_pred[:, frame_idx] = p_rec

        print(f"[solver] recorded frame {frame_idx}/{num_frames-1} at t = {t_now:.4e}")
        frame_idx += 1

    return Vx_pred, density_pred, pressure_pred
