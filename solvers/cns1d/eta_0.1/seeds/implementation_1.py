
import numpy as np

# ------------------------------------------------------------------------
#                           helper utilities
# ------------------------------------------------------------------------
_GAMMA = 5.0 / 3.0               # ideal-gas gamma
_CFL   = 0.8                     # convective Courant number

def _primitive(rho, mom, E):
    """conservative -> primitive: returns v, p (array-wise)."""
    v   = mom / rho
    p   = (_GAMMA - 1.0) * (E - 0.5 * rho * v**2)
    return v, np.maximum(p, 1.0e-12)          # positivity safeguard


def _euler_flux(rho, mom, E, p):
    """1-D Euler flux vector (shape like rho)."""
    v = mom / rho
    return (
        mom,                                # F_rho
        mom * v + p,                        # F_mom
        (E + p) * v                         # F_E
    )


def _rusanov_step(rho, mom, E, dx, dt):
    """
    One explicit step of the Euler part using a Rusanov flux.
    All arrays have shape (B, N) and are modified in-place.
    """
    # primitives in left cells
    vL, pL = _primitive(rho, mom, E)
    cL = np.sqrt(_GAMMA * pL / rho)

    # right-shifted (periodic) states
    rhoR = np.roll(rho, -1, axis=-1)
    momR = np.roll(mom, -1, axis=-1)
    ER   = np.roll(E,   -1, axis=-1)
    vR, pR = _primitive(rhoR, momR, ER)
    cR = np.sqrt(_GAMMA * pR / rhoR)

    # Euler fluxes on both sides
    FL1, FL2, FL3 = _euler_flux(rho,  mom,  E,  pL)
    FR1, FR2, FR3 = _euler_flux(rhoR, momR, ER, pR)

    a_face = np.maximum(np.abs(vL) + cL, np.abs(vR) + cR)

    dU1 = rhoR - rho
    dU2 = momR - mom
    dU3 = ER   - E

    F1 = 0.5 * (FL1 + FR1) - 0.5 * a_face * dU1
    F2 = 0.5 * (FL2 + FR2) - 0.5 * a_face * dU2
    F3 = 0.5 * (FL3 + FR3) - 0.5 * a_face * dU3

    # divergence (F_{i+½} – F_{i-½}) / dx
    F1m = np.roll(F1, 1, axis=-1)
    F2m = np.roll(F2, 1, axis=-1)
    F3m = np.roll(F3, 1, axis=-1)

    rho -= dt * (F1 - F1m) / dx
    mom -= dt * (F2 - F2m) / dx
    E   -= dt * (F3 - F3m) / dx


def _viscosity_filter(rho, mom, dt, nu, ksq):
    """
    Implicit (exact) diffusion step in Fourier space.
    Internal energy is kept, only momentum / kinetic energy are diffused.
    """
    v = mom / rho                                    # shape (B,N)
    v_hat = np.fft.rfft(v, axis=-1)
    v_hat *= np.exp(-nu * ksq * dt)[None, :]
    v_new = np.fft.irfft(v_hat, n=v.shape[-1], axis=-1)

    # update momentum and return kinetic energy difference
    mom[:]  = rho * v_new


# ------------------------------------------------------------------------
#                               main solver
# ------------------------------------------------------------------------
def solver(Vx0, density0, pressure0, t_coordinate, eta, zeta):
    """
    Periodic 1-D compressible Navier–Stokes solver:
      – Euler part   : explicit Rusanov (1st order)
      – Viscous part : exact spectral diffusion (unconditionally stable)

    Shapes:
        Vx0, density0, pressure0 : [B, N]
        t_coordinate             : [T+1]  with t₀ = 0
    Returns:
        Vx_pred, density_pred, pressure_pred : [B, T+1, N]
    """
    # ---------- initial conservative variables -------------------------
    rho = density0.copy()
    v   = Vx0.copy()
    p   = pressure0.copy()

    mom = rho * v
    E   = p / (_GAMMA - 1.0) + 0.5 * rho * v**2    # total energy

    batch, N = rho.shape
    dx = 2.0 / N                                   # domain [-1,1]

    # wavenumbers for spectral diffusion
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)     # shape (N//2+1,)
    ksq = k**2
    nu  = zeta + 4.0 * eta / 3.0                   # effective viscosity

    # ---------- allocate outputs --------------------------------------
    T_out = len(t_coordinate)
    density_pred  = np.empty((batch, T_out, N), dtype=rho.dtype)
    Vx_pred       = np.empty_like(density_pred)
    pressure_pred = np.empty_like(density_pred)

    density_pred[:, 0]  = rho
    Vx_pred[:,     0]   = v
    pressure_pred[:, 0] = p

    # ---------- time marching ----------------------------------------
    t_now   = 0.0
    out_idx = 1

    for t_next in t_coordinate[1:]:
        while t_now < t_next - 1e-14:
            # convective CFL
            v_now, p_now = _primitive(rho, mom, E)
            c_now = np.sqrt(_GAMMA * p_now / rho)
            dt = _CFL * dx / np.max(np.abs(v_now) + c_now)

            # limit final sub-step to land exactly on t_next
            if t_now + dt > t_next:
                dt = t_next - t_now

            # ---------- explicit Euler step (inviscid) --------------
            _rusanov_step(rho, mom, E, dx, dt)

            # ---------- implicit viscosity ---------------------------
            eps  = E - 0.5 * mom**2 / rho          # internal energy before filter
            _viscosity_filter(rho, mom, dt, nu, ksq)
            E[:] = eps + 0.5 * mom**2 / rho        # keep ε, update kinetic

            # positivity safety nets
            rho[:] = np.maximum(rho, 1e-10)

            t_now += dt

        # ---------- store snapshot ----------------------------------
        density_pred[:,  out_idx] = rho
        Vx_pred[:,       out_idx] = mom / rho
        pressure_pred[:, out_idx] = (_GAMMA - 1.0) * (
                                        E - 0.5 * mom**2 / rho)

        print(f"[solver] stored frame {out_idx}   (t = {t_next:.3e})")
        out_idx += 1

    return Vx_pred, density_pred, pressure_pred
