from __future__ import annotations

import logging
import os
import random
import sys
import time
from functools import partial
from math import ceil, exp, log
import math as mt
import numpy as np
import hydra
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d
from jax import device_put, jit, lax, nn, random, scipy, vmap
from omegaconf import DictConfig

def interpolate_solution(u_fine, x_fine, t_fine, x_coarse, t_coarse):
    """
    Interpolates the fine solution onto the coarse grid in both space and time.
    """
    # Interpolate in space
    space_interp_func = interp1d(x_fine, u_fine, axis=2, kind='linear', fill_value="extrapolate")
    # finding the values of the u_fine function over the grid points of x
    u_fine_interp_space = space_interp_func(x_coarse) 

    # Interpolate in time
    time_interp_func = interp1d(t_fine, u_fine_interp_space, axis=1, kind='linear', fill_value="extrapolate")
    # finding the values of the u_fine_interp_sapce function over the grid points of time.
    u_fine_interp = time_interp_func(t_coarse)

    return u_fine_interp

# def compute_error(u_coarse, u_fine, x_coarse, x_fine, t_coarse, t_fine):
def compute_error(coarse_tuple, fine_tuple):
    """
    Computes the error between coarse and fine grid solutions by interpolating in both space and time.
    """
    u_coarse, x_coarse, t_coarse = coarse_tuple
    u_fine, x_fine, t_fine = fine_tuple
    u_fine_interp = interpolate_solution(u_fine, x_fine, t_fine, x_coarse, t_coarse)

    # Compute L2 norm error
    print(u_coarse.shape)
    print(u_fine_interp.shape)
    error = np.linalg.norm(u_coarse - u_fine_interp) / np.sqrt(u_coarse.size)
    return error


# import all the functions from cns_utils.py
@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def init_multi_HD(
    xc,
    yc,
    zc,
    numbers=10000,
    k_tot=10,
    init_key=2022,
    num_choise_k=2,
    if_renorm=False,
    umax=1.0e4,
    umin=1.0e-8,
):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """

    def _pass(carry):
        return carry

    def select_A(carry):
        def _func(carry):
            return jnp.abs(carry)

        cond, value = carry
        value = lax.cond(cond == 1, _func, _pass, value)
        return cond, value

    def select_W(carry):
        def _window(carry):
            xx, val, xL, xR, trns = carry
            val = 0.5 * (jnp.tanh((xx - xL) / trns) - jnp.tanh((xx - xR) / trns))
            return xx, val, xL, xR, trns

        cond, value, xx, xL, xR, trns = carry

        carry = xx, value, xL, xR, trns
        xx, value, xL, xR, trns = lax.cond(cond == 1, _window, _pass, carry)
        return cond, value, xx, xL, xR, trns

    def renormalize(carry):
        def _norm(carry):
            u, key = carry
            u -= jnp.min(u, axis=1, keepdims=True)  # positive value
            u /= jnp.max(u, axis=1, keepdims=True)  # normalize

            key, subkey = random.split(key)
            m_val = random.uniform(
                key, shape=[numbers], minval=mt.log(umin), maxval=mt.log(umax)
            )
            m_val = jnp.exp(m_val)
            key, subkey = random.split(key)
            b_val = random.uniform(
                key, shape=[numbers], minval=mt.log(umin), maxval=mt.log(umax)
            )
            b_val = jnp.exp(b_val)
            return u * m_val[:, None] + b_val[:, None], key

        cond, u, key = carry
        carry = u, key
        u, key = lax.cond(cond is True, _norm, _pass, carry)
        return cond, u, key

    assert numbers % jax.device_count() == 0, "numbers should be : GPUs x integer!!"

    key = random.PRNGKey(init_key)

    selected = random.randint(
        key, shape=[numbers, num_choise_k], minval=0, maxval=k_tot
    )
    selected = nn.one_hot(selected, k_tot, dtype=int).sum(axis=1)
    kk = jnp.pi * 2.0 * jnp.arange(1, k_tot + 1) * selected / (xc[-1] - xc[0])
    amp = random.uniform(key, shape=[numbers, k_tot, 1])
    key, subkey = random.split(key)

    phs = 2.0 * jnp.pi * random.uniform(key, shape=[numbers, k_tot, 1])
    _u = amp * jnp.sin(kk[:, :, jnp.newaxis] * xc[jnp.newaxis, jnp.newaxis, :] + phs)
    _u = jnp.sum(_u, axis=1)

    # perform absolute value function
    cond = random.choice(key, 2, p=jnp.array([0.9, 0.1]), shape=([numbers]))
    carry = (cond, _u)

    cond, _u = vmap(select_A, 0, 0)(carry)
    sgn = random.choice(key, a=jnp.array([1, -1]), shape=([numbers, 1]))
    _u *= sgn  # random flip of signature

    # perform window function
    key, subkey = random.split(key)
    cond = random.choice(key, 2, p=jnp.array([0.5, 0.5]), shape=([numbers]))
    _xc = jnp.repeat(xc[None, :], numbers, axis=0)
    mask = jnp.ones_like(_xc)
    xL = random.uniform(key, shape=([numbers]), minval=0.1, maxval=0.45)
    xR = random.uniform(key, shape=([numbers]), minval=0.55, maxval=0.9)
    trns = 0.01 * jnp.ones_like(cond)
    carry = cond, mask, _xc, xL, xR, trns
    cond, mask, _xc, xL, xR, trns = vmap(select_W, 0, 0)(carry)

    _u *= mask

    carry = if_renorm, _u, key
    _, _u, _ = renormalize(carry)  # renormalize value between a given values

    return _u[..., None, None]


def VLlimiter(a, b, c, alpha=2.0):
    return (
        jnp.sign(c)
        * (0.5 + 0.5 * jnp.sign(a * b))
        * jnp.minimum(alpha * jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))
    )

def limiting_HD(u, if_second_order):
    _, nx, _, _ = u.shape
    uL, uR = u, u
    nx -= 4

    du_L = u[:, 1 : nx + 3, :, :] - u[:, 0 : nx + 2, :, :]
    du_R = u[:, 2 : nx + 4, :, :] - u[:, 1 : nx + 3, :, :]
    du_M = (u[:, 2 : nx + 4, :, :] - u[:, 0 : nx + 2, :, :]) * 0.5
    gradu = VLlimiter(du_L, du_R, du_M) * if_second_order
    # -1:Ncell
    uL = uL.at[:, 1 : nx + 3, :, :].set(
        u[:, 1 : nx + 3, :, :] - 0.5 * gradu
    )  # left of cell
    uR = uR.at[:, 1 : nx + 3, :, :].set(
        u[:, 1 : nx + 3, :, :] + 0.5 * gradu
    )  # right of cell

    uL = jnp.where(uL[0] > 0.0, uL, u)
    uL = jnp.where(uL[4] > 0.0, uL, u)
    uR = jnp.where(uR[0] > 0.0, uR, u)
    uR = jnp.where(uR[4] > 0.0, uR, u)

    return uL, uR

def Courant_HD(u, dx, dy, dz, gamma):
    cs = jnp.sqrt(gamma * u[4] / u[0])  # sound velocity
    stability_adv_x = dx / (jnp.max(cs + jnp.abs(u[1])) + 1.0e-8)
    stability_adv_y = dy / (jnp.max(cs + jnp.abs(u[2])) + 1.0e-8)
    stability_adv_z = dz / (jnp.max(cs + jnp.abs(u[3])) + 1.0e-8)
    return jnp.min(jnp.array([stability_adv_x, stability_adv_y, stability_adv_z]))


def Courant_vis_HD(dx, dy, dz, eta, zeta):
    # visc = jnp.max(jnp.array([eta, zeta]))
    visc = 4.0 / 3.0 * eta + zeta  # maximum
    stability_dif_x = 0.5 * dx**2 / (visc + 1.0e-8)
    stability_dif_y = 0.5 * dy**2 / (visc + 1.0e-8)
    stability_dif_z = 0.5 * dz**2 / (visc + 1.0e-8)
    return jnp.min(jnp.array([stability_dif_x, stability_dif_y, stability_dif_z]))

def bc_HD(u, mode):
    _, Nx, Ny, Nz = u.shape
    Nx -= 2
    Ny -= 2
    Nz -= 2
    if mode == "periodic":  # periodic boundary condition
        # left hand side
        u = u.at[:, 0:2, 2:-2, 2:-2].set(u[:, Nx - 2 : Nx, 2:-2, 2:-2])  # x
        u = u.at[:, 2:-2, 0:2, 2:-2].set(u[:, 2:-2, Ny - 2 : Ny, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 0:2].set(u[:, 2:-2, 2:-2, Nz - 2 : Nz])  # z
        u = u.at[:, Nx : Nx + 2, 2:-2, 2:-2].set(u[:, 2:4, 2:-2, 2:-2])
        u = u.at[:, 2:-2, Ny : Ny + 2, 2:-2].set(u[:, 2:-2, 2:4, 2:-2])
        u = u.at[:, 2:-2, 2:-2, Nz : Nz + 2].set(u[:, 2:-2, 2:-2, 2:4])

        # u = u.loc[:, 2:-2, 0:2, 2:-2].set(u[:, 2:-2, Ny - 2 : Ny, 2:-2])  # y
        # u = u.loc[:, 2:-2, 2:-2, 0:2].set(u[:, 2:-2, 2:-2, Nz - 2 : Nz])  # z
        # # right hand side
        # u = u.loc[:, Nx : Nx + 2, 2:-2, 2:-2].set(u[:, 2:4, 2:-2, 2:-2])
        # u = u.loc[:, 2:-2, Ny : Ny + 2, 2:-2].set(u[:, 2:-2, 2:4, 2:-2])
        # u = u.loc[:, 2:-2, 2:-2, Nz : Nz + 2].set(u[:, 2:-2, 2:-2, 2:4])
    elif mode == "trans":  # periodic boundary condition
        # left hand side
        u = u.loc[:, 0, 2:-2, 2:-2].set(u[:, 3, 2:-2, 2:-2])  # x
        u = u.loc[:, 2:-2, 0, 2:-2].set(u[:, 2:-2, 3, 2:-2])  # y
        u = u.loc[:, 2:-2, 2:-2, 0].set(u[:, 2:-2, 2:-2, 3])  # z
        u = u.loc[:, 1, 2:-2, 2:-2].set(u[:, 2, 2:-2, 2:-2])  # x
        u = u.loc[:, 2:-2, 1, 2:-2].set(u[:, 2:-2, 2, 2:-2])  # y
        u = u.loc[:, 2:-2, 2:-2, 1].set(u[:, 2:-2, 2:-2, 2])  # z
        # right hand side
        u = u.loc[:, -2, 2:-2, 2:-2].set(u[:, -3, 2:-2, 2:-2])
        u = u.loc[:, 2:-2, -2, 2:-2].set(u[:, 2:-2, -3, 2:-2])
        u = u.loc[:, 2:-2, 2:-2, -2].set(u[:, 2:-2, 2:-2, -3])
        u = u.loc[:, -1, 2:-2, 2:-2].set(u[:, -4, 2:-2, 2:-2])
        u = u.loc[:, 2:-2, -1, 2:-2].set(u[:, 2:-2, -4, 2:-2])
        u = u.loc[:, 2:-2, 2:-2, -1].set(u[:, 2:-2, 2:-2, -4])
    elif mode == "KHI":  # x: periodic, y, z : trans
        # left hand side
        u = u.loc[:, 0:2, 2:-2, 2:-2].set(u[:, Nx - 2 : Nx, 2:-2, 2:-2])  # x
        u = u.loc[:, 2:-2, 0, 2:-2].set(u[:, 2:-2, 3, 2:-2])  # y
        u = u.loc[:, 2:-2, 2:-2, 0].set(u[:, 2:-2, 2:-2, 3])  # z
        u = u.loc[:, 2:-2, 1, 2:-2].set(u[:, 2:-2, 2, 2:-2])  # y
        u = u.loc[:, 2:-2, 2:-2, 1].set(u[:, 2:-2, 2:-2, 2])  # z
        # right hand side
        u = u.loc[:, Nx : Nx + 2, 2:-2, 2:-2].set(u[:, 2:4, 2:-2, 2:-2])
        u = u.loc[:, 2:-2, -2, 2:-2].set(u[:, 2:-2, -3, 2:-2])
        u = u.loc[:, 2:-2, 2:-2, -2].set(u[:, 2:-2, 2:-2, -3])
        u = u.loc[:, 2:-2, -1, 2:-2].set(u[:, 2:-2, -4, 2:-2])
        u = u.loc[:, 2:-2, 2:-2, -1].set(u[:, 2:-2, 2:-2, -4])
    return u



logger = logging.getLogger(__name__)

# Hydra

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"

sys.path.append("..")


def _pass(carry):
    return carry


# Init arguments with Hydra
def run_step(cfg, nx, dt_save):
    # physical constants
    ny = 1
    nz = 1
    gamma = cfg.args.gamma  # 3D non-relativistic gas
    gammi1 = gamma - 1.0
    gamminv1 = 1.0 / gammi1
    gamgamm1inv = gamma * gamminv1
    gammi1 = gamma - 1.0

    BCs = ["trans", "periodic", "KHI"]  # reflect
    assert cfg.args.bc in BCs, "bc should be in 'trans, reflect, periodic'"

    dx = (cfg.args.xR - cfg.args.xL) / nx
    dx_inv = 1.0 / dx
    dy = (cfg.args.yR - cfg.args.yL) / ny
    dy_inv = 1.0 / dy
    dz = (cfg.args.zR - cfg.args.zL) / nz
    dz_inv = 1.0 / dz

    # cell edge coordinate
    xe = jnp.linspace(cfg.args.xL, cfg.args.xR, nx + 1)
    ye = jnp.linspace(cfg.args.yL, cfg.args.yR, ny + 1)
    ze = jnp.linspace(cfg.args.zL, cfg.args.zR, nz + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx
    yc = ye[:-1] + 0.5 * dy
    zc = ze[:-1] + 0.5 * dz

    show_steps = cfg.args.show_steps
    ini_time = cfg.args.ini_time
    fin_time = cfg.args.fin_time

    # t-coordinate
    it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    tc = jnp.arange(it_tot + 1) * dt_save

    # set viscosity
    if cfg.args.if_rand_param:
        zeta = exp(
            random.uniform(log(0.001), log(10))
        )  # uniform number between 0.01 to 100
        eta = exp(
            random.uniform(log(0.001), log(10))
        )  # uniform number between 0.01 to 100
    else:
        zeta = cfg.args.zeta
        eta = cfg.args.eta
    logger.info(f"zeta: {zeta:>5f}, eta: {eta:>5f}")
    visc = zeta + eta / 3.0

    def evolve(Q):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        dt = 0.0

        tm_ini = time.time()

        DDD = jnp.zeros([it_tot, nx, ny, nz])
        VVx = jnp.zeros([it_tot, nx, ny, nz])
        VVy = jnp.zeros([it_tot, nx, ny, nz])
        VVz = jnp.zeros([it_tot, nx, ny, nz])
        PPP = jnp.zeros([it_tot, nx, ny, nz])
        # initial time-step
        DDD = DDD.at[0].set(Q[0, 2:-2, 2:-2, 2:-2])
        VVx = VVx.at[0].set(Q[1, 2:-2, 2:-2, 2:-2])
        VVy = VVy.at[0].set(Q[2, 2:-2, 2:-2, 2:-2])
        VVz = VVz.at[0].set(Q[3, 2:-2, 2:-2, 2:-2])
        PPP = PPP.at[0].set(Q[4, 2:-2, 2:-2, 2:-2])

        cond_fun = lambda x: x[0] < fin_time

        def _body_fun(carry):
            def _save(_carry):
                Q, tsave, i_save, DDD, VVx, VVy, VVz, PPP = _carry

                DDD = DDD.at[i_save].set(Q[0, 2:-2, 2:-2, 2:-2])
                VVx = VVx.at[i_save].set(Q[1, 2:-2, 2:-2, 2:-2])
                VVy = VVy.at[i_save].set(Q[2, 2:-2, 2:-2, 2:-2])
                VVz = VVz.at[i_save].set(Q[3, 2:-2, 2:-2, 2:-2])
                PPP = PPP.at[i_save].set(Q[4, 2:-2, 2:-2, 2:-2])

                tsave += dt_save
                i_save += 1
                return (Q, tsave, i_save, DDD, VVx, VVy, VVz, PPP)

            t, tsave, steps, i_save, dt, Q, DDD, VVx, VVy, VVz, PPP = carry

            # if save data
            carry = (Q, tsave, i_save, DDD, VVx, VVy, VVz, PPP)
            Q, tsave, i_save, DDD, VVx, VVy, VVz, PPP = lax.cond(
                t >= tsave, _save, _pass, carry
            )

            carry = (Q, t, dt, steps, tsave)
            Q, t, dt, steps, tsave = lax.fori_loop(0, show_steps, simulation_fn, carry)

            return (t, tsave, steps, i_save, dt, Q, DDD, VVx, VVy, VVz, PPP)

        carry = t, tsave, steps, i_save, dt, Q, DDD, VVx, VVy, VVz, PPP
        t, tsave, steps, i_save, dt, Q, DDD, VVx, VVy, VVz, PPP = lax.while_loop(
            cond_fun, _body_fun, carry
        )

        tm_fin = time.time()
        logger.info(f"total elapsed time is {tm_fin - tm_ini} sec")
        DDD = DDD.at[-1].set(Q[0, 2:-2, 2:-2, 2:-2])
        VVx = VVx.at[-1].set(Q[1, 2:-2, 2:-2, 2:-2])
        VVy = VVy.at[-1].set(Q[2, 2:-2, 2:-2, 2:-2])
        VVz = VVz.at[-1].set(Q[3, 2:-2, 2:-2, 2:-2])
        PPP = PPP.at[-1].set(Q[4, 2:-2, 2:-2, 2:-2])
        return t, DDD, VVx, VVy, VVz, PPP

    @jit
    def simulation_fn(i, carry):
        Q, t, dt, steps, tsave = carry
        dt = (
            Courant_HD(Q[:, 2:-2, 2:-2, 2:-2], dx, dy, dz, cfg.args.gamma)
            * cfg.args.CFL
        )
        dt = jnp.min(jnp.array([dt, cfg.args.fin_time - t, tsave - t]))

        def _update(carry):
            Q, dt = carry

            # preditor step for calculating t+dt/2-th time step
            Q_tmp = bc_HD(
                Q, mode=cfg.args.bc
            )  # index 2 for _U is equivalent with index 0 for u
            Q_tmp = update(Q, Q_tmp, dt * 0.5)
            # update using flux at t+dt/2-th time step
            Q_tmp = bc_HD(
                Q_tmp, mode=cfg.args.bc
            )  # index 2 for _U is equivalent with index 0 for u
            Q = update(Q, Q_tmp, dt)


            dt_vis = Courant_vis_HD(dx, dy, dz, eta, zeta) * cfg.args.CFL
            dt_vis = jnp.min(jnp.array([dt_vis, dt]))
            t_vis = 0.0

            carry = Q, dt, dt_vis, t_vis
            Q, dt, dt_vis, t_vis = lax.while_loop(
                lambda x: x[1] - x[3] > 1.0e-8, update_vis, carry
            )
            return Q, dt

        carry = Q, dt
        Q, dt = lax.cond(dt > 1.0e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return Q, t, dt, steps, tsave

    @jit
    def update(Q, Q_tmp, dt):
        # calculate conservative variables
        D0 = Q[0]
        Mx = Q[1] * Q[0]
        My = Q[2] * Q[0]
        Mz = Q[3] * Q[0]
        E0 = Q[4] * gamminv1 + 0.5 * (Mx * Q[1] + My * Q[2] + Mz * Q[3])

        D0 = D0[2:-2, 2:-2, 2:-2]
        Mx = Mx[2:-2, 2:-2, 2:-2]
        My = My[2:-2, 2:-2, 2:-2]
        Mz = Mz[2:-2, 2:-2, 2:-2]
        E0 = E0[2:-2, 2:-2, 2:-2]

        # calculate flux
        fx = flux_x(Q_tmp)
        fy = flux_y(Q_tmp)
        fz = flux_z(Q_tmp)

        # update conservative variables
        dtdx, dtdy, dtdz = dt * dx_inv, dt * dy_inv, dt * dz_inv
        D0 -= (
            dtdx * (fx[0, 1:, 2:-2, 2:-2] - fx[0, :-1, 2:-2, 2:-2])
            + dtdy * (fy[0, 2:-2, 1:, 2:-2] - fy[0, 2:-2, :-1, 2:-2])
            + dtdz * (fz[0, 2:-2, 2:-2, 1:] - fz[0, 2:-2, 2:-2, :-1])
        )

        Mx -= (
            dtdx * (fx[1, 1:, 2:-2, 2:-2] - fx[1, :-1, 2:-2, 2:-2])
            + dtdy * (fy[1, 2:-2, 1:, 2:-2] - fy[1, 2:-2, :-1, 2:-2])
            + dtdz * (fz[1, 2:-2, 2:-2, 1:] - fz[1, 2:-2, 2:-2, :-1])
        )

        My -= (
            dtdx * (fx[2, 1:, 2:-2, 2:-2] - fx[2, :-1, 2:-2, 2:-2])
            + dtdy * (fy[2, 2:-2, 1:, 2:-2] - fy[2, 2:-2, :-1, 2:-2])
            + dtdz * (fz[2, 2:-2, 2:-2, 1:] - fz[2, 2:-2, 2:-2, :-1])
        )

        Mz -= (
            dtdx * (fx[3, 1:, 2:-2, 2:-2] - fx[3, :-1, 2:-2, 2:-2])
            + dtdy * (fy[3, 2:-2, 1:, 2:-2] - fy[3, 2:-2, :-1, 2:-2])
            + dtdz * (fz[3, 2:-2, 2:-2, 1:] - fz[3, 2:-2, 2:-2, :-1])
        )

        E0 -= (
            dtdx * (fx[4, 1:, 2:-2, 2:-2] - fx[4, :-1, 2:-2, 2:-2])
            + dtdy * (fy[4, 2:-2, 1:, 2:-2] - fy[4, 2:-2, :-1, 2:-2])
            + dtdz * (fz[4, 2:-2, 2:-2, 1:] - fz[4, 2:-2, 2:-2, :-1])
        )

        # reverse primitive variables
        Q = Q.at[0, 2:-2, 2:-2, 2:-2].set(D0)  # d
        Q = Q.at[1, 2:-2, 2:-2, 2:-2].set(Mx / D0)  # vx
        Q = Q.at[2, 2:-2, 2:-2, 2:-2].set(My / D0)  # vy
        Q = Q.at[3, 2:-2, 2:-2, 2:-2].set(Mz / D0)  # vz
        Q = Q.at[4, 2:-2, 2:-2, 2:-2].set(
            gammi1 * (E0 - 0.5 * (Mx**2 + My**2 + Mz**2) / D0)
        )  # p
        return Q.at[4].set(jnp.where(Q[4] > 1.0e-8, Q[4], cfg.args.p_floor))

    @jit
    def update_vis(carry):
        def _update_vis_x(carry):
            Q, dt = carry
            # calculate conservative variables
            D0 = Q[0]
            Mx = Q[1] * D0
            My = Q[2] * D0
            Mz = Q[3] * D0
            E0 = Q[4] * gamminv1 + 0.5 * (Mx * Q[1] + My * Q[2] + Mz * Q[3])

            # calculate flux
            dtdx = dt * dx_inv
            # here the viscosity is eta*D0, so that dv/dt = eta*d^2v/dx^2 (not realistic viscosity but fast to calculate)
            Dm = 0.5 * (D0[2:-1, 2:-2, 2:-2] + D0[1:-2, 2:-2, 2:-2])

            fMx = (
                (eta + visc)
                * Dm
                * dx_inv
                * (Q[1, 2:-1, 2:-2, 2:-2] - Q[1, 1:-2, 2:-2, 2:-2])
            )
            fMy = eta * Dm * dx_inv * (Q[2, 2:-1, 2:-2, 2:-2] - Q[2, 1:-2, 2:-2, 2:-2])
            fMz = eta * Dm * dx_inv * (Q[3, 2:-1, 2:-2, 2:-2] - Q[3, 1:-2, 2:-2, 2:-2])
            fE = 0.5 * (eta + visc) * Dm * dx_inv * (
                Q[1, 2:-1, 2:-2, 2:-2] ** 2 - Q[1, 1:-2, 2:-2, 2:-2] ** 2
            ) + 0.5 * eta * Dm * dx_inv * (
                (Q[2, 2:-1, 2:-2, 2:-2] ** 2 - Q[2, 1:-2, 2:-2, 2:-2] ** 2)
                + (Q[3, 2:-1, 2:-2, 2:-2] ** 2 - Q[3, 1:-2, 2:-2, 2:-2] ** 2)
            )

            D0 = D0[2:-2, 2:-2, 2:-2]
            Mx = Mx[2:-2, 2:-2, 2:-2]
            My = My[2:-2, 2:-2, 2:-2]
            Mz = Mz[2:-2, 2:-2, 2:-2]
            E0 = E0[2:-2, 2:-2, 2:-2]

            Mx += dtdx * (fMx[1:, :, :] - fMx[:-1, :, :])
            My += dtdx * (fMy[1:, :, :] - fMy[:-1, :, :])
            Mz += dtdx * (fMz[1:, :, :] - fMz[:-1, :, :])
            E0 += dtdx * (fE[1:, :, :] - fE[:-1, :, :])

            # reverse primitive variables
            Q = Q.at[1, 2:-2, 2:-2, 2:-2].set(Mx / D0)  # vx
            Q = Q.at[2, 2:-2, 2:-2, 2:-2].set(My / D0)  # vy
            Q = Q.at[3, 2:-2, 2:-2, 2:-2].set(Mz / D0)  # vz
            Q = Q.at[4, 2:-2, 2:-2, 2:-2].set(
                gammi1 * (E0 - 0.5 * (Mx**2 + My**2 + Mz**2) / D0)
            )  # p

            return Q, dt

        def _update_vis_y(carry):
            Q, dt = carry
            # calculate conservative variables
            D0 = Q[0]
            Mx = Q[1] * D0
            My = Q[2] * D0
            Mz = Q[3] * D0
            E0 = Q[4] * gamminv1 + 0.5 * (Mx * Q[1] + My * Q[2] + Mz * Q[3])

            # calculate flux
            dtdy = dt * dy_inv
            # here the viscosity is eta*D0, so that dv/dt = eta*d^2v/dx^2 (not realistic viscosity but fast to calculate)
            Dm = 0.5 * (D0[2:-2, 2:-1, 2:-2] + D0[2:-2, 1:-2, 2:-2])

            fMx = eta * Dm * dy_inv * (Q[1, 2:-2, 2:-1, 2:-2] - Q[1, 2:-2, 1:-2, 2:-2])
            fMy = (
                (eta + visc)
                * Dm
                * dy_inv
                * (Q[2, 2:-2, 2:-1, 2:-2] - Q[2, 2:-2, 1:-2, 2:-2])
            )
            fMz = eta * Dm * dy_inv * (Q[3, 2:-2, 2:-1, 2:-2] - Q[3, 2:-2, 1:-2, 2:-2])
            fE = 0.5 * (eta + visc) * Dm * dy_inv * (
                Q[2, 2:-2, 2:-1, 2:-2] ** 2 - Q[2, 2:-2, 1:-2, 2:-2] ** 2
            ) + 0.5 * eta * Dm * dy_inv * (
                (Q[3, 2:-2, 2:-1, 2:-2] ** 2 - Q[3, 2:-2, 1:-2, 2:-2] ** 2)
                + (Q[1, 2:-2, 2:-1, 2:-2] ** 2 - Q[1, 2:-2, 1:-2, 2:-2] ** 2)
            )

            D0 = D0[2:-2, 2:-2, 2:-2]
            Mx = Mx[2:-2, 2:-2, 2:-2]
            My = My[2:-2, 2:-2, 2:-2]
            Mz = Mz[2:-2, 2:-2, 2:-2]
            E0 = E0[2:-2, 2:-2, 2:-2]

            Mx += dtdy * (fMx[:, 1:, :] - fMx[:, :-1, :])
            My += dtdy * (fMy[:, 1:, :] - fMy[:, :-1, :])
            Mz += dtdy * (fMz[:, 1:, :] - fMz[:, :-1, :])
            E0 += dtdy * (fE[:, 1:, :] - fE[:, :-1, :])

            # reverse primitive variables
            Q = Q.at[1, 2:-2, 2:-2, 2:-2].set(Mx / D0)  # vx
            Q = Q.at[2, 2:-2, 2:-2, 2:-2].set(My / D0)  # vy
            Q = Q.at[3, 2:-2, 2:-2, 2:-2].set(Mz / D0)  # vz
            Q = Q.at[4, 2:-2, 2:-2, 2:-2].set(
                gammi1 * (E0 - 0.5 * (Mx**2 + My**2 + Mz**2) / D0)
            )  # p

            return Q, dt

        def _update_vis_z(carry):
            Q, dt = carry
            # calculate conservative variables
            D0 = Q[0]
            Mx = Q[1] * D0
            My = Q[2] * D0
            Mz = Q[3] * D0
            E0 = Q[4] * gamminv1 + 0.5 * (Mx * Q[1] + My * Q[2] + Mz * Q[3])

            # calculate flux
            dtdz = dt * dz_inv
            # here the viscosity is eta*D0, so that dv/dt = eta*d^2v/dx^2 (not realistic viscosity but fast to calculate)
            Dm = 0.5 * (D0[2:-2, 2:-2, 2:-1] + D0[2:-2, 2:-2, 1:-2])

            fMx = eta * Dm * dz_inv * (Q[1, 2:-2, 2:-2, 2:-1] - Q[1, 2:-2, 2:-2, 1:-2])
            fMy = eta * Dm * dz_inv * (Q[2, 2:-2, 2:-2, 2:-1] - Q[2, 2:-2, 2:-2, 1:-2])
            fMz = (
                (eta + visc)
                * Dm
                * dz_inv
                * (Q[3, 2:-2, 2:-2, 2:-1] - Q[3, 2:-2, 2:-2, 1:-2])
            )
            fE = 0.5 * (eta + visc) * Dm * dz_inv * (
                Q[3, 2:-2, 2:-2, 2:-1] ** 2 - Q[3, 2:-2, 2:-2, 1:-2] ** 2
            ) + 0.5 * eta * Dm * dz_inv * (
                (Q[1, 2:-2, 2:-2, 2:-1] ** 2 - Q[1, 2:-2, 2:-2, 1:-2] ** 2)
                + (Q[2, 2:-2, 2:-2, 2:-1] ** 2 - Q[2, 2:-2, 2:-2, 1:-2] ** 2)
            )

            D0 = D0[2:-2, 2:-2, 2:-2]
            Mx = Mx[2:-2, 2:-2, 2:-2]
            My = My[2:-2, 2:-2, 2:-2]
            Mz = Mz[2:-2, 2:-2, 2:-2]
            E0 = E0[2:-2, 2:-2, 2:-2]

            Mx += dtdz * (fMx[:, :, 1:] - fMx[:, :, :-1])
            My += dtdz * (fMy[:, :, 1:] - fMy[:, :, :-1])
            Mz += dtdz * (fMz[:, :, 1:] - fMz[:, :, :-1])
            E0 += dtdz * (fE[:, :, 1:] - fE[:, :, :-1])

            # reverse primitive variables
            Q = Q.at[1, 2:-2, 2:-2, 2:-2].set(Mx / D0)  # vx
            Q = Q.at[2, 2:-2, 2:-2, 2:-2].set(My / D0)  # vy
            Q = Q.at[3, 2:-2, 2:-2, 2:-2].set(Mz / D0)  # vz
            Q = Q.at[4, 2:-2, 2:-2, 2:-2].set(
                gammi1 * (E0 - 0.5 * (Mx**2 + My**2 + Mz**2) / D0)
            )  # p

            return Q, dt

        Q, dt, dt_vis, t_vis = carry
        Q = bc_HD(
            Q, mode=cfg.args.bc
        )  # index 2 for _U is equivalent with index 0 for u
        dt_ev = jnp.min(jnp.array([dt, dt_vis, dt - t_vis]))

        carry = Q, dt_ev
        # directional split
        carry = _update_vis_x(carry)  # x
        carry = _update_vis_y(carry)  # y
        Q, d_ev = _update_vis_z(carry)  # z

        t_vis += dt_ev

        return Q, dt, dt_vis, t_vis

    @jit
    def flux_x(Q):
        QL, QR = limiting_HD(Q, if_second_order=cfg.args.if_second_order)
        # f_Riemann = HLL(QL, QR, direc=0)
        return HLLC(QL, QR, direc=0)

    @jit
    def flux_y(Q):
        _Q = jnp.transpose(Q, (0, 2, 3, 1))  # (y, z, x)
        QL, QR = limiting_HD(_Q, if_second_order=cfg.args.if_second_order)
        # f_Riemann = jnp.transpose(HLL(QL, QR, direc=1), (0, 3, 1, 2))  # (x,y,z) = (Z,X,Y)
        return jnp.transpose(HLLC(QL, QR, direc=1), (0, 3, 1, 2))  # (x,y,z) = (Z,X,Y)

    @jit
    def flux_z(Q):
        _Q = jnp.transpose(Q, (0, 3, 1, 2))  # (z, x, y)
        QL, QR = limiting_HD(_Q, if_second_order=cfg.args.if_second_order)
        # f_Riemann = jnp.transpose(HLL(QL, QR, direc=2), (0, 2, 3, 1))
        return jnp.transpose(HLLC(QL, QR, direc=2), (0, 2, 3, 1))

    @partial(jit, static_argnums=(2,))
    def HLL(QL, QR, direc):
        # direc = 0, 1, 2: (X, Y, Z)
        iX, iY, iZ = direc + 1, (direc + 1) % 3 + 1, (direc + 2) % 3 + 1
        cfL = jnp.sqrt(gamma * QL[4] / QL[0])
        cfR = jnp.sqrt(gamma * QR[4] / QR[0])
        Sfl = jnp.minimum(QL[iX, 2:-1], QR[iX, 1:-2]) - jnp.maximum(
            cfL[2:-1], cfR[1:-2]
        )  # left-going wave
        Sfr = jnp.maximum(QL[iX, 2:-1], QR[iX, 1:-2]) + jnp.maximum(
            cfL[2:-1], cfR[1:-2]
        )  # right-going wave
        dcfi = 1.0 / (Sfr - Sfl + 1.0e-8)

        UL, UR = jnp.zeros_like(QL), jnp.zeros_like(QR)
        UL = UL.at[0].set(QL[0])
        UL = UL.at[iX].set(QL[0] * QL[iX])
        UL = UL.at[iY].set(QL[0] * QL[iY])
        UL = UL.at[iZ].set(QL[0] * QL[iZ])
        UL = UL.at[4].set(
            gamminv1 * QL[4]
            + 0.5 * (UL[iX] * QL[iX] + UL[iY] * QL[iY] + UL[iZ] * QL[iZ])
        )
        UR = UR.at[0].set(QR[0])
        UR = UR.at[iX].set(QR[0] * QR[iX])
        UR = UR.at[iY].set(QR[0] * QR[iY])
        UR = UR.at[iZ].set(QR[0] * QR[iZ])
        UR = UR.at[4].set(
            gamminv1 * QR[4]
            + 0.5 * (UR[iX] * QR[iX] + UR[iY] * QR[iY] + UR[iZ] * QR[iZ])
        )

        fL, fR = jnp.zeros_like(QL), jnp.zeros_like(QR)
        fL = fL.at[0].set(UL[iX])
        fL = fL.at[iX].set(UL[iX] * QL[iX] + QL[4])
        fL = fL.at[iY].set(UL[iX] * QL[iY])
        fL = fL.at[iZ].set(UL[iX] * QL[iZ])
        fL = fL.at[4].set((UL[4] + QL[4]) * QL[iX])
        fR = fR.at[0].set(UR[iX])
        fR = fR.at[iX].set(UR[iX] * QR[iX] + QR[4])
        fR = fR.at[iY].set(UR[iX] * QR[iY])
        fR = fR.at[iZ].set(UR[iX] * QR[iZ])
        fR = fR.at[4].set((UR[4] + QR[4]) * QR[iX])
        # upwind advection scheme
        fHLL = dcfi * (
            Sfr * fR[:, 1:-2]
            - Sfl * fL[:, 2:-1]
            + Sfl * Sfr * (UL[:, 2:-1] - UR[:, 1:-2])
        )

        # L: left of cell = right-going,  R: right of cell: left-going
        f_Riemann = jnp.where(Sfl > 0.0, fR[:, 1:-2], fHLL)
        return jnp.where(Sfr < 0.0, fL[:, 2:-1], f_Riemann)

    @partial(jit, static_argnums=(2,))
    def HLLC(QL, QR, direc):
        """full-Godunov method -- exact shock solution"""

        iX, iY, iZ = direc + 1, (direc + 1) % 3 + 1, (direc + 2) % 3 + 1
        cfL = jnp.sqrt(gamma * QL[4] / QL[0])
        cfR = jnp.sqrt(gamma * QR[4] / QR[0])
        Sfl = jnp.minimum(QL[iX, 2:-1], QR[iX, 1:-2]) - jnp.maximum(
            cfL[2:-1], cfR[1:-2]
        )  # left-going wave
        Sfr = jnp.maximum(QL[iX, 2:-1], QR[iX, 1:-2]) + jnp.maximum(
            cfL[2:-1], cfR[1:-2]
        )  # right-going wave

        UL, UR = jnp.zeros_like(QL), jnp.zeros_like(QR)
        UL = UL.at[0].set(QL[0])
        UL = UL.at[iX].set(QL[0] * QL[iX])
        UL = UL.at[iY].set(QL[0] * QL[iY])
        UL = UL.at[iZ].set(QL[0] * QL[iZ])
        UL = UL.at[4].set(
            gamminv1 * QL[4]
            + 0.5 * (UL[iX] * QL[iX] + UL[iY] * QL[iY] + UL[iZ] * QL[iZ])
        )
        UR = UR.at[0].set(QR[0])
        UR = UR.at[iX].set(QR[0] * QR[iX])
        UR = UR.at[iY].set(QR[0] * QR[iY])
        UR = UR.at[iZ].set(QR[0] * QR[iZ])
        UR = UR.at[4].set(
            gamminv1 * QR[4]
            + 0.5 * (UR[iX] * QR[iX] + UR[iY] * QR[iY] + UR[iZ] * QR[iZ])
        )

        Va = (
            (Sfr - QL[iX, 2:-1]) * UL[iX, 2:-1]
            - (Sfl - QR[iX, 1:-2]) * UR[iX, 1:-2]
            - QL[4, 2:-1]
            + QR[4, 1:-2]
        )
        Va /= (Sfr - QL[iX, 2:-1]) * QL[0, 2:-1] - (Sfl - QR[iX, 1:-2]) * QR[0, 1:-2]
        Pa = QR[4, 1:-2] + QR[0, 1:-2] * (Sfl - QR[iX, 1:-2]) * (Va - QR[iX, 1:-2])

        # shock jump condition
        Dal = QR[0, 1:-2] * (Sfl - QR[iX, 1:-2]) / (Sfl - Va)  # right-hand density
        Dar = QL[0, 2:-1] * (Sfr - QL[iX, 2:-1]) / (Sfr - Va)  # left-hand density

        fL, fR = jnp.zeros_like(QL), jnp.zeros_like(QR)
        fL = fL.at[0].set(UL[iX])
        fL = fL.at[iX].set(UL[iX] * QL[iX] + QL[4])
        fL = fL.at[iY].set(UL[iX] * QL[iY])
        fL = fL.at[iZ].set(UL[iX] * QL[iZ])
        fL = fL.at[4].set((UL[4] + QL[4]) * QL[iX])
        fR = fR.at[0].set(UR[iX])
        fR = fR.at[iX].set(UR[iX] * QR[iX] + QR[4])
        fR = fR.at[iY].set(UR[iX] * QR[iY])
        fR = fR.at[iZ].set(UR[iX] * QR[iZ])
        fR = fR.at[4].set((UR[4] + QR[4]) * QR[iX])
        # upwind advection scheme
        far, fal = jnp.zeros_like(QL[:, 2:-1]), jnp.zeros_like(QR[:, 1:-2])
        far = far.at[0].set(Dar * Va)
        far = far.at[iX].set(Dar * Va**2 + Pa)
        far = far.at[iY].set(Dar * Va * QL[iY, 2:-1])
        far = far.at[iZ].set(Dar * Va * QL[iZ, 2:-1])
        far = far.at[4].set(
            (
                gamgamm1inv * Pa
                + 0.5 * Dar * (Va**2 + QL[iY, 2:-1] ** 2 + QL[iZ, 2:-1] ** 2)
            )
            * Va
        )
        fal = fal.at[0].set(Dal * Va)
        fal = fal.at[iX].set(Dal * Va**2 + Pa)
        fal = fal.at[iY].set(Dal * Va * QR[iY, 1:-2])
        fal = fal.at[iZ].set(Dal * Va * QR[iZ, 1:-2])
        fal = fal.at[4].set(
            (
                gamgamm1inv * Pa
                + 0.5 * Dal * (Va**2 + QR[iY, 1:-2] ** 2 + QR[iZ, 1:-2] ** 2)
            )
            * Va
        )

        f_Riemann = jnp.where(
            Sfl > 0.0, fR[:, 1:-2], fL[:, 2:-1]
        )  # Sf2 > 0 : supersonic
        f_Riemann = jnp.where(
            Sfl * Va < 0.0, fal, f_Riemann
        )  # SL < 0 and Va > 0 : sub-sonic
        return jnp.where(
            Sfr * Va < 0.0, far, f_Riemann
        )  # Va < 0 and SR > 0 : sub-sonic
        # f_Riemann = jnp.where(Sfr < 0., fL[:, 2:-1], f_Riemann) # SR < 0 : supersonic

    Q = jnp.zeros(
        [cfg.args.numbers, 5, nx + 4, ny + 4, nz + 4]
    )

    Q = Q.at[:, 0, 2:-2, 2:-2, 2:-2].set(
        init_multi_HD(
            xc,
            yc,
            zc,
            numbers=cfg.args.numbers,
            k_tot=3,
            init_key=cfg.args.init_key,
            num_choise_k=2,
            umin=1.0e0,
            umax=1.0e1,
            if_renorm=True,
        )
    )

    Q = device_put(Q)  # putting variables in GPU (not necessary??)

    DDDs = []
    VVxs = []
    VVys = []
    VVzs = []
    PPPs = []
    for i in range(Q.shape[0]):
        t, DDD, VVx, VVy, VVz, PPP = evolve(Q[i])
        DDDs.append(jnp.squeeze(DDD))
        VVxs.append(jnp.squeeze(VVx))
        VVys.append(jnp.squeeze(VVy))
        VVzs.append(jnp.squeeze(VVz))
        PPPs.append(jnp.squeeze(PPP))
    
    density = jnp.stack(DDDs)
    ux = jnp.stack(VVxs)
    pressure = jnp.stack(PPPs)

    return ux, density, pressure, xc, tc

    
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    nxs = cfg.args.nx
    dt_saves = cfg.args.dt_save
    outputs = []
    xcs = []
    tcs = []
    for nx, dt_save in zip(nxs, dt_saves):
        print(nx, dt_save)
        u, density, pressure, xc, tc = run_step(cfg, nx, dt_save)
        full = np.stack([u, density, pressure], axis=-1)
        outputs.append(full)
        xcs.append(xc)
        tcs.append(tc[1:])

    # now we try to compute error. 
    errors = []
    for i in range(len(nxs) - 1):
        coarse_tuple = (outputs[i], xcs[i], tcs[i])
        fine_tuple = (outputs[i+1], xcs[i+1], tcs[i+1])
        error = compute_error(
            coarse_tuple, fine_tuple
        )
        errors.append(error)
    breakpoint()

if __name__ == "__main__":
    main()
