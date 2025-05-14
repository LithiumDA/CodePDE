
import numpy as np

def solver(u0_batch, t_coordinate, beta):
    """Solves the Advection equation for all times in t_coordinate.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N], 
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. 
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        beta (float): Constant advection speed.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1

    solutions = np.zeros((batch_size, T + 1, N))
    solutions[:, 0, :] = u0_batch

    dx = 1.0 / N
    dt_max = dx / beta  # CFL condition
    print(f"dx = {dx}, dt_max = {dt_max}")

    u = u0_batch.copy()
    t_idx = 1 # Index for storing output

    current_time = 0.0
    for i in range(1, len(t_coordinate)):
        target_time = t_coordinate[i]
        while current_time < target_time:
            dt = min(dt_max, target_time - current_time)
            
            # Upwind scheme with periodic boundary conditions
            u_new = u - beta * dt / dx * (u - np.roll(u, shift=1, axis=1))

            u = u_new.copy()
            current_time += dt
            
        solutions[:, t_idx, :] = u
        t_idx += 1

    return solutions
