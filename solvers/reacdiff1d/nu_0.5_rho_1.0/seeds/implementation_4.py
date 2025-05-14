
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solver(u0_batch, t_coordinate, nu, rho):
    """Solves the 1D reaction diffusion equation for all times in t_coordinate.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N], 
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. 
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        nu (float): The parameter nu in the equation.
        rho (float): The parameter rho in the equation.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    # Extract dimensions
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1

    # Spatial grid
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]

    # Time step
    dt = t_coordinate[1] - t_coordinate[0]

    # Initialize solutions array
    solutions = np.zeros((batch_size, T + 1, N))
    solutions[:, 0, :] = u0_batch

    # Crank-Nicolson coefficients
    alpha = nu * dt / (2 * dx ** 2)
    beta = rho * dt / 2

    # Construct the tridiagonal matrix for Crank-Nicolson
    main_diag = 1 + 2 * alpha
    off_diag = -alpha
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N)).toarray()
    A[0, -1] = off_diag  # Periodic boundary condition
    A[-1, 0] = off_diag  # Periodic boundary condition

    # Time stepping loop
    for t_idx in range(T):
        u_current = solutions[:, t_idx, :]

        for b in range(batch_size):
            # Reaction term
            reaction_term = beta * u_current[b] * (1 - u_current[b])

            # Right-hand side vector
            b_vector = (1 - 2 * alpha) * u_current[b] + alpha * np.roll(u_current[b], -1) + alpha * np.roll(u_current[b], 1) + reaction_term

            # Solve the linear system
            u_next = spsolve(A, b_vector)

            # Clamp the solution to a reasonable range
            u_next = np.clip(u_next, 0, 1)

            # Update the solution
            solutions[b, t_idx + 1, :] = u_next

        # Print intermediate results for debugging
        print(f"Time step {t_idx + 1}/{T}, max(u) = {np.max(solutions[:, t_idx + 1, :])}, min(u) = {np.min(solutions[:, t_idx + 1, :])}")

    return solutions

# Example usage
if __name__ == "__main__":
    # Define initial condition
    batch_size = 2
    N = 100
    u0_batch = np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False)).reshape(1, N).repeat(batch_size, axis=0)

    # Define time coordinates
    t_coordinate = np.linspace(0, 1, 11)

    # Parameters
    nu = 0.5
    rho = 1.0

    # Solve PDE
    solutions = solver(u0_batch, t_coordinate, nu, rho)

    print(solutions.shape)
    print(solutions)
