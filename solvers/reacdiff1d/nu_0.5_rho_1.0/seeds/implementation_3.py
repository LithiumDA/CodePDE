
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solver(u0_batch, t_coordinate, nu, rho):
    """Solves the 1D reaction diffusion equation for all times in t_coordinate using the Crank-Nicolson method.

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

    # Time step size
    dt = t_coordinate[1] - t_coordinate[0]
    
    # Spatial step size
    dx = 1.0 / (N - 1)
    
    # Initialize the solution array
    solutions = np.zeros((batch_size, T + 1, N))
    solutions[:, 0, :] = u0_batch
    
    # Precompute the coefficients
    alpha = nu * dt / (2 * dx**2)
    beta = 0.5 * rho * dt
    
    # Construct the tridiagonal matrix for the Crank-Nicolson method
    main_diag = 1 + 2 * alpha
    off_diag = -alpha
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N)).toarray()
    A[0, -1] = -alpha  # Periodic boundary condition
    A[-1, 0] = -alpha  # Periodic boundary condition
    
    # Time stepping loop
    u_current = u0_batch
    for t_idx in range(T):
        # Right-hand side of the Crank-Nicolson equation
        b = (1 - 2 * alpha) * u_current + alpha * (np.roll(u_current, -1, axis=-1) + np.roll(u_current, 1, axis=-1)) + beta * u_current * (1 - u_current)
        
        # Solve the linear system for each batch
        u_next = np.zeros_like(u_current)
        for i in range(batch_size):
            u_next[i, :] = spsolve(A, b[i, :])
        
        # Store the solution
        solutions[:, t_idx + 1, :] = u_next
        
        # Update the current solution
        u_current = u_next
    
    return solutions

# Example usage
if __name__ == "__main__":
    # Define the initial condition
    batch_size = 100
    N = 1024
    x = np.linspace(0, 1, N)
    u0_batch = np.sin(2 * np.pi * x)[None, :].repeat(batch_size, axis=0)
    
    # Define the time coordinates
    T = 100
    t_coordinate = np.linspace(0, 1, T + 1)
    
    # Parameters
    nu = 0.5
    rho = 1.0
    
    # Solve the PDE
    solutions = solver(u0_batch, t_coordinate, nu, rho)
    
    # Print the shape of the solution
    print("Solution shape:", solutions.shape)
