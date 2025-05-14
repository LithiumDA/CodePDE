
import torch
import numpy as np

def central_diff(u, dx):
    """Compute the central difference for the spatial derivative."""
    # Using periodic boundary conditions
    return (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2 * dx)

def second_order_diff(u, dx):
    """Compute the second-order central difference for the diffusion term."""
    # Using periodic boundary conditions
    return (torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)) / (dx**2)

def burgers_step(u, dt, dx, nu):
    """Perform a single time step of the Burgers' equation."""
    # Compute the nonlinear term: u * u_x
    u_x = central_diff(u, dx)
    nonlinear_term = -u * u_x
    
    # Compute the diffusion term: nu * u_xx
    u_xx = second_order_diff(u, dx)
    diffusion_term = nu * u_xx
    
    # Update u using the explicit Euler method
    u_new = u + dt * (nonlinear_term + diffusion_term)
    
    return u_new

def solver(u0_batch, t_coordinate, nu):
    """Solves the Burgers' equation for all times in t_coordinate.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N], 
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. 
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        nu (float): Viscosity coefficient.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    # Convert inputs to PyTorch tensors and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u0_batch = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    t_coordinate = torch.tensor(t_coordinate, dtype=torch.float32, device=device)
    
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1
    
    # Spatial grid
    dx = 1.0 / (N - 1)
    
    # Determine internal time step based on CFL condition
    dt_internal = 0.1 * dx**2 / nu  # Conservative choice for stability
    
    # Initialize solution array
    solutions = torch.zeros((batch_size, T+1, N), dtype=torch.float32, device=device)
    solutions[:, 0, :] = u0_batch
    
    # Time stepping
    for i in range(T):
        t_start = t_coordinate[i]
        t_end = t_coordinate[i+1]
        
        u_current = solutions[:, i, :]
        
        # Perform internal time steps
        t = t_start
        while t < t_end:
            dt = min(dt_internal, t_end - t)
            u_current = burgers_step(u_current, dt, dx, nu)
            t += dt
        
        # Store the solution at the next time step
        solutions[:, i+1, :] = u_current
    
    # Convert back to numpy array for compatibility
    return solutions.cpu().numpy()
