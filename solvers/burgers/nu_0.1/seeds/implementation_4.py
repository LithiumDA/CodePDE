
import numpy as np
import torch

def compute_spatial_derivatives(u, dx):
    """Compute the spatial derivatives using central differences.
    
    Args:
        u (torch.Tensor): Solution at current time step of shape [batch_size, N].
        dx (float): Spatial step size.
        
    Returns:
        du_dx (torch.Tensor): First spatial derivative of shape [batch_size, N].
        d2u_dx2 (torch.Tensor): Second spatial derivative of shape [batch_size, N].
    """
    # Periodic boundary conditions
    u_padded = torch.cat([u[:, -1:], u, u[:, :1]], dim=1)
    
    # Central difference for first derivative
    du_dx = (u_padded[:, 2:] - u_padded[:, :-2]) / (2 * dx)
    
    # Central difference for second derivative
    d2u_dx2 = (u_padded[:, 2:] - 2 * u + u_padded[:, :-2]) / (dx ** 2)
    
    return du_dx, d2u_dx2

def rk4_step(u, dt, dx, nu):
    """Perform one Runge-Kutta 4th order step for the Burgers' equation.
    
    Args:
        u (torch.Tensor): Solution at current time step of shape [batch_size, N].
        dt (float): Time step size.
        dx (float): Spatial step size.
        nu (float): Viscosity coefficient.
        
    Returns:
        u_next (torch.Tensor): Solution at the next time step of shape [batch_size, N].
    """
    k1 = -dt * (u * compute_spatial_derivatives(u, dx)[0] - nu * compute_spatial_derivatives(u, dx)[1])
    k2 = -dt * ((u + 0.5 * k1) * compute_spatial_derivatives(u + 0.5 * k1, dx)[0] - nu * compute_spatial_derivatives(u + 0.5 * k1, dx)[1])
    k3 = -dt * ((u + 0.5 * k2) * compute_spatial_derivatives(u + 0.5 * k2, dx)[0] - nu * compute_spatial_derivatives(u + 0.5 * k2, dx)[1])
    k4 = -dt * ((u + k3) * compute_spatial_derivatives(u + k3, dx)[0] - nu * compute_spatial_derivatives(u + k3, dx)[1])
    
    u_next = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return u_next

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
    # Convert inputs to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u0_batch = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    t_coordinate = torch.tensor(t_coordinate, dtype=torch.float32, device=device)
    
    # Number of spatial points and time steps
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1
    
    # Spatial and time step sizes
    dx = 1.0 / (N - 1)
    dt = t_coordinate[1] - t_coordinate[0]
    
    # Ensure stability by using an extremely small time step size
    cfl_number = 0.01  # Extremely small CFL number for better stability
    max_u = torch.max(torch.abs(u0_batch))
    dt_stable = cfl_number * dx / max_u
    dt = min(dt, dt_stable)
    num_internal_steps = int((t_coordinate[1] - t_coordinate[0]) / dt)
    
    # Initialize solution array
    solutions = torch.zeros((batch_size, T + 1, N), dtype=torch.float32, device=device)
    solutions[:, 0, :] = u0_batch
    
    # Time-stepping loop
    u = u0_batch
    for i in range(T):
        for _ in range(num_internal_steps):
            u = rk4_step(u, dt, dx, nu)
            if torch.isnan(u).any() or torch.isinf(u).any():
                print(f"NaN or inf detected at time step {i}, internal step {_}")
                break
            # Clip the solution to a reasonable range to avoid instability
            u = torch.clamp(u, -10, 10)
        solutions[:, i + 1, :] = u
    
    # Convert back to numpy array
    solutions = solutions.cpu().numpy()
    
    return solutions
