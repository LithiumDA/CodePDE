
import torch
import numpy as np

def central_diff(u, dx):
    """Compute the central difference of u with respect to x.
    
    Args:
        u (torch.Tensor): Input tensor of shape [batch_size, N].
        dx (float): Spatial step size.
    
    Returns:
        torch.Tensor: Central difference of u, shape [batch_size, N].
    """
    # Roll the tensor to compute the central difference
    u_plus = torch.roll(u, shifts=-1, dims=-1)
    u_minus = torch.roll(u, shifts=1, dims=-1)
    return (u_plus - u_minus) / (2 * dx)

def second_order_central_diff(u, dx):
    """Compute the second-order central difference of u with respect to x.
    
    Args:
        u (torch.Tensor): Input tensor of shape [batch_size, N].
        dx (float): Spatial step size.
    
    Returns:
        torch.Tensor: Second-order central difference of u, shape [batch_size, N].
    """
    # Roll the tensor to compute the second-order central difference
    u_plus = torch.roll(u, shifts=-1, dims=-1)
    u_minus = torch.roll(u, shifts=1, dims=-1)
    return (u_plus - 2 * u + u_minus) / (dx ** 2)

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
    
    # Spatial grid setup
    x = torch.linspace(0, 1, N, device=device)
    dx = x[1] - x[0]
    
    # Time step setup
    # Calculate the maximum allowed time step based on CFL condition
    max_u = torch.max(torch.abs(u0_batch))
    dt_cfl_convective = dx / (2 * max_u)  # CFL condition for convective term
    dt_cfl_diffusion = dx**2 / (2 * nu)  # CFL condition for diffusion term
    dt_internal = min(dt_cfl_convective, dt_cfl_diffusion) * 0.5  # Use a larger time step for efficiency
    
    solutions = torch.zeros((batch_size, T+1, N), device=device)
    solutions[:, 0, :] = u0_batch
    
    # Time stepping loop
    current_time = 0.0
    u = u0_batch.clone()
    for i in range(1, T+1):
        target_time = t_coordinate[i]
        while current_time < target_time:
            # Compute the convective term: -u * du/dx
            convective_term = -u * central_diff(u, dx)
            
            # Compute the diffusion term: nu * d^2u/dx^2
            diffusion_term = nu * second_order_central_diff(u, dx)
            
            # Update u using explicit Euler method
            u = u + dt_internal * (convective_term + diffusion_term)
            
            # Enforce periodic boundary conditions
            u = torch.roll(u, shifts=1, dims=-1)
            u = torch.roll(u, shifts=-1, dims=-1)
            
            current_time += dt_internal
        
        # Store the solution at the target time
        solutions[:, i, :] = u
    
    # Convert the result back to numpy array
    return solutions.cpu().numpy()
