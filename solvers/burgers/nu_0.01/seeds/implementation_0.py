
import numpy as np
import torch

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
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert numpy arrays to PyTorch tensors and move to device
    batch_size, N = u0_batch.shape
    u0_tensor = torch.tensor(u0_batch, dtype=torch.float32).to(device)
    
    # Initialize solutions tensor with the initial condition
    t_steps = len(t_coordinate)
    solutions_tensor = torch.zeros((batch_size, t_steps, N), dtype=torch.float32).to(device)
    solutions_tensor[:, 0, :] = u0_tensor
    
    # Compute spatial grid and wave numbers for spectral method
    dx = 1.0 / N  # Assuming domain is [0, 1]
    k = torch.fft.fftfreq(N, d=dx) * 2 * np.pi  # Wave numbers
    k = k.to(device)
    
    # For stability, determine internal time step based on CFL condition
    # and diffusion stability condition
    max_u = torch.max(torch.abs(u0_tensor)).item()
    dt_cfl = 0.5 * dx / max_u if max_u > 0 else 1.0  # CFL condition
    dt_diff = 0.5 * dx**2 / nu  # Diffusion stability
    dt_internal = min(0.01, 0.5 * min(dt_cfl, dt_diff))  # Conservative choice
    print(f"Internal time step: {dt_internal}")
    
    # Current solution and current time
    u_current = u0_tensor.clone()
    t_current = 0.0
    next_output_idx = 1  # Index for the next output time
    
    # RK4 coefficients
    rk4_a = [0.5, 0.5, 1.0]
    rk4_b = [1/6, 1/3, 1/3, 1/6]
    
    # Time integration loop
    while next_output_idx < t_steps:
        next_output_time = t_coordinate[next_output_idx]
        
        # Integrate until we reach the next output time
        while t_current < next_output_time:
            # Adjust dt if necessary to exactly hit the next output time
            dt = min(dt_internal, next_output_time - t_current)
            
            # Apply RK4 time integration
            k1 = dt * burgers_rhs(u_current, k, nu, device)
            k2 = dt * burgers_rhs(u_current + rk4_a[0] * k1, k, nu, device)
            k3 = dt * burgers_rhs(u_current + rk4_a[1] * k2, k, nu, device)
            k4 = dt * burgers_rhs(u_current + rk4_a[2] * k3, k, nu, device)
            
            u_current = u_current + rk4_b[0]*k1 + rk4_b[1]*k2 + rk4_b[2]*k3 + rk4_b[3]*k4
            t_current += dt
            
        # Store the solution at the output time
        solutions_tensor[:, next_output_idx, :] = u_current
        next_output_idx += 1
        
        # Print progress
        if next_output_idx % 5 == 0 or next_output_idx == t_steps:
            print(f"Completed {next_output_idx}/{t_steps} time steps")
    
    # Convert back to numpy array
    solutions = solutions_tensor.cpu().numpy()
    return solutions

def burgers_rhs(u, k, nu, device):
    """Compute the right-hand side of Burgers' equation using spectral method.
    
    Args:
        u (torch.Tensor): Current solution [batch_size, N]
        k (torch.Tensor): Wave numbers [N]
        nu (float): Viscosity coefficient
        device: The device to perform computations on
        
    Returns:
        du_dt (torch.Tensor): Time derivative [batch_size, N]
    """
    batch_size, N = u.shape
    
    # Compute the nonlinear term using FFT
    u_hat = torch.fft.fft(u, dim=1)
    
    # Compute the derivative of u^2/2 in spectral space
    u_squared = 0.5 * u * u
    u_squared_hat = torch.fft.fft(u_squared, dim=1)
    du_squared_dx_hat = 1j * k.reshape(1, -1) * u_squared_hat
    du_squared_dx = torch.fft.ifft(du_squared_dx_hat, dim=1).real
    
    # Compute the diffusion term in spectral space
    d2u_dx2_hat = -k.reshape(1, -1)**2 * u_hat
    d2u_dx2 = torch.fft.ifft(d2u_dx2_hat, dim=1).real
    
    # Combine terms for the RHS of Burgers' equation
    du_dt = -du_squared_dx + nu * d2u_dx2
    
    return du_dt
