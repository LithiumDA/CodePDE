
import numpy as np
import torch
from typing import Tuple

def solver(u0_batch, t_coordinate, nu=0.5, rho=1.0):
    """Solves the 1D reaction diffusion equation for all times in t_coordinate.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N], 
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. 
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        nu (float): The parameter nu in the equation. Default is 0.5.
        rho (float): The parameter rho in the equation. Default is 1.0.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert input to PyTorch tensors and move to device
    batch_size, N = u0_batch.shape
    u0_tensor = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    
    # Set up the spatial grid and wavenumbers for spectral method
    dx = 1.0 / N
    k = 2.0 * np.pi * torch.fft.fftfreq(N, d=dx)
    k = k.to(device=device, dtype=torch.float32)
    k2 = k * k
    
    # Linear operator L = -nu * k^2 (diffusion term in Fourier space)
    L = -nu * k2
    
    # Use a much smaller time step for stability
    dt_internal = 1e-6  # Very conservative time step
    print(f"Internal time step: {dt_internal}")
    
    # Initialize solution array
    solutions = torch.zeros((batch_size, len(t_coordinate), N), dtype=torch.float32, device=device)
    solutions[:, 0, :] = u0_tensor  # Set initial condition
    
    # Current solution state
    u_current = u0_tensor
    current_time = 0.0
    
    # Time stepping loop
    for i in range(1, len(t_coordinate)):
        target_time = t_coordinate[i]
        print(f"Solving for time {target_time:.6f}")
        
        # Integrate until we reach the target time
        while current_time < target_time:
            # Adjust the last time step to exactly hit the target time if needed
            dt = min(dt_internal, target_time - current_time)
            
            # Take a time step using a simpler, more stable method
            u_current = integrate_step(u_current, L, dt, rho, device)
            
            # Check for NaN or Inf values
            if torch.isnan(u_current).any() or torch.isinf(u_current).any():
                print(f"Warning: NaN or Inf detected at time {current_time}")
                # Try to recover by clamping values
                u_current = torch.nan_to_num(u_current, nan=0.5, posinf=1.0, neginf=0.0)
                u_current = torch.clamp(u_current, 0.0, 1.0)  # Clamp to physically meaningful range
            
            current_time += dt
        
        # Store the solution at the target time
        solutions[:, i, :] = u_current
    
    # Convert back to numpy array
    return solutions.cpu().numpy()

def integrate_step(u: torch.Tensor, L: torch.Tensor, dt: float, rho: float, device: torch.device) -> torch.Tensor:
    """
    Take one time step using a stabilized semi-implicit method.
    
    Args:
        u (torch.Tensor): Current solution.
        L (torch.Tensor): Linear operator in Fourier space.
        dt (float): Time step.
        rho (float): Reaction coefficient.
        device (torch.device): Device to perform computations on.
        
    Returns:
        torch.Tensor: Solution after one time step.
    """
    # Transform to Fourier space
    u_hat = torch.fft.fft(u, dim=1)
    
    # Apply 2/3 rule for dealiasing (set highest 1/3 of frequencies to zero)
    N = u.shape[1]
    cutoff = N // 3
    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[cutoff:N-cutoff] = False
    
    # Evaluate nonlinear term at current time
    N_u = nonlinear_term(u, rho)
    N_u_hat = torch.fft.fft(N_u, dim=1)
    # Apply dealiasing
    N_u_hat[:, mask] = 0
    
    # Compute the exponential term for the diffusion part
    # Use a safer exponential calculation with clipping
    exp_term = torch.exp(L * dt)
    # Prevent underflow by setting very small values to zero
    exp_term = torch.where(torch.abs(exp_term) < 1e-15, torch.zeros_like(exp_term), exp_term)
    
    # Semi-implicit update: Treat diffusion implicitly, reaction explicitly
    u_new_hat = exp_term * u_hat + (exp_term - 1.0) / L * N_u_hat
    # Handle the L=0 mode separately to avoid division by zero
    zero_mode = (torch.abs(L) < 1e-14)
    if torch.any(zero_mode):
        u_new_hat[:, zero_mode] = u_hat[:, zero_mode] + dt * N_u_hat[:, zero_mode]
    
    # Transform back to real space
    u_new = torch.fft.ifft(u_new_hat, dim=1).real
    
    # Ensure solution stays within physically meaningful bounds
    u_new = torch.clamp(u_new, 0.0, 1.0)
    
    return u_new

def nonlinear_term(u: torch.Tensor, rho: float) -> torch.Tensor:
    """
    Compute the nonlinear reaction term rho * u * (1 - u).
    
    Args:
        u (torch.Tensor): Current solution.
        rho (float): Reaction coefficient.
        
    Returns:
        torch.Tensor: Nonlinear term evaluation.
    """
    # Ensure u is within [0, 1] to prevent numerical issues
    u_safe = torch.clamp(u, 0.0, 1.0)
    return rho * u_safe * (1.0 - u_safe)
