
import numpy as np
import torch

def burgers_step(u, nu, dx, dt):
    """Computes the next time step for the Burgers' equation using finite differences."""
    # Central difference for spatial derivative
    u_x = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2 * dx)
    # Central difference for second spatial derivative (diffusion term)
    u_xx = (torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)) / (dx ** 2)
    # Update rule
    u_next = u - dt * (u * u_x) + nu * dt * u_xx
    
    # Clipping to prevent numerical overflow
    u_next = torch.clamp(u_next, -1e5, 1e5)
    return u_next

def solver(u0_batch, t_coordinate, nu):
    """Solves the Burgers' equation for all times in t_coordinate."""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert initial conditions to PyTorch tensors
    u0_batch = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    
    # Extract shapes and time step details
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1
    dx = 1.0 / N
    # Stricter CFL condition: consider both the velocity and viscosity
    max_velocity = torch.max(torch.abs(u0_batch)).item()
    dt_internal = (0.4 * dx ** 2) / (nu + max_velocity * dx)  # Adjusted CFL condition
    
    # Initialize solution array
    solutions = torch.zeros((batch_size, T + 1, N), device=device)
    solutions[:, 0, :] = u0_batch  # Set initial condition
    
    # Step through time
    for t in range(T):
        t_start = t_coordinate[t]
        t_end = t_coordinate[t + 1]
        
        # Internal time stepping
        u_current = solutions[:, t, :].clone()
        while t_start < t_end:
            effective_dt = min(dt_internal, t_end - t_start)
            
            # Update the current state using the Burgers equation step
            u_current = burgers_step(u_current, nu, dx, effective_dt)
            t_start += effective_dt
            
            # Debug print to ensure loop progresses
            print(f"t_start: {t_start}, dt_internal: {dt_internal}, effective_dt: {effective_dt}")
        
        # Store the result at the next time step
        solutions[:, t + 1, :] = u_current
    
    # Convert back to numpy for output
    solutions = solutions.cpu().numpy()
    return solutions
