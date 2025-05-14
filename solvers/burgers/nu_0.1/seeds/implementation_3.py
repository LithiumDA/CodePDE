
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
    
    # Convert numpy arrays to PyTorch tensors
    batch_size, N = u0_batch.shape
    u0_tensor = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    
    # Set up the spatial domain
    L = 1.0  # Domain length (0 to 1)
    dx = L / N
    x = np.linspace(0, L - dx, N)  # Periodic domain excludes the endpoint
    
    # Set up wavenumbers for spectral differentiation
    k = 2 * np.pi * torch.fft.fftfreq(N, d=dx)
    k = torch.tensor(k, dtype=torch.float32, device=device)
    k_squared = k**2
    
    # Initialize solutions array
    solutions = torch.zeros((batch_size, len(t_coordinate), N), dtype=torch.float32, device=device)
    solutions[:, 0, :] = u0_tensor  # Set initial condition
    
    # Determine internal time step for stability
    # For spectral methods with Burgers' equation, we need a smaller time step
    # than what might be provided in t_coordinate
    max_u = torch.max(torch.abs(u0_tensor)).item()
    # CFL-like condition for stability
    dt_internal = min(0.1 * dx / max_u, 0.5 * dx**2 / nu)
    
    # Ensure dt_internal is not too small to avoid excessive computation
    dt_internal = max(dt_internal, 1e-4)
    print(f"Using internal time step: {dt_internal}")
    
    # Define the right-hand side of the PDE for the pseudospectral method
    def rhs(u):
        # Compute the spatial derivatives using FFT
        u_hat = torch.fft.fft(u, dim=-1)
        
        # Compute first derivative for advection term
        ux_hat = 1j * k * u_hat
        ux = torch.fft.ifft(ux_hat, dim=-1).real
        
        # Compute second derivative for diffusion term
        uxx_hat = -k_squared * u_hat
        uxx = torch.fft.ifft(uxx_hat, dim=-1).real
        
        # Burgers' equation: u_t + u*u_x = nu*u_xx
        return -u * ux + nu * uxx
    
    # 4th-order Runge-Kutta method for time integration
    def rk4_step(u, dt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Time integration loop
    current_t = 0.0
    current_u = u0_tensor
    
    for i in range(1, len(t_coordinate)):
        target_t = t_coordinate[i]
        
        # Integrate from current_t to target_t using internal time steps
        while current_t < target_t:
            # Adjust the last step to exactly hit target_t
            dt = min(dt_internal, target_t - current_t)
            current_u = rk4_step(current_u, dt)
            current_t += dt
            
            # Monitor maximum value to detect instabilities
            if torch.isnan(current_u).any() or torch.isinf(current_u).any():
                print(f"Warning: Numerical instability detected at t={current_t}")
                # Try to recover with a smaller time step
                current_u = solutions[:, i-1, :]
                dt_internal *= 0.5
                current_t = t_coordinate[i-1]
                print(f"Reducing time step to {dt_internal} and restarting from previous solution")
                if dt_internal < 1e-6:
                    print("Time step too small, simulation unstable")
                    break
        
        # Store the solution at the target time
        solutions[:, i, :] = current_u
        
        # Print progress
        if i % max(1, len(t_coordinate) // 10) == 0:
            print(f"Completed time step {i}/{len(t_coordinate)-1}, t={target_t:.4f}")
    
    # Convert back to numpy array
    return solutions.cpu().numpy()
