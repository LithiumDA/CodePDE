
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
    
    # Set up spatial grid (assuming uniform grid on [0,1])
    dx = 1.0 / N
    x = torch.linspace(0, 1.0 - dx, N, device=device)
    
    # Prepare wavenumbers for spectral differentiation
    k = torch.fft.fftfreq(N, d=dx) * 2 * np.pi
    k = k.to(device)
    
    # Initialize solutions array
    solutions = torch.zeros((batch_size, len(t_coordinate), N), dtype=torch.float32, device=device)
    solutions[:, 0, :] = u0_tensor
    
    # Current solution at time t
    u_current = u0_tensor.clone()
    
    # Adaptive time stepping
    # For ν=0.01, we can determine a good internal time step
    # CFL condition: dt < dx^2 / (2*nu) for diffusion term stability
    # For advection term: dt < dx / max(abs(u))
    
    # Estimate max velocity for initial condition
    max_u = torch.max(torch.abs(u_current)).item()
    
    # Calculate internal time step based on stability criteria
    # Using a safety factor of 0.5
    dt_diff = 0.5 * dx**2 / (2 * nu)
    dt_adv = 0.5 * dx / max(max_u, 1e-10)  # Avoid division by zero
    dt_internal = min(dt_diff, dt_adv)
    
    # Make sure internal time step is small enough
    dt_internal = min(dt_internal, 0.001)  # Cap at 0.001 for stability
    
    print(f"Using internal time step: {dt_internal}")
    
    # Define RK4 step function
    def rk4_step(u, dt):
        """Fourth-order Runge-Kutta step."""
        k1 = dt * burgers_rhs(u)
        k2 = dt * burgers_rhs(u + 0.5 * k1)
        k3 = dt * burgers_rhs(u + 0.5 * k2)
        k4 = dt * burgers_rhs(u + k3)
        return u + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    # Define the right-hand side of Burgers' equation using spectral method
    def burgers_rhs(u):
        """Compute right-hand side of Burgers' equation using spectral method."""
        # Transform to Fourier space
        u_hat = torch.fft.fft(u, dim=1)
        
        # Compute derivatives in Fourier space
        u_x_hat = 1j * k * u_hat
        u_xx_hat = -k**2 * u_hat
        
        # Transform back to physical space
        u_x = torch.fft.ifft(u_x_hat, dim=1).real
        u_xx = torch.fft.ifft(u_xx_hat, dim=1).real
        
        # Compute right-hand side: -u*u_x + nu*u_xx
        return -u * u_x + nu * u_xx
    
    # Iterate through required output time points
    current_t = 0.0
    for i in range(1, len(t_coordinate)):
        target_t = t_coordinate[i]
        
        # Integrate until we reach the target time
        while current_t < target_t:
            # Adjust the last step to exactly hit the target time
            step_dt = min(dt_internal, target_t - current_t)
            
            # Apply RK4 step
            u_current = rk4_step(u_current, step_dt)
            current_t += step_dt
            
            # Periodically check if we need to adjust time step based on solution
            if i % 10 == 0:
                max_u = torch.max(torch.abs(u_current)).item()
                dt_adv = 0.5 * dx / max(max_u, 1e-10)
                new_dt = min(dt_diff, dt_adv, 0.001)
                
                # Only update if significantly different
                if abs(new_dt - dt_internal) / dt_internal > 0.2:
                    dt_internal = new_dt
                    print(f"Adjusted internal time step to: {dt_internal}")
        
        # Store the solution at this time point
        solutions[:, i, :] = u_current
        
        # Print progress
        if i % 5 == 0 or i == len(t_coordinate) - 1:
            print(f"Completed time step {i}/{len(t_coordinate)-1}, t={target_t:.4f}")
    
    # Convert back to numpy array
    return solutions.cpu().numpy()
