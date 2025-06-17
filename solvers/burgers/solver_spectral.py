import numpy as np
import torch

def solver(u0_batch, t_coordinate, nu):
    """Solves the Burgers' equation for all times in t_coordinate using a spectral method.
    
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
    # Check if GPU is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert numpy arrays to PyTorch tensors and move to device
    batch_size, N = u0_batch.shape
    u0_tensor = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    
    # Spatial grid parameters (domain [0,1] with N points)
    dx = 1.0 / N  # Grid spacing
    
    # Wavenumbers for spectral differentiation
    # For real FFT, we only need N//2 + 1 frequencies due to conjugate symmetry
    k = 2 * np.pi * torch.fft.rfftfreq(N, dx).to(device)  # Scaled for domain [0,1]
    
    # Size of the rfft output
    M = N // 2 + 1
    
    # Dealiasing mask using the 2/3 rule to prevent aliasing errors from nonlinear terms
    # We set the highest 1/3 of frequencies to zero
    dealias_mask = torch.ones(M, dtype=torch.bool, device=device)
    cutoff_idx = int(2 * M // 3)  # Cut off at 2/3 of the Nyquist frequency
    dealias_mask[cutoff_idx:] = False
    
    # Initialize solution array to store results at specified time points
    T_steps = len(t_coordinate) - 1
    solutions_tensor = torch.zeros((batch_size, T_steps + 1, N), dtype=torch.float32, device=device)
    solutions_tensor[:, 0, :] = u0_tensor  # Store initial condition
    
    # Determine internal time step for stability
    # 1. Advection stability (CFL condition)
    # For Burgers' equation, the wave speed is the solution itself
    max_u = torch.max(torch.abs(u0_tensor)).item()  # Maximum wave speed
    C_cfl = 0.5  # Safety factor
    dt_cfl = C_cfl * dx / max(max_u, 1e-10)  # Avoid division by zero
    
    # 2. Diffusion stability
    C_diff = 0.2  # Conservative for spectral methods
    dt_diff = C_diff * dx**2 / (2 * nu) if nu > 0 else float('inf')
    
    # Use the most restrictive condition and cap at a reasonable value for accuracy
    dt_internal = min(dt_cfl, dt_diff, 1e-3)
    
    print(f"Grid resolution (N): {N}")
    print(f"Viscosity coefficient (nu): {nu}")
    print(f"Maximum wave speed: {max_u:.6f}")
    print(f"Internal time step: {dt_internal:.6f} (CFL: {dt_cfl:.6f}, Diffusion: {dt_diff:.6f})")
    
    # Semi-implicit time-stepping function with integrating factor for the diffusion term
    def advance_timestep(u_hat, dt):
        """Advance the solution one time step using a semi-implicit scheme with integrating factor.
        
        Args:
            u_hat: Current solution in Fourier space [batch_size, M]
            dt: Time step
            
        Returns:
            Updated solution in Fourier space
        """
        # Transform to physical space to compute the nonlinear term
        u = torch.fft.irfft(u_hat, n=N, dim=1)  # [batch_size, N]
        
        # Compute nonlinear term: u * du/dx = 0.5 * d(u^2)/dx
        nonlinear = 0.5 * u**2  # [batch_size, N]
        nonlinear_hat = torch.fft.rfft(nonlinear, n=N, dim=1)  # [batch_size, M]
        
        # Apply dealiasing to prevent aliasing errors
        nonlinear_hat = nonlinear_hat * dealias_mask.unsqueeze(0)  # Broadcasting
        
        # Compute the nonlinear contribution: -i*k*F{u^2/2}
        nonlinear_contribution = -1j * k.unsqueeze(0) * nonlinear_hat  # [batch_size, M]
        
        # Integrating factor for diffusion term: exp(-nu*k^2*dt)
        # This treats the diffusion term implicitly for better stability
        integrating_factor = torch.exp(-nu * k.unsqueeze(0)**2 * dt)  # [1, M]
        
        # Apply semi-implicit update:
        # û(t+dt) = exp(-nu*k^2*dt) * (û(t) + dt * nonlinear_contribution)
        u_hat_new = integrating_factor * (u_hat + dt * nonlinear_contribution)
        
        return u_hat_new
    
    # Initialize solution in Fourier space
    u_hat = torch.fft.rfft(u0_tensor, n=N, dim=1)  # [batch_size, M]
    
    # Time-stepping loop
    t_current = 0.0
    t_idx = 1  # Start from the first output time (t_coordinate[1])
    
    while t_idx <= T_steps:
        t_target = t_coordinate[t_idx]
        steps_taken = 0
        
        # Take internal steps until we reach the target time
        while t_current < t_target:
            # Ensure we hit the target time exactly
            dt = min(dt_internal, t_target - t_current)
            
            # Update solution in Fourier space
            u_hat = advance_timestep(u_hat, dt)
            
            t_current += dt
            steps_taken += 1
            
            # Adapt time step based on current solution (every 50 steps)
            # This accounts for changing wave speeds during the simulation
            if steps_taken % 50 == 0:
                u = torch.fft.irfft(u_hat, n=N, dim=1)
                new_max_u = torch.max(torch.abs(u)).item()
                
                # Update CFL condition based on current maximum wave speed
                new_dt_cfl = C_cfl * dx / max(new_max_u, 1e-10)
                dt_internal = min(new_dt_cfl, dt_diff, 1e-3)
        
        # Store solution at this output time
        u = torch.fft.irfft(u_hat, n=N, dim=1)  # Transform back to physical space
        solutions_tensor[:, t_idx, :] = u
        
        # Print progress
        if t_idx % max(1, T_steps // 10) == 0 or t_idx == T_steps:
            print(f"Completed time step {t_idx}/{T_steps}, t = {t_target:.4f} ({steps_taken} internal steps)")
            # Also report maximum value of solution (for debugging)
            current_max_u = torch.max(torch.abs(u)).item()
            print(f"  Current max|u|: {current_max_u:.6f}")
        
        t_idx += 1
    
    # Convert solution to numpy array for return
    solutions = solutions_tensor.cpu().numpy()
    
    return solutions