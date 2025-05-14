
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
    
    # Initialize the solution array with the initial condition
    T = len(t_coordinate) - 1
    solutions_tensor = torch.zeros((batch_size, T+1, N), dtype=torch.float32, device=device)
    solutions_tensor[:, 0, :] = u0_tensor
    
    # Compute the wave numbers for spectral differentiation
    k = torch.fft.fftfreq(N, d=1.0/N) * 2 * np.pi  # Wave numbers
    k = k.to(device)
    
    # Initialize current solution
    u_current = u0_tensor.clone()
    
    # Determine a suitable internal time step for stability
    # For Burgers' with ν=0.01, we need smaller steps than the provided ones
    dx = 1.0 / N
    # CFL-like condition for stability: dt ≤ dx²/(2*ν) for diffusion part
    # and dt ≤ dx/max(|u|) for advection part
    max_u_magnitude = torch.max(torch.abs(u0_tensor)).item()
    dt_diff = 0.5 * dx**2 / nu  # Diffusion stability
    dt_adv = 0.5 * dx / max(max_u_magnitude, 1e-6)  # Advection stability
    dt_internal = min(dt_diff, dt_adv) * 0.5  # Safety factor
    
    print(f"Internal timestep: {dt_internal:.6f}")
    
    # Solve for each time point in t_coordinate
    current_t = 0.0
    for i in range(1, T+1):
        target_t = t_coordinate[i]
        
        # Integrate until we reach the target time
        while current_t < target_t:
            # Adjust the last step to exactly hit the target time
            step_size = min(dt_internal, target_t - current_t)
            
            # Use 4th-order Runge-Kutta method for time stepping
            u_current = rk4_step(u_current, step_size, k, nu, device)
            
            current_t += step_size
        
        # Store the solution at the target time
        solutions_tensor[:, i, :] = u_current
        
        if i % 5 == 0 or i == T:
            print(f"Completed time step {i}/{T}, t = {current_t:.4f}")
    
    # Convert back to numpy array
    solutions = solutions_tensor.cpu().numpy()
    return solutions

def rk4_step(u, dt, k, nu, device):
    """Perform one step of 4th-order Runge-Kutta time integration.
    
    Args:
        u (torch.Tensor): Current solution [batch_size, N]
        dt (float): Time step
        k (torch.Tensor): Wave numbers for spectral differentiation
        nu (float): Viscosity coefficient
        device: PyTorch device
        
    Returns:
        torch.Tensor: Solution after one time step
    """
    k1 = burgers_rhs(u, k, nu, device)
    k2 = burgers_rhs(u + 0.5 * dt * k1, k, nu, device)
    k3 = burgers_rhs(u + 0.5 * dt * k2, k, nu, device)
    k4 = burgers_rhs(u + dt * k3, k, nu, device)
    
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def burgers_rhs(u, k, nu, device):
    """Compute the right-hand side of Burgers' equation using spectral method.
    
    Args:
        u (torch.Tensor): Current solution [batch_size, N]
        k (torch.Tensor): Wave numbers for spectral differentiation
        nu (float): Viscosity coefficient
        device: PyTorch device
        
    Returns:
        torch.Tensor: Right-hand side of the equation
    """
    batch_size, N = u.shape
    
    # Compute the nonlinear term using spectral differentiation
    u_hat = torch.fft.fft(u, dim=1)
    
    # Compute the spatial derivative using spectral method
    ux_hat = 1j * k * u_hat  # du/dx in Fourier space
    ux = torch.fft.ifft(ux_hat, dim=1).real
    
    # Compute the second derivative for diffusion term
    uxx_hat = -k**2 * u_hat  # d²u/dx² in Fourier space
    uxx = torch.fft.ifft(uxx_hat, dim=1).real
    
    # Compute the right-hand side: -u*du/dx + ν*d²u/dx²
    # For Burgers' equation: du/dt + d(u²/2)/dx = ν*d²u/dx²
    # Which can be rewritten as: du/dt = -u*du/dx + ν*d²u/dx²
    rhs = -u * ux + nu * uxx
    
    return rhs
