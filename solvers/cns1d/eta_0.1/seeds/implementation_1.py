
import numpy as np
import torch

def solver(Vx0, density0, pressure0, t_coordinate, eta, zeta):
    """Solves the 1D compressible Navier-Stokes equations for all times in t_coordinate.

    Args:
        Vx0 (np.ndarray): Initial velocity of shape [batch_size, N] 
            where N is the number of evenly spaced spatial points.
        density0 (np.ndarray): Initial density [batch_size, N].
        pressure0 (np.ndarray): Initial pressure [batch_size, N].
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. 
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        eta (float): The shear viscosity coefficients in the equation.
        zeta (float): The bulk viscosity coefficients in the equation.

    Returns:
        Vx_pred (np.ndarray): Shape [batch_size, len(t_coordinate), N].
            The first timeframe is identical to Vx0.
            The subsequent timeframes are the solutions at the corresponding time steps.
        density_pred (np.ndarray): Shape [batch_size, len(t_coordinate), N].
        pressure_pred (np.ndarray): Shape [batch_size, len(t_coordinate), N].
    """
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert numpy arrays to PyTorch tensors
    Vx0_torch = torch.tensor(Vx0, dtype=torch.float32, device=device)
    density0_torch = torch.tensor(density0, dtype=torch.float32, device=device)
    pressure0_torch = torch.tensor(pressure0, dtype=torch.float32, device=device)
    
    batch_size, N = Vx0.shape
    T = len(t_coordinate) - 1
    
    # Initialize output arrays
    Vx_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float32, device=device)
    density_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float32, device=device)
    pressure_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float32, device=device)
    
    # Set initial conditions
    Vx_pred[:, 0, :] = Vx0_torch
    density_pred[:, 0, :] = density0_torch
    pressure_pred[:, 0, :] = pressure0_torch
    
    # Constants
    gamma = 5/3  # Specific heat ratio
    
    # Spatial step (assuming uniform grid with periodic boundary)
    dx = 1.0 / N
    
    # CFL number for stability but not too small
    cfl = 0.2
    
    # Minimum and maximum time step
    dt_min = 1e-6
    dt_max = 1e-3
    
    # Current time
    current_time = 0.0
    
    # Main time loop
    for t_idx in range(1, T+1):
        target_time = t_coordinate[t_idx]
        print(f"Advancing to time step {t_idx}/{T}, target time: {target_time:.6f}")
        
        # Current state
        vx = Vx_pred[:, t_idx-1, :].clone()
        rho = density_pred[:, t_idx-1, :].clone()
        p = pressure_pred[:, t_idx-1, :].clone()
        
        # Ensure physical values for initial state and limit extreme values
        rho = torch.clamp(rho, min=1e-2, max=1e3)
        p = torch.clamp(p, min=1e-2, max=1e3)
        vx = torch.clamp(vx, min=-1e2, max=1e2)  # Limit velocity to reasonable values
        
        # Simulate until we reach the target time
        steps_taken = 0
        
        while current_time < target_time:
            # Calculate sound speed
            c = torch.sqrt(gamma * p / rho)
            
            # Determine time step based on CFL condition
            max_wave_speed = torch.max(torch.abs(vx) + c).item()
            dt_cfl = cfl * dx / (max_wave_speed + 1.0)  # Add 1.0 to prevent division by near-zero
            
            # Consider viscous stability
            dt_visc = 0.5 * dx**2 / (eta + zeta)
            
            # Take the minimum of the constraints, but enforce min/max limits
            dt = min(dt_cfl, dt_visc, dt_max)
            dt = max(dt, dt_min)
            
            # Make sure we don't overshoot the target time
            if current_time + dt > target_time:
                dt = target_time - current_time
            
            # Update time
            current_time += dt
            steps_taken += 1
            
            # Use a very simple and stable update scheme
            rho_new, vx_new, p_new = simple_stable_update(rho, vx, p, dx, dt, gamma, eta, zeta)
            
            # Enforce physical bounds and limit extreme values
            rho_new = torch.clamp(rho_new, min=1e-2, max=1e3)
            p_new = torch.clamp(p_new, min=1e-2, max=1e3)
            vx_new = torch.clamp(vx_new, min=-1e2, max=1e2)
            
            # Update state
            rho = rho_new
            vx = vx_new
            p = p_new
            
            if steps_taken % 100 == 0:
                min_rho = torch.min(rho).item()
                min_p = torch.min(p).item()
                max_v = torch.max(torch.abs(vx)).item()
                print(f"  Sub-step {steps_taken}: t={current_time:.6f}, dt={dt:.8f}, max_v={max_v:.4f}, min_rho={min_rho:.6e}, min_p={min_p:.6e}")
        
        # Store the result at the target time
        Vx_pred[:, t_idx, :] = vx
        density_pred[:, t_idx, :] = rho
        pressure_pred[:, t_idx, :] = p
        
        print(f"  Completed time step {t_idx} with {steps_taken} sub-steps")
    
    # Convert back to numpy arrays
    return Vx_pred.cpu().numpy(), density_pred.cpu().numpy(), pressure_pred.cpu().numpy()

def simple_stable_update(rho, vx, p, dx, dt, gamma, eta, zeta):
    """
    Perform a simple, stable update using a combination of upwind for advection 
    and central differences for diffusion.
    
    Args:
        rho (torch.Tensor): Density tensor
        vx (torch.Tensor): Velocity tensor
        p (torch.Tensor): Pressure tensor
        dx (float): Spatial step size
        dt (float): Time step size
        gamma (float): Specific heat ratio
        eta (float): Shear viscosity coefficient
        zeta (float): Bulk viscosity coefficient
        
    Returns:
        rho_new, vx_new, p_new: Updated state variables
    """
    # Calculate internal energy
    epsilon = p / (gamma - 1)
    
    # Apply smoothing to all fields
    rho = apply_smoothing(rho)
    vx = apply_smoothing(vx)
    p = apply_smoothing(p)
    epsilon = apply_smoothing(epsilon)
    
    # First derivatives using upwind scheme for stability
    dvx_dx_upwind = upwind_derivative(vx, vx, dx)
    drho_dx_upwind = upwind_derivative(rho, vx, dx)
    dp_dx_upwind = upwind_derivative(p, vx, dx)
    
    # First derivatives using central differences for viscous terms
    dvx_dx_central = central_derivative(vx, dx)
    
    # Second derivatives for viscous terms
    d2vx_dx2 = second_derivative(vx, dx)
    
    # Mass conservation: ∂ρ/∂t + ∂(ρv)/∂x = 0
    drho_dt = -rho * dvx_dx_upwind - vx * drho_dx_upwind
    
    # Momentum conservation: ρ(∂v/∂t + v∂v/∂x) = -∂p/∂x + η∂²v/∂x² + (ζ+η/3)∂(∂v/∂x)/∂x
    # Viscous stress
    viscous_term = eta * d2vx_dx2 + (zeta + eta/3) * central_derivative(dvx_dx_central, dx)
    
    # Momentum equation
    dvx_dt = -vx * dvx_dx_upwind - (1.0/rho) * dp_dx_upwind + (1.0/rho) * viscous_term
    
    # Energy equation: ∂ε/∂t + v∂ε/∂x + p∂v/∂x = (ζ+4η/3)(∂v/∂x)²
    # Viscous heating term
    viscous_heating = (zeta + 4*eta/3) * dvx_dx_central**2
    
    # Energy equation
    depsilon_dt = -vx * upwind_derivative(epsilon, vx, dx) - p * dvx_dx_upwind + viscous_heating
    
    # Apply updates
    rho_new = rho + dt * drho_dt
    vx_new = vx + dt * dvx_dt
    epsilon_new = epsilon + dt * depsilon_dt
    
    # Calculate new pressure from internal energy
    p_new = (gamma - 1) * epsilon_new
    
    return rho_new, vx_new, p_new

def apply_smoothing(field, alpha=0.1):
    """Apply smoothing to reduce high-frequency oscillations."""
    field_m1 = torch.roll(field, shifts=1, dims=1)
    field_p1 = torch.roll(field, shifts=-1, dims=1)
    
    return (1-alpha) * field + alpha * 0.5 * (field_m1 + field_p1)

def upwind_derivative(field, velocity, dx):
    """Compute upwind derivative based on the sign of velocity."""
    field_m1 = torch.roll(field, shifts=1, dims=1)  # f_{i-1}
    field_p1 = torch.roll(field, shifts=-1, dims=1)  # f_{i+1}
    
    # Where velocity is positive, use backward difference
    # Where velocity is negative, use forward difference
    upwind = torch.where(
        velocity > 0,
        (field - field_m1) / dx,
        (field_p1 - field) / dx
    )
    
    return upwind

def central_derivative(field, dx):
    """Compute central difference derivative."""
    field_m1 = torch.roll(field, shifts=1, dims=1)  # f_{i-1}
    field_p1 = torch.roll(field, shifts=-1, dims=1)  # f_{i+1}
    
    return (field_p1 - field_m1) / (2 * dx)

def second_derivative(field, dx):
    """Compute second derivative using central differences."""
    field_m1 = torch.roll(field, shifts=1, dims=1)  # f_{i-1}
    field_p1 = torch.roll(field, shifts=-1, dims=1)  # f_{i+1}
    
    return (field_p1 - 2 * field + field_m1) / (dx**2)
