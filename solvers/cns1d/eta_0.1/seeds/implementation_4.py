
import numpy as np
import torch
import math

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
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert numpy arrays to PyTorch tensors and move to device
    Vx = torch.tensor(Vx0, dtype=torch.float64).to(device)  # Using double precision
    density = torch.tensor(density0, dtype=torch.float64).to(device)
    pressure = torch.tensor(pressure0, dtype=torch.float64).to(device)
    
    # Ensure positive pressure and density
    density = torch.clamp(density, min=1e-6)
    pressure = torch.clamp(pressure, min=1e-6)
    
    # Get dimensions
    batch_size, N = Vx.shape
    T = len(t_coordinate) - 1
    
    # Initialize output arrays
    Vx_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float64).to(device)
    density_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float64).to(device)
    pressure_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float64).to(device)
    
    # Set initial conditions
    Vx_pred[:, 0, :] = Vx
    density_pred[:, 0, :] = density
    pressure_pred[:, 0, :] = pressure
    
    # Physical parameters
    Gamma = 5/3  # Adiabatic index
    
    # Domain setup (assuming uniform grid on [-1, 1])
    L = 2.0  # Domain length
    dx = L / N  # Spatial step size
    
    # Calculate internal energy
    def calc_internal_energy(p):
        return p / (Gamma - 1)
    
    # Calculate total energy
    def calc_total_energy(rho, v, p):
        return calc_internal_energy(p) + 0.5 * rho * v**2
    
    # First-order upwind derivative for advection terms
    def upwind_derivative(f, v):
        # v > 0: use backward difference, v < 0: use forward difference
        f_right = torch.roll(f, -1, dims=-1)
        f_left = torch.roll(f, 1, dims=-1)
        
        # Compute backward and forward differences
        backward_diff = (f - f_left) / dx
        forward_diff = (f_right - f) / dx
        
        # Choose based on velocity direction
        return torch.where(v > 0, backward_diff, forward_diff)
    
    # Central difference for non-advection terms
    def central_difference(f):
        f_right = torch.roll(f, -1, dims=-1)
        f_left = torch.roll(f, 1, dims=-1)
        return (f_right - f_left) / (2 * dx)
    
    # Second derivative with periodic boundaries
    def second_derivative(f):
        f_right = torch.roll(f, -1, dims=-1)
        f_left = torch.roll(f, 1, dims=-1)
        return (f_right - 2*f + f_left) / (dx**2)
    
    # Artificial viscosity to stabilize shocks
    def artificial_viscosity(rho, v, p, c):
        # Compute divergence of velocity
        div_v = central_difference(v)
        
        # Only apply where compression occurs (div_v < 0)
        art_visc = torch.zeros_like(p)
        compression = div_v < 0
        if compression.any():
            # Quadratic artificial viscosity term
            art_coeff = 1.0  # Coefficient (adjust as needed)
            art_visc[compression] = art_coeff * rho[compression] * dx**2 * div_v[compression]**2
        
        return art_visc
    
    # Current simulation time
    current_time = 0.0
    
    # Main time-stepping loop
    for i in range(1, T+1):
        target_time = t_coordinate[i]
        
        # Adaptive time stepping
        while current_time < target_time:
            # Calculate sound speed (ensure pressure and density are positive)
            sound_speed = torch.sqrt(Gamma * torch.clamp(pressure, min=1e-10) / 
                                    torch.clamp(density, min=1e-10))
            
            # CFL condition (taking into account viscosity)
            v_max = torch.max(torch.abs(Vx) + sound_speed).item()
            visc_dt = 0.25 * dx**2 / max(eta, zeta)  # More conservative viscous constraint
            dt_cfl = 0.2 * dx / (v_max + 1e-10)  # More conservative CFL condition
            dt = min(dt_cfl, visc_dt, target_time - current_time)
            
            # Ensure dt is positive and not too small
            if dt < 1e-10:
                dt = 1e-10
                
            # Print diagnostics
            if current_time % 0.001 < dt:  # Only print occasionally
                print(f"Time: {current_time:.6f}, dt: {dt:.6f}, Target: {target_time:.6f}, "
                      f"Max vel: {torch.max(torch.abs(Vx)).item():.4f}, "
                      f"Min density: {torch.min(density).item():.4e}, "
                      f"Min pressure: {torch.min(pressure).item():.4e}")
            
            # Check for NaN or inf in tensors
            if torch.isnan(Vx).any() or torch.isinf(Vx).any() or \
               torch.isnan(density).any() or torch.isinf(density).any() or \
               torch.isnan(pressure).any() or torch.isinf(pressure).any():
                print("NaN or Inf detected in tensors! Stopping simulation.")
                break
            
            # Check for NaN in dt (using Python's math.isnan for scalar values)
            if math.isnan(dt):
                print("NaN detected in dt! Stopping simulation.")
                break
            
            # Calculate energy
            energy = calc_total_energy(density, Vx, pressure)
            
            # Calculate artificial viscosity
            art_visc = artificial_viscosity(density, Vx, pressure, sound_speed)
            
            # Update with Lax-Friedrichs scheme for better stability
            # For density (continuity equation)
            density_flux = density * Vx
            density_right = torch.roll(density, -1, dims=-1)
            density_left = torch.roll(density, 1, dims=-1)
            flux_right = torch.roll(density_flux, -1, dims=-1)
            flux_left = torch.roll(density_flux, 1, dims=-1)
            
            density_new = 0.5 * (density_right + density_left) - \
                          0.5 * dt/dx * (flux_right - flux_left)
            
            # Ensure density remains positive
            density_new = torch.clamp(density_new, min=1e-6)
            
            # For momentum (momentum equation)
            momentum = density * Vx
            momentum_flux = density * Vx * Vx + pressure + art_visc
            
            # Viscous terms
            viscous_term = eta * second_derivative(Vx) + (zeta + eta/3) * central_difference(central_difference(Vx))
            
            momentum_right = torch.roll(momentum, -1, dims=-1)
            momentum_left = torch.roll(momentum, 1, dims=-1)
            flux_right = torch.roll(momentum_flux, -1, dims=-1)
            flux_left = torch.roll(momentum_flux, 1, dims=-1)
            visc_right = torch.roll(viscous_term, -1, dims=-1)
            visc_left = torch.roll(viscous_term, 1, dims=-1)
            
            momentum_new = 0.5 * (momentum_right + momentum_left) - \
                           0.5 * dt/dx * (flux_right - flux_left) + \
                           0.5 * dt * (visc_right + visc_left)
            
            # For energy (energy equation)
            energy_flux = (energy + pressure + art_visc) * Vx
            
            # Viscous work term
            viscous_work = Vx * ((zeta + 4*eta/3) * central_difference(Vx))
            viscous_work_flux = viscous_work
            
            energy_right = torch.roll(energy, -1, dims=-1)
            energy_left = torch.roll(energy, 1, dims=-1)
            flux_right = torch.roll(energy_flux, -1, dims=-1)
            flux_left = torch.roll(energy_flux, 1, dims=-1)
            work_right = torch.roll(viscous_work_flux, -1, dims=-1)
            work_left = torch.roll(viscous_work_flux, 1, dims=-1)
            
            energy_new = 0.5 * (energy_right + energy_left) - \
                         0.5 * dt/dx * (flux_right - flux_left) + \
                         0.5 * dt/dx * (work_right - work_left)
            
            # Calculate new velocity from momentum and density
            Vx_new = momentum_new / density_new
            
            # Calculate new pressure from energy
            kinetic_energy = 0.5 * density_new * Vx_new**2
            internal_energy = energy_new - kinetic_energy
            pressure_new = (Gamma - 1) * internal_energy
            
            # Ensure pressure remains positive
            pressure_new = torch.clamp(pressure_new, min=1e-6)
            
            # Update variables for next step
            density = density_new
            Vx = Vx_new
            pressure = pressure_new
            
            # Update time
            current_time += dt
            
            # If we've reached the target time
            if current_time >= target_time:
                break
        
        # Store results at requested time points
        Vx_pred[:, i, :] = Vx
        density_pred[:, i, :] = density
        pressure_pred[:, i, :] = pressure
        
        # If we encountered NaN, no need to continue simulation
        if torch.isnan(Vx).any() or torch.isinf(Vx).any() or \
           torch.isnan(density).any() or torch.isinf(density).any() or \
           torch.isnan(pressure).any() or torch.isinf(pressure).any():
            print(f"NaN or Inf detected at time {current_time}. Filling remaining time steps with last valid values.")
            for j in range(i+1, T+1):
                Vx_pred[:, j, :] = Vx_pred[:, i, :]
                density_pred[:, j, :] = density_pred[:, i, :]
                pressure_pred[:, j, :] = pressure_pred[:, i, :]
            break
    
    # Convert back to numpy arrays
    return Vx_pred.cpu().numpy(), density_pred.cpu().numpy(), pressure_pred.cpu().numpy()
