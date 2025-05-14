
import torch
import numpy as np

def compute_fluxes(density, velocity, pressure, epsilon, gamma=5/3):
    """Compute the fluxes for mass, momentum, and energy."""
    mass_flux = density * velocity
    momentum_flux = density * velocity**2 + pressure
    energy_flux = (epsilon + pressure + 0.5 * density * velocity**2) * velocity
    return mass_flux, momentum_flux, energy_flux

def compute_viscous_terms(velocity, eta, zeta, dx):
    """Compute the viscous terms in the momentum and energy equations."""
    dv_dx = (torch.roll(velocity, -1, dims=-1) - torch.roll(velocity, 1, dims=-1)) / (2 * dx)
    d2v_dx2 = (torch.roll(velocity, -1, dims=-1) - 2 * velocity + torch.roll(velocity, 1, dims=-1)) / dx**2
    sigma_prime = (zeta + 4/3 * eta) * dv_dx
    viscous_momentum = eta * d2v_dx2 + (zeta + eta/3) * (torch.roll(dv_dx, -1, dims=-1) - torch.roll(dv_dx, 1, dims=-1)) / (2 * dx)
    viscous_energy = -velocity * sigma_prime
    return viscous_momentum, viscous_energy

def time_step(density, velocity, pressure, epsilon, eta, zeta, dx, dt, gamma=5/3):
    """Perform a single time step using the Runge-Kutta 4th order method."""
    # Compute fluxes and viscous terms
    mass_flux, momentum_flux, energy_flux = compute_fluxes(density, velocity, pressure, epsilon, gamma)
    viscous_momentum, viscous_energy = compute_viscous_terms(velocity, eta, zeta, dx)
    
    # Update density, velocity, and pressure
    density_new = density - dt * (torch.roll(mass_flux, -1, dims=-1) - torch.roll(mass_flux, 1, dims=-1)) / (2 * dx)
    velocity_new = velocity - dt * (torch.roll(momentum_flux, -1, dims=-1) - torch.roll(momentum_flux, 1, dims=-1)) / (2 * dx) + dt * viscous_momentum / density
    epsilon_new = epsilon - dt * (torch.roll(energy_flux, -1, dims=-1) - torch.roll(energy_flux, 1, dims=-1)) / (2 * dx) + dt * viscous_energy
    pressure_new = (gamma - 1) * epsilon_new
    
    # Clamp values to prevent overflow
    density_new = torch.clamp(density_new, min=1e-10, max=1e10)
    velocity_new = torch.clamp(velocity_new, min=-1e5, max=1e5)
    pressure_new = torch.clamp(pressure_new, min=1e-10, max=1e10)
    
    return density_new, velocity_new, pressure_new, epsilon_new

def solver(Vx0, density0, pressure0, t_coordinate, eta, zeta):
    """Solves the 1D compressible Navier-Stokes equations for all times in t_coordinate."""
    # Convert input NumPy arrays to PyTorch tensors
    Vx0 = torch.tensor(Vx0, dtype=torch.float32)
    density0 = torch.tensor(density0, dtype=torch.float32)
    pressure0 = torch.tensor(pressure0, dtype=torch.float32)
    
    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Vx0 = Vx0.to(device)
    density0 = density0.to(device)
    pressure0 = pressure0.to(device)
    
    batch_size, N = Vx0.shape
    T = len(t_coordinate) - 1
    dx = 1.0 / N  # Assuming spatial domain is [0, 1]
    
    # Initialize arrays to store results
    Vx_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float32, device=device)
    density_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float32, device=device)
    pressure_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float32, device=device)
    
    # Set initial conditions
    Vx_pred[:, 0, :] = Vx0
    density_pred[:, 0, :] = density0
    pressure_pred[:, 0, :] = pressure0
    
    # Internal time step for stability
    dt_internal = 0.001 * dx  # Smaller time step for stability
    
    # Time stepping loop
    for t_idx in range(T):
        t_start = t_coordinate[t_idx]
        t_end = t_coordinate[t_idx+1]
        
        # Initialize variables for current time step
        density = density_pred[:, t_idx, :]
        velocity = Vx_pred[:, t_idx, :]
        pressure = pressure_pred[:, t_idx, :]
        epsilon = pressure / (5/3 - 1)
        
        # Perform internal time steps
        while t_start < t_end:
            dt = min(dt_internal, t_end - t_start)
            density, velocity, pressure, epsilon = time_step(density, velocity, pressure, epsilon, eta, zeta, dx, dt)
            t_start += dt
        
        # Store results
        Vx_pred[:, t_idx+1, :] = velocity
        density_pred[:, t_idx+1, :] = density
        pressure_pred[:, t_idx+1, :] = pressure
    
    return Vx_pred.cpu().numpy(), density_pred.cpu().numpy(), pressure_pred.cpu().numpy()
