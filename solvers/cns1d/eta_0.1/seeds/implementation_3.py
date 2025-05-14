
import torch
import numpy as np

def compute_fluxes(density, velocity, pressure, epsilon, gamma=5/3):
    """Compute the fluxes for mass, momentum, and energy equations."""
    mass_flux = density * velocity
    momentum_flux = density * velocity**2 + pressure
    energy_flux = (epsilon + pressure + 0.5 * density * velocity**2) * velocity
    return mass_flux, momentum_flux, energy_flux

def compute_viscous_terms(velocity, eta, zeta, dx):
    """Compute the viscous terms in the momentum and energy equations."""
    dv_dx = (torch.roll(velocity, -1, dims=-1) - torch.roll(velocity, 1, dims=-1)) / (2 * dx)
    d2v_dx2 = (torch.roll(velocity, -1, dims=-1) - 2 * velocity + torch.roll(velocity, 1, dims=-1)) / (dx**2)
    sigma_prime = (zeta + 4/3 * eta) * dv_dx
    viscous_momentum = eta * d2v_dx2 + (zeta + eta/3) * (torch.roll(dv_dx, -1, dims=-1) - torch.roll(dv_dx, 1, dims=-1)) / (2 * dx)
    viscous_energy = -velocity * sigma_prime
    return viscous_momentum, viscous_energy

def time_step(density, velocity, pressure, epsilon, dx, dt, eta, zeta, gamma=5/3):
    """Perform a single time step using Runge-Kutta 4th order."""
    mass_flux, momentum_flux, energy_flux = compute_fluxes(density, velocity, pressure, epsilon, gamma)
    viscous_momentum, viscous_energy = compute_viscous_terms(velocity, eta, zeta, dx)
    
    # Update density
    density_new = density - dt * (torch.roll(mass_flux, -1, dims=-1) - torch.roll(mass_flux, 1, dims=-1)) / (2 * dx)
    
    # Update velocity
    velocity_new = velocity - dt * (torch.roll(momentum_flux, -1, dims=-1) - torch.roll(momentum_flux, 1, dims=-1)) / (2 * dx) / density
    velocity_new += dt * viscous_momentum / density
    
    # Update pressure (via energy equation)
    energy_new = epsilon + 0.5 * density * velocity**2
    energy_new -= dt * (torch.roll(energy_flux, -1, dims=-1) - torch.roll(energy_flux, 1, dims=-1)) / (2 * dx)
    energy_new += dt * viscous_energy
    epsilon_new = energy_new - 0.5 * density_new * velocity_new**2
    pressure_new = (gamma - 1) * epsilon_new
    
    return density_new, velocity_new, pressure_new, epsilon_new

def solver(Vx0, density0, pressure0, t_coordinate, eta, zeta):
    """Solves the 1D compressible Navier-Stokes equations for all times in t_coordinate."""
    batch_size, N = Vx0.shape
    T = len(t_coordinate) - 1
    dx = 1.0 / N  # Assuming spatial domain is [0, 1]
    
    # Convert inputs to PyTorch tensors
    Vx0 = torch.tensor(Vx0, dtype=torch.float32)
    density0 = torch.tensor(density0, dtype=torch.float32)
    pressure0 = torch.tensor(pressure0, dtype=torch.float32)
    
    # Initialize output arrays
    Vx_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float32)
    density_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float32)
    pressure_pred = torch.zeros((batch_size, T+1, N), dtype=torch.float32)
    
    # Set initial conditions
    Vx_pred[:, 0, :] = Vx0
    density_pred[:, 0, :] = density0
    pressure_pred[:, 0, :] = pressure0
    
    # Internal time step for stability
    dt_internal = 0.001 * dx  # Smaller time step for stability
    
    # Time stepping loop
    for t_idx in range(1, T+1):
        t_start = t_coordinate[t_idx-1]
        t_end = t_coordinate[t_idx]
        t_current = t_start
        
        # Initialize variables for the current time step
        density = density_pred[:, t_idx-1, :].clone()
        velocity = Vx_pred[:, t_idx-1, :].clone()
        pressure = pressure_pred[:, t_idx-1, :].clone()
        epsilon = pressure / (5/3 - 1)
        
        while t_current < t_end:
            dt = min(dt_internal, t_end - t_current)
            density, velocity, pressure, epsilon = time_step(density, velocity, pressure, epsilon, dx, dt, eta, zeta)
            t_current += dt
        
        # Store results at the specified time step
        Vx_pred[:, t_idx, :] = velocity
        density_pred[:, t_idx, :] = density
        pressure_pred[:, t_idx, :] = pressure
    
    return Vx_pred.numpy(), density_pred.numpy(), pressure_pred.numpy()
