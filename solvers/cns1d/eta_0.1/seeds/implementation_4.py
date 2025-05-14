
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

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
    # Try to use GPU if available
    try:
        jax.config.update('jax_platform_name', 'gpu')
        print("JAX is using GPU")
    except:
        print("JAX is using CPU")
    
    # Convert numpy arrays to JAX arrays
    Vx0_jax = jnp.array(Vx0)
    density0_jax = jnp.array(density0)
    pressure0_jax = jnp.array(pressure0)
    t_coordinate_jax = jnp.array(t_coordinate)
    
    # Get dimensions
    batch_size, N = Vx0_jax.shape
    T = len(t_coordinate_jax) - 1
    
    # Initialize output arrays
    Vx_pred = jnp.zeros((batch_size, T+1, N))
    density_pred = jnp.zeros((batch_size, T+1, N))
    pressure_pred = jnp.zeros((batch_size, T+1, N))
    
    # Set initial conditions
    Vx_pred = Vx_pred.at[:, 0, :].set(Vx0_jax)
    density_pred = density_pred.at[:, 0, :].set(density0_jax)
    pressure_pred = pressure_pred.at[:, 0, :].set(pressure0_jax)
    
    # Define gamma (adiabatic index)
    gamma = 5/3
    
    # Compute spatial step (assuming uniform grid with periodic boundary)
    dx = 1.0 / N
    
    # Simple fixed time step - safer approach
    # We'll use a fixed small time step based on the problem parameters
    # This avoids potential NaN issues in adaptive time stepping
    fixed_dt = 1e-6  # Very conservative fixed time step
    
    # Loop through time steps
    for t_idx in range(1, T+1):
        # Current time and target time
        t_current = float(t_coordinate_jax[t_idx-1])
        t_target = float(t_coordinate_jax[t_idx])
        dt_target = t_target - t_current
        
        # Get current state
        vx_current = Vx_pred[:, t_idx-1, :]
        density_current = density_pred[:, t_idx-1, :]
        pressure_current = pressure_pred[:, t_idx-1, :]
        
        # Determine number of sub-steps needed with fixed time step
        n_steps = max(1, int(np.ceil(dt_target / fixed_dt)))
        sub_dt = dt_target / n_steps
        
        print(f"Time step {t_idx}/{T}: Using {n_steps} sub-steps with dt={sub_dt:.6f}")
        
        # Define a function for a single step
        def single_step(vx, density, pressure):
            """Single step of the numerical scheme"""
            # Calculate internal energy
            internal_energy = pressure / (gamma - 1)
            
            # Calculate total energy (internal + kinetic)
            total_energy = internal_energy + 0.5 * density * vx**2
            
            # Compute viscous stress tensor coefficient
            visc_coef = zeta + (4/3) * eta
            
            # Use a simpler first-order upwind scheme for stability
            # Density update (continuity equation)
            flux_density = density * vx
            flux_density_left = jnp.roll(flux_density, 1, axis=1)
            flux_density_right = jnp.roll(flux_density, -1, axis=1)
            
            # Upwind differencing for density
            flux_diff = jnp.where(vx > 0, 
                                 (flux_density - flux_density_left) / dx,
                                 (flux_density_right - flux_density) / dx)
            density_new = density - sub_dt * flux_diff
            
            # Momentum equation
            # Central differences for pressure and viscous terms
            pressure_grad = (jnp.roll(pressure, -1, axis=1) - jnp.roll(pressure, 1, axis=1)) / (2*dx)
            
            # Second derivative of velocity (for viscous term)
            d2vx_dx2 = (jnp.roll(vx, -1, axis=1) - 2*vx + jnp.roll(vx, 1, axis=1)) / (dx**2)
            
            # First derivative of velocity
            dvx_dx = (jnp.roll(vx, -1, axis=1) - jnp.roll(vx, 1, axis=1)) / (2*dx)
            
            # Gradient of dvx_dx using central differences
            d_dvx_dx = (jnp.roll(dvx_dx, -1, axis=1) - jnp.roll(dvx_dx, 1, axis=1)) / (2*dx)
            
            # Viscous term
            viscous_term = eta * d2vx_dx2 + (zeta + eta/3) * d_dvx_dx
            
            # Upwind differencing for momentum advection
            momentum = density * vx
            momentum_left = jnp.roll(momentum, 1, axis=1)
            momentum_right = jnp.roll(momentum, -1, axis=1)
            
            # Momentum flux due to advection
            momentum_flux = jnp.where(vx > 0,
                                     vx * (momentum - momentum_left) / dx,
                                     vx * (momentum_right - momentum) / dx)
            
            # Total momentum update
            momentum_new = momentum - sub_dt * (momentum_flux + pressure_grad - viscous_term)
            
            # Energy equation
            # Energy flux
            energy_flux = (total_energy + pressure) * vx - vx * visc_coef * dvx_dx
            energy_flux_left = jnp.roll(energy_flux, 1, axis=1)
            energy_flux_right = jnp.roll(energy_flux, -1, axis=1)
            
            # Upwind differencing for energy
            energy_flux_diff = jnp.where(vx > 0,
                                        (energy_flux - energy_flux_left) / dx,
                                        (energy_flux_right - energy_flux) / dx)
            
            # Energy update
            total_energy_new = total_energy - sub_dt * energy_flux_diff
            
            # Recover new velocity from momentum and density
            vx_new = momentum_new / jnp.maximum(density_new, 1e-10)
            
            # Recover pressure from total energy
            kinetic_energy_new = 0.5 * density_new * vx_new**2
            internal_energy_new = jnp.maximum(total_energy_new - kinetic_energy_new, 1e-10)
            pressure_new = internal_energy_new * (gamma - 1)
            
            # Ensure physical constraints
            density_new = jnp.maximum(density_new, 1e-10)
            pressure_new = jnp.maximum(pressure_new, 1e-10)
            
            return vx_new, density_new, pressure_new
        
        # JIT compile the single step function
        jit_step = jax.jit(single_step)
        
        # Perform sub-steps
        for _ in range(n_steps):
            vx_current, density_current, pressure_current = jit_step(
                vx_current, density_current, pressure_current
            )
        
        # Store results
        Vx_pred = Vx_pred.at[:, t_idx, :].set(vx_current)
        density_pred = density_pred.at[:, t_idx, :].set(density_current)
        pressure_pred = pressure_pred.at[:, t_idx, :].set(pressure_current)
    
    # Convert back to numpy for output
    return np.array(Vx_pred), np.array(density_pred), np.array(pressure_pred)
