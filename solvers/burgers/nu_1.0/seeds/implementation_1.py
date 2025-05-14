
import jax.numpy as jnp
from jax import vmap, lax

def step_forward(u, dt, dx, nu):
    """Perform a single time step using explicit Euler method."""
    du_dx = (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2 * dx)
    u_xx = (jnp.roll(u, -1, axis=-1) - 2*u + jnp.roll(u, 1, axis=-1)) / (dx**2)
    advection = -u * du_dx
    diffusion = nu * u_xx
    return u + dt * (advection + diffusion)

def solver(u0_batch, t_coordinate, nu):
    """Solves the Burgers' equation using explicit Euler with JAX acceleration."""
    batch_size, N = u0_batch.shape
    dx = 1.0 / N
    dt = (dx**2) / (2 * nu) * 0.5  # Stable time step
    
    solutions = jnp.zeros((batch_size, len(t_coordinate), N))
    solutions = solutions.at[:, 0, :].set(u0_batch)
    current_u = u0_batch
    
    for i in range(1, len(t_coordinate)):
        target_time = t_coordinate[i]
        delta_t = target_time - t_coordinate[i-1]
        
        # Compute full steps and remaining time
        steps = int(delta_t // dt)
        remaining = delta_t - steps * dt
        
        # Vectorize internal steps using JAX scan
        def scan_step(u, _):
            return step_forward(u, dt, dx, nu), None
        
        current_u, _ = lax.scan(scan_step, current_u, None, length=steps)
        
        # Apply remaining step if needed
        if remaining > 1e-10:
            current_u = step_forward(current_u, remaining, dx, nu)
        
        solutions = solutions.at[:, i, :].set(current_u)
    
    return solutions.block_until_ready().copy()
