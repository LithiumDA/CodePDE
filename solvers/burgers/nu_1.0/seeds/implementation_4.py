
import jax
import jax.numpy as jnp
from jax import lax

def solver(u0_batch, t_coordinate, nu):
    """Solves the Burgers' equation for all times in t_coordinate using JAX for acceleration."""
    # Convert inputs to JAX arrays
    u0 = jnp.array(u0_batch)
    t = jnp.array(t_coordinate)
    batch_size, N = u0.shape

    # Spatial discretization parameters
    dx = 1.0 / N
    dt_int_max = (dx ** 2) / (2.0 * nu)  # Stability condition for viscous term

    solutions = [u0]  # List to store solutions at each time step
    current_u = u0

    # Define the step function outside loops to avoid recompilation
    def body(carry, x_dt_step):
        current_u = carry
        dt_step = x_dt_step

        # Compute flux term F = u²/2 and its derivatives
        F = (current_u ** 2) / 2.0
        F_plus = jnp.roll(F, shift=-1, axis=1)
        F_minus = jnp.roll(F, shift=1, axis=1)
        flux_term = (F_plus - F_minus) / (2.0 * dx)

        # Compute viscous term (second derivative of u)
        u_plus = jnp.roll(current_u, shift=-1, axis=1)
        u_minus = jnp.roll(current_u, shift=1, axis=1)
        viscous_term = (u_plus - 2.0 * current_u + u_minus) / (dx ** 2)

        # Update using explicit Euler method
        next_u = current_u - dt_step * flux_term + nu * dt_step * viscous_term
        return next_u, None

    # Iterate over each time interval
    for i in range(1, len(t)):
        prev_t = t[i-1]
        current_t = t[i]
        delta_t_total = current_t - prev_t

        # Compute number of internal steps and step size
        m = int(jnp.ceil(delta_t_total / dt_int_max))
        dt_step_i = delta_t_total / m

        # Prepare step sizes for this interval
        xs = jnp.full(m, dt_step_i)

        # Perform internal steps using JAX scan
        current_u, _ = lax.scan(body, current_u, xs)

        solutions.append(current_u)

    # Stack solutions and convert to NumPy array
    solutions_array = jnp.stack(solutions, axis=1)
    
    # Convert JAX array to NumPy array using device_get for compatibility
    return jax.device_get(solutions_array.block_until_ready())
