
import jax
import jax.numpy as jnp
from jax import vmap

def solver(u0_batch, t_coordinate, beta):
    """Solves the Advection equation for all times in t_coordinate using the method of characteristics with linear interpolation."""
    u0_batch = jnp.asarray(u0_batch)
    batch_size, N = u0_batch.shape
    T_plus_1 = len(t_coordinate)
    solutions = jnp.zeros((batch_size, T_plus_1, N))
    solutions = solutions.at[:, 0, :].set(u0_batch)
    
    # Compute shifts for all time steps except the first
    times = t_coordinate[1:]
    s_shifts = beta * times * N  # s_shift = beta * t * N
    
    # Define the function to compute the shifted array for a single time step
    def compute_shift(u0, s_shift):
        j = jnp.arange(N)
        shifted_j_float = (j - s_shift) % N
        int_part = jnp.floor(shifted_j_float).astype(jnp.int32)
        next_index = (int_part + 1) % N
        frac = shifted_j_float - int_part
        indices = jnp.stack([int_part, next_index], axis=-1)  # shape (N, 2)
        weights = jnp.stack([(1 - frac), frac], axis=-1)  # shape (N, 2)
        
        # Expand weights to have batch dimension
        weights = weights[None, :, :]  # shape (1, N, 2)
        
        # Select values and interpolate
        selected = u0[..., indices]  # shape (batch_size, N, 2)
        weighted = selected * weights  # shape (batch_size, N, 2)
        shifted = weighted.sum(axis=-1)  # shape (batch_size, N)
        return shifted
    
    # Vectorize over time steps
    compute_shift_vmap = vmap(compute_shift, in_axes=(None, 0))
    shifted_arrays = compute_shift_vmap(u0_batch, s_shifts)
    
    # Reshape to (batch_size, T, N)
    shifted_arrays = shifted_arrays.swapaxes(0, 1)
    
    # Assign to the solutions array
    solutions = solutions.at[:, 1:, :].set(shifted_arrays)
    
    # Convert to NumPy array using jax.device_get()
    return jax.device_get(solutions)
