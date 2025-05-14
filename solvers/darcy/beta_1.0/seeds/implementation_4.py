
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, device_get

def solver(a):
    """Solve the Darcy equation using JAX and conjugate gradient method.
    
    Args:
        a (np.ndarray): Shape [batch_size, N, N], the coefficient of the Darcy equation.
        
    Returns:
        solutions (np.ndarray): Shape [batch_size, N, N].
    """
    a = jnp.array(a)
    B, N, _ = a.shape
    M = N - 2
    h = 1.0 / (N - 1)
    rhs = -h**2 * jnp.ones((B, M, M))
    
    # Compute edge coefficients for x and y directions
    a_x = (a[:, :-1, :] + a[:, 1:, :]) / 2.0  # Vertical edges (between rows)
    a_y = (a[:, :, :-1] + a[:, :, 1:]) / 2.0  # Horizontal edges (between columns)
    
    def matvec(x_grid):
        """Matrix-vector product for the discretized system."""
        # North contribution (downward direction)
        a_x_north = a_x[:, 1:M+1, 1:M+1]
        x_north = jnp.roll(x_grid, shift=-1, axis=1)
        x_north = x_north.at[:, -1, :].set(0.0)
        north = a_x_north * x_north

        # South contribution (upward direction)
        a_x_south = a_x[:, 0:M, 1:M+1]
        x_south = jnp.roll(x_grid, shift=1, axis=1)
        x_south = x_south.at[:, 0, :].set(0.0)
        south = a_x_south * x_south

        # East contribution (right direction)
        a_y_east = a_y[:, 1:M+1, 1:M+1]
        x_east = jnp.roll(x_grid, shift=-1, axis=2)
        x_east = x_east.at[:, :, -1].set(0.0)
        east = a_y_east * x_east

        # West contribution (left direction)
        a_y_west = a_y[:, 1:M+1, 0:M]
        x_west = jnp.roll(x_grid, shift=1, axis=2)
        x_west = x_west.at[:, :, 0].set(0.0)
        west = a_y_west * x_west

        total_contrib = north + south + east + west

        # Compute diagonal terms
        diag_coeff = a_x_north + a_x_south + a_y_east + a_y_west
        diag_term = diag_coeff * x_grid

        return total_contrib - diag_term

    def conjugate_gradient(rhs, maxiter=1000, tol=1e-6):
        """Batched conjugate gradient method."""
        x = jnp.zeros_like(rhs)
        r = rhs - matvec(x)
        p = r
        rsold = jnp.sum(r * r, axis=(1, 2))
        for _ in range(maxiter):
            Ap = matvec(p)
            alpha = (rsold / jnp.sum(p * Ap, axis=(1, 2))).reshape(B, 1, 1)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = jnp.sum(r * r, axis=(1, 2))
            beta = (rsnew / rsold).reshape(B, 1, 1)
            p = r + beta * p
            rsold = rsnew
            # Early termination if converged
            if jnp.all(rsnew < tol * jnp.sum(rhs * rhs, axis=(1, 2))):
                break
        return x

    # Solve the system using conjugate gradient
    solution = conjugate_gradient(rhs)
    # Pad with zeros on the boundaries
    full_solution = jnp.zeros((B, N, N))
    full_solution = full_solution.at[:, 1:-1, 1:-1].set(solution)
    full_solution.block_until_ready()  # Ensure computation completes
    return device_get(full_solution)  # Use jax.device_get() for universal compatibility
