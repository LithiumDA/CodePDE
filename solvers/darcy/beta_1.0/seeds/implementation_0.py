
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

def _compute_flux(u_padded, a, h):
    """Compute the flux terms for the Darcy equation discretization."""
    # Extract interior points of a (excluding boundaries)
    a_center = a[1:-1, 1:-1]
    
    # Compute averaged coefficients at cell faces
    a_east = 0.5 * (a_center + a[2:, 1:-1])  # East face
    a_west = 0.5 * (a_center + a[:-2, 1:-1])  # West face
    a_north = 0.5 * (a_center + a[1:-1, 2:])  # North face
    a_south = 0.5 * (a_center + a[1:-1, :-2])  # South face
    
    # X-direction fluxes
    u_center_x = u_padded[1:-1, 1:-1]
    flux_east = a_east * (u_padded[2:, 1:-1] - u_center_x) / h
    flux_west = a_west * (u_center_x - u_padded[:-2, 1:-1]) / h
    flux_x = (flux_east - flux_west) / h
    
    # Y-direction fluxes
    u_center_y = u_padded[1:-1, 1:-1]
    flux_north = a_north * (u_padded[1:-1, 2:] - u_center_y) / h
    flux_south = a_south * (u_center_y - u_padded[1:-1, :-2]) / h
    flux_y = (flux_north - flux_south) / h
    
    return flux_x + flux_y

@jax.jit
def solve_single(a):
    """Solve the Darcy equation for a single coefficient field (shape [N, N])."""
    N = a.shape[0]
    h = 1.0 / (N - 1)
    rhs = jnp.ones((N-2, N-2))  # Interior RHS is 1 (original PDE scaled correctly)
    u0 = jnp.zeros_like(rhs)
    
    def matvec(u_flat):
        """Matrix-vector product for the linear operator."""
        u = u_flat.reshape(N-2, N-2)
        u_padded = jnp.pad(u, ((1,1), (1,1)), mode='constant')  # Pad with zeros
        total_flux = _compute_flux(u_padded, a, h)
        return (-total_flux).ravel()  # A = -div(a grad)
    
    # Solve using Conjugate Gradient with increased maxiter for convergence
    u_flat, _ = jax.scipy.sparse.linalg.cg(matvec, rhs.ravel(), x0=u0.ravel(), maxiter=2000)
    u_interior = u_flat.reshape(N-2, N-2)
    
    # Pad with boundary zeros to get full solution
    return jnp.pad(u_interior, ((1,1), (1,1)), mode='constant')

def solver(a):
    """Batch solver for the Darcy equation using JAX acceleration."""
    a_jax = jnp.asarray(a)  # Convert input to JAX array
    # Vectorize over the batch dimension using vmap
    batch_solve = vmap(solve_single)
    solutions = batch_solve(a_jax)
    return np.asarray(solutions)  # Convert back to numpy array for output
