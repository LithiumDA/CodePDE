
import jax
import jax.numpy as jnp
from jax import lax

def solver(u0_batch, t_coordinate, nu):
    # Enable double precision (requires JAX_ENABLE_X64=1 in environment)
    jax.config.update("jax_enable_x64", True)
    
    B, N = u0_batch.shape
    dx = 1.0 / N
    k = 2 * jnp.pi * jnp.fft.fftfreq(N, d=dx).astype(u0_batch.dtype)
    
    # Stability criteria: diffusion (dt_diffusion) and safety factor (0.1)
    dt_diffusion = (dx ** 2) / (2 * nu)
    dt = 0.1 * dt_diffusion  # Smaller step for stability
    
    T = len(t_coordinate)
    solutions = jnp.zeros((B, T, N), dtype=u0_batch.dtype)
    solutions = solutions.at[:, 0, :].set(u0_batch)
    
    current_u = u0_batch
    current_time = 0.0
    
    for i in range(1, T):
        target_time = t_coordinate[i]
        delta_t = target_time - current_time
        n_steps = jnp.floor(delta_t / dt).astype(jnp.int32)
        remaining = delta_t % dt
        
        def step_body(s, u):
            u_hat = jnp.fft.fft(u, axis=1)
            u_x_hat = 1j * k * u_hat
            u_x = jnp.fft.ifft(u_x_hat, axis=1).real
            u_xx_hat = -k**2 * u_hat
            u_xx = jnp.fft.ifft(u_xx_hat, axis=1).real
            rhs = -u * u_x + nu * u_xx
            return u + dt * rhs
        
        current_u = lax.fori_loop(0, n_steps, step_body, current_u)
        
        if remaining > 0:
            u_hat = jnp.fft.fft(current_u, axis=1)
            u_x_hat = 1j * k * u_hat
            u_x = jnp.fft.ifft(u_x_hat, axis=1).real
            u_xx_hat = -k**2 * u_hat
            u_xx = jnp.fft.ifft(u_xx_hat, axis=1).real
            rhs = -current_u * u_x + nu * u_xx
            current_u = current_u + remaining * rhs
        
        # Validate solution
        if jnp.isnan(current_u).any():
            raise ValueError("NaN detected during time stepping")
        
        current_time = target_time
        solutions = solutions.at[:, i, :].set(current_u)
    
    return solutions
