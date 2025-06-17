
import numpy as np

def compute_spatial_derivative(u_batch, order=1):
    """
    Compute spatial derivatives using Fourier spectral method.
    
    Args:
        u_batch: Array of shape [batch_size, N]
        order: Order of the derivative (1 for first, 2 for second)
    
    Returns:
        Derivative of shape [batch_size, N]
    """
    batch_size, N = u_batch.shape
    
    # Wavenumbers
    k = np.fft.fftfreq(N, 1.0/N) * 2 * np.pi
    
    # Compute FFT
    u_hat = np.fft.fft(u_batch, axis=1)
    
    # Compute derivative in Fourier space
    if order == 1:
        du_hat = 1j * k * u_hat
    elif order == 2:
        du_hat = -k**2 * u_hat
    else:
        raise ValueError(f"Derivative order {order} not supported")
    
    # Transform back to physical space
    du = np.real(np.fft.ifft(du_hat, axis=1))
    
    return du

def rhs_burgers(u_batch, nu):
    """
    Compute the right-hand side of Burgers' equation: -u*u_x + nu*u_xx
    
    Args:
        u_batch: Array of shape [batch_size, N]
        nu: Viscosity coefficient
    
    Returns:
        Time derivative of shape [batch_size, N]
    """
    u_x = compute_spatial_derivative(u_batch, order=1)
    u_xx = compute_spatial_derivative(u_batch, order=2)
    
    # Compute -u*u_x + nu*u_xx
    return -u_batch * u_x + nu * u_xx

def rk4_step(u_batch, dt, nu):
    """
    Perform a single 4th-order Runge-Kutta step.
    
    Args:
        u_batch: Current solution of shape [batch_size, N]
        dt: Time step
        nu: Viscosity coefficient
    
    Returns:
        Updated solution of shape [batch_size, N]
    """
    k1 = rhs_burgers(u_batch, nu)
    k2 = rhs_burgers(u_batch + 0.5 * dt * k1, nu)
    k3 = rhs_burgers(u_batch + 0.5 * dt * k2, nu)
    k4 = rhs_burgers(u_batch + dt * k3, nu)
    
    return u_batch + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def estimate_stable_dt(u_batch, dx, nu, cfl_safety=0.5):
    """
    Estimate a stable time step based on CFL condition.
    
    Args:
        u_batch: Current solution of shape [batch_size, N]
        dx: Spatial step size
        nu: Viscosity coefficient
        cfl_safety: Safety factor for CFL condition
    
    Returns:
        Stable time step
    """
    # Maximum wave speed (conservative estimate)
    max_u = np.max(np.abs(u_batch))
    
    # CFL condition for advection
    dt_adv = dx / max_u if max_u > 0 else np.inf
    
    # CFL condition for diffusion
    dt_diff = 0.5 * dx**2 / nu if nu > 0 else np.inf
    
    # Take the most restrictive condition
    dt = min(dt_adv, dt_diff) * cfl_safety
    
    return dt

def solver(u0_batch, t_coordinate, nu):
    """Solves the Burgers' equation for all times in t_coordinate.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N], 
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. 
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        nu (float): Viscosity coefficient.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1
    
    # Initialize solution array
    solutions = np.zeros((batch_size, T+1, N))
    solutions[:, 0, :] = u0_batch
    
    # Spatial step size (assuming domain [0, 1])
    dx = 1.0 / N
    
    # Current solution and time
    u_current = u0_batch.copy()
    t_current = 0.0
    
    print(f"Starting simulation with nu={nu}, grid points={N}, batch size={batch_size}")
    
    # For each output time step
    for i in range(1, T+1):
        t_target = t_coordinate[i]
        
        # Integrate from t_current to t_target using adaptive time stepping
        while t_current < t_target:
            # Estimate stable time step
            dt_stable = estimate_stable_dt(u_current, dx, nu)
            
            # Ensure we don't overshoot the target time
            dt = min(dt_stable, t_target - t_current)
            
            # Update solution using RK4
            u_current = rk4_step(u_current, dt, nu)
            t_current += dt
            
            # Print progress occasionally (not for every small step)
            if i % max(1, T//5) == 0 and dt == dt_stable:
                print(f"Time {t_current:.6f}/{t_target:.6f}, dt={dt:.6e}, max|u|={np.max(np.abs(u_current)):.4f}")
        
        # Store the solution at the required time point
        solutions[:, i, :] = u_current
        
        # Print progress for each major time step
        if i % max(1, T//10) == 0:
            print(f"Completed time step {i}/{T}, t={t_target:.4f}")
    
    print("Simulation completed")
    return solutions
