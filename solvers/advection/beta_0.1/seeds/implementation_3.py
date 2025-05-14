
import numpy as np

def lax_wendroff_step(u, dx, dt, beta):
    """
    Performs one step of the Lax-Wendroff method for the advection equation.
    
    Args:
        u (np.ndarray): Current solution [batch_size, N]
        dx (float): Spatial step size
        dt (float): Time step size
        beta (float): Advection speed
        
    Returns:
        np.ndarray: Updated solution after one time step
    """
    batch_size, N = u.shape
    
    # Calculate CFL number
    c = beta * dt / dx
    
    # Create new array for the updated solution
    u_new = np.zeros_like(u)
    
    # Lax-Wendroff scheme for interior points
    for i in range(N):
        i_minus_1 = (i - 1) % N  # Periodic boundary
        i_plus_1 = (i + 1) % N   # Periodic boundary
        
        # Lax-Wendroff formula
        u_new[:, i] = u[:, i] - 0.5 * c * (u[:, i_plus_1] - u[:, i_minus_1]) + \
                     0.5 * c**2 * (u[:, i_plus_1] - 2 * u[:, i] + u[:, i_minus_1])
    
    return u_new

def solver(u0_batch, t_coordinate, beta):
    """Solves the Advection equation for all times in t_coordinate.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N], 
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. 
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        beta (float): Constant advection speed.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1
    
    # Initialize solutions array
    solutions = np.zeros((batch_size, T+1, N))
    solutions[:, 0, :] = u0_batch
    
    # Calculate dx (assuming domain is [0, 1])
    dx = 1.0 / N
    
    # Determine a suitable internal time step for stability
    # For Lax-Wendroff, CFL < 1 is required for stability
    # Since beta = 0.1, we can use a relatively large time step
    cfl_target = 0.8  # Target CFL number (conservative)
    dt_internal = cfl_target * dx / beta
    
    print(f"Spatial points: {N}, Time points: {T+1}")
    print(f"dx: {dx:.6f}, Internal dt: {dt_internal:.6f}")
    print(f"CFL number: {beta * dt_internal / dx:.4f}")
    
    # Current solution and time
    u_current = u0_batch.copy()
    t_current = 0.0
    
    # Index for the next time point to capture
    next_t_idx = 1
    
    # Main time-stepping loop
    while next_t_idx <= T:
        # Determine the next time step
        if t_current + dt_internal > t_coordinate[next_t_idx]:
            # If the next internal step would overshoot the target time,
            # adjust dt to hit the target time exactly
            dt = t_coordinate[next_t_idx] - t_current
        else:
            dt = dt_internal
        
        # Perform a time step
        u_current = lax_wendroff_step(u_current, dx, dt, beta)
        t_current += dt
        
        # Check if we've reached or passed a target time point
        if abs(t_current - t_coordinate[next_t_idx]) < 1e-10 or t_current > t_coordinate[next_t_idx]:
            solutions[:, next_t_idx, :] = u_current
            print(f"Captured solution at t = {t_coordinate[next_t_idx]:.4f}")
            next_t_idx += 1
    
    return solutions
