
import numpy as np

def lax_wendroff_step(u, beta, dx, dt):
    """
    Perform one step of the Lax-Wendroff scheme for the advection equation.
    
    Args:
        u (np.ndarray): Current solution [batch_size, N]
        beta (float): Advection speed
        dx (float): Spatial step size
        dt (float): Time step size
        
    Returns:
        np.ndarray: Solution after one time step [batch_size, N]
    """
    # Calculate coefficients
    c = beta * dt / dx
    c2 = c * c
    
    # Shift arrays for periodic boundary conditions
    u_minus = np.roll(u, 1, axis=1)  # u_{j-1}
    u_plus = np.roll(u, -1, axis=1)  # u_{j+1}
    
    # Lax-Wendroff update
    u_new = u - 0.5 * c * (u_plus - u_minus) + 0.5 * c2 * (u_plus - 2*u + u_minus)
    
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
    
    # Initialize solution array
    solutions = np.zeros((batch_size, T+1, N))
    solutions[:, 0, :] = u0_batch
    
    # Calculate spatial step size (assuming domain [0, 1])
    dx = 1.0 / N
    
    # For stability, we need to ensure CFL condition: |beta|*dt/dx <= 1
    # Let's use a safety factor of 0.8
    safety_factor = 0.8
    dt_stable = safety_factor * dx / abs(beta)
    
    # Print some information about the simulation
    print(f"Spatial points: {N}, Time points: {T+1}")
    print(f"Domain dx: {dx:.6f}, Stable dt: {dt_stable:.6f}")
    
    # Current simulation time
    current_time = 0.0
    
    # Current solution state
    current_u = u0_batch.copy()
    
    # For each requested output time point
    for i in range(1, T+1):
        target_time = t_coordinate[i]
        
        # Determine number of internal steps needed
        steps_needed = max(1, int(np.ceil((target_time - current_time) / dt_stable)))
        internal_dt = (target_time - current_time) / steps_needed
        
        print(f"Time {target_time:.4f}: Using {steps_needed} internal steps with dt={internal_dt:.6f}")
        
        # Perform internal steps
        for _ in range(steps_needed):
            current_u = lax_wendroff_step(current_u, beta, dx, internal_dt)
        
        # Store the solution at the requested time point
        solutions[:, i, :] = current_u
        current_time = target_time
    
    return solutions
