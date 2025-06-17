import numpy as np

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
    # Try to auto-detect and use the best available backend
    try:
        import torch
        use_torch = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using PyTorch backend on {device}")
    except ImportError:
        use_torch = False
        try:
            import jax
            import jax.numpy as jnp
            use_jax = True
            print(f"Using JAX backend")
        except ImportError:
            use_jax = False
            print(f"Using NumPy backend")
    
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1  # Number of time steps to output
    
    # Spatial discretization
    dx = 1.0 / N  # Spatial step size assuming domain [0, 1]
    
    # For beta = 0.1, optimize time step selection
    # Determine a suitable internal time step based on CFL condition
    # Using a safety factor of 0.8 for stability
    cfl_factor = 0.8
    dt_cfl = cfl_factor * dx / beta
    
    # Determine the number of internal time steps needed
    total_time = t_coordinate[-1]
    n_internal_steps = int(np.ceil(total_time / dt_cfl))
    dt_internal = total_time / n_internal_steps
    
    print(f"Spatial step size (dx): {dx:.6f}")
    print(f"Internal time step (dt): {dt_internal:.6f}")
    print(f"Number of internal time steps: {n_internal_steps}")
    print(f"CFL number: {beta * dt_internal / dx:.6f}")
    
    # Initialize solutions array
    solutions = np.zeros((batch_size, T+1, N))
    solutions[:, 0, :] = u0_batch  # Set initial condition
    
    # Precompute coefficient for time stepping
    coeff = beta * dt_internal / (2 * dx)
    
    # Set up current state based on backend
    if use_torch:
        u_current = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    elif use_jax:
        u_current = jnp.array(u0_batch, dtype=jnp.float32)
        
        # JIT-compile the time stepping function for better performance
        @jax.jit
        def time_step(u):
            # Central difference with periodic boundary conditions
            u_prev = jnp.roll(u, 1, axis=1)  # Shift right (i-1)
            u_next = jnp.roll(u, -1, axis=1)  # Shift left (i+1)
            return u - coeff * (u_next - u_prev)
    else:  # numpy
        u_current = u0_batch.copy()
    
    # Time-stepping loop
    current_time = 0.0
    next_output_idx = 1  # Index for the next output time
    
    for step in range(n_internal_steps):
        # Calculate next time
        current_time += dt_internal
        
        # Apply time step based on backend
        if use_torch:
            # Central difference with periodic boundary conditions
            u_prev = torch.roll(u_current, 1, dims=1)  # Shift right (i-1)
            u_next = torch.roll(u_current, -1, dims=1)  # Shift left (i+1)
            u_current = u_current - coeff * (u_next - u_prev)
        elif use_jax:
            u_current = time_step(u_current)
        else:  # numpy
            # Central difference with periodic boundary conditions
            u_prev = np.roll(u_current, 1, axis=1)  # Shift right (i-1)
            u_next = np.roll(u_current, -1, axis=1)  # Shift left (i+1)
            u_current = u_current - coeff * (u_next - u_prev)
        
        # Check if we need to store the solution at this time
        while next_output_idx <= T and current_time >= t_coordinate[next_output_idx]:
            # Convert to numpy if needed
            if use_torch:
                solutions[:, next_output_idx, :] = u_current.cpu().numpy()
            elif use_jax:
                solutions[:, next_output_idx, :] = np.array(u_current)
            else:  # numpy
                solutions[:, next_output_idx, :] = u_current
            
            # Print progress periodically
            if next_output_idx % 10 == 0 or next_output_idx == T:
                print(f"Stored solution at time {t_coordinate[next_output_idx]:.4f} (Step {step+1}/{n_internal_steps})")
            
            next_output_idx += 1
    
    return solutions