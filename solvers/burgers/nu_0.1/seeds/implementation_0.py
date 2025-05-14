
import numpy as np
import torch
import math

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
    # Choose device: GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Convert numpy array into torch tensor of type float32 and move to the device.
    u = torch.tensor(u0_batch, dtype=torch.float32, device=device)

    # Get batch_size and number of spatial grid points.
    batch_size, N = u.shape
    
    # Spatial domain [0,1] with periodic condition.
    dx = 1.0 / N

    # Number of output time snapshots.
    T_plus_1 = t_coordinate.shape[0]
  
    # Allocate tensor to store the solution: shape [batch_size, T+1, N]
    solutions = torch.zeros((batch_size, T_plus_1, N), dtype=torch.float32, device=device)
    solutions[:, 0, :] = u.clone()  # set initial condition

    # Define the RHS of the PDE using finite differences.
    def compute_rhs(u):
        """
        Compute the right-hand side of the Burgers' equation:
            u_t = - d/dx(u^2/2) + nu*d2u/dx2
        using central finite differences with periodic boundary conditions.
        
        u: tensor of shape [batch_size, N]
        Returns: tensor of shape [batch_size, N]
        """
        # Compute flux: (u^2)/2
        flux = 0.5 * u**2

        # Periodic shift indices:
        u_right  = torch.roll(u, shifts=-1, dims=1)  # u_{j+1}
        u_left   = torch.roll(u, shifts=1, dims=1)    # u_{j-1}
        flux_right = torch.roll(flux, shifts=-1, dims=1)
        flux_left  = torch.roll(flux, shifts=1, dims=1)
        
        # First derivative using central differences for flux derivative.
        dflux_dx = (flux_right - flux_left) / (2.0 * dx)
        
        # Second derivative (diffusive term) using central differences.
        d2u_dx2 = (u_right - 2*u + u_left) / (dx * dx)
        
        # Combine convective and diffusive parts.
        rhs = -dflux_dx + nu * d2u_dx2
        return rhs

    # Time integrator: second-order Runge-Kutta (RK2)
    def rk2_step(u, dt):
        """
        Performs one RK2 step for the PDE.
        
        Args:
            u: current solution [batch_size, N]
            dt: time step size (scalar)
        Returns: updated solution after one RK2 step [batch_size, N]
        """
        k1 = compute_rhs(u)
        u_temp = u + dt * k1
        k2 = compute_rhs(u_temp)
        u_next = u + (dt / 2.0) * (k1 + k2)
        return u_next

    # Loop over provided output time snapshots.
    for ti in range(1, T_plus_1):
        # Determine the total time interval for the current output.
        dt_interval = t_coordinate[ti] - t_coordinate[ti - 1]
        
        # Estimate the maximum absolute value of u (for CFL condition)
        u_max = torch.max(torch.abs(u)).item() + 1e-6  # add a small number to avoid zero
        
        # Compute stability limits.
        dt_stable_diff = dx**2 / (2 * nu)
        dt_stable_conv = dx / u_max
        # Safety factor to ensure additional stability.
        safety = 0.5
        dt_safe = safety * min(dt_stable_diff, dt_stable_conv)
        
        # Determine number of internal sub-steps required.
        sub_steps = max(math.ceil(dt_interval / dt_safe), 1)
        dt_internal = dt_interval / sub_steps
        
        print(f"Processing time interval [{t_coordinate[ti-1]:.4f}, {t_coordinate[ti]:.4f}] "
              f"with {sub_steps} sub-steps, internal dt = {dt_internal:.8f}")
        
        # Perform internal timestepping.
        for step in range(sub_steps):
            u = rk2_step(u, dt_internal)
            # Check for non-finite values to debug potential issues.
            if not torch.isfinite(u).all():
                raise ValueError("Non-finite values encountered in the simulation.")
        
        # Store the result at the current output time.
        solutions[:, ti, :] = u.clone()
        print(f"Stored solution at t = {t_coordinate[ti]:.4f}")

    # Convert final solution back to numpy array.
    return solutions.cpu().numpy()
