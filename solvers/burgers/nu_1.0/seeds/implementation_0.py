
import numpy as np
import torch

def solve_burgers_step(u, dt, dx, nu):
    """
    Computes one time step update using explicit Euler for the Burgers' equation.
    
    Args:
        u (torch.Tensor): Current solution of shape [batch_size, N].
        dt (float): Time step size.
        dx (float): Spatial grid spacing.
        nu (float): Viscosity.
        
    Returns:
        u_new (torch.Tensor): Updated solution of shape [batch_size, N].
    """
    # Compute the flux f = 0.5*u^2
    flux = 0.5 * u * u
    
    # Compute the spatial derivative of the flux using central differences.
    # Using torch.roll to account for periodic boundary conditions.
    flux_x = (torch.roll(flux, shifts=-1, dims=1) - torch.roll(flux, shifts=1, dims=1)) / (2 * dx)
    
    # Compute the second derivative u_xx for the diffusion term.
    u_xx = (torch.roll(u, shifts=-1, dims=1) - 2*u + torch.roll(u, shifts=1, dims=1)) / (dx*dx)
    
    # Explicit Euler update: u_new = u - dt*(flux derivative) + dt*nu*(u_xx)
    u_new = u - dt * flux_x + dt * nu * u_xx
    
    return u_new


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
    # Print initial debug info.
    print("Starting solver for Burgers' equation")
    
    # Determine device: use GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # Convert the initial condition to a torch tensor with float type.
    # u0_batch: shape [batch_size, N]
    u = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    
    batch_size, N = u.shape
    
    # The spatial grid spacing.
    dx = 1.0 / N  
    
    # Set a reasonable internal time step dt_internal based on diffusive stability condition.
    # For explicit Euler, a sufficient condition is dt < C * dx^2/nu. Use C=0.2 for safety.
    dt_internal = 0.2 * dx * dx / nu
    print("Internal time step dt_internal =", dt_internal)
    
    # Total number of output time steps provided in t_coordinate.
    T_plus_one = len(t_coordinate)
    
    # Preallocate a tensor (on device, then later convert) for the final solution.
    solution_tensor = torch.empty((batch_size, T_plus_one, N), dtype=torch.float32, device=device)
    
    # Set the initial condition
    solution_tensor[:, 0, :] = u

    # current simulation time starts from the initial time.
    current_time = t_coordinate[0]
    output_index = 1  # next output index to fill from time coordinate.
    
    # Get the final simulation time we need to compute until.
    final_time = t_coordinate[-1]
    
    internal_step = 0  # counter for debugging
    
    # Continue integration until we reach the final output time.
    while current_time < final_time:
        # Take one internal time step.
        # Note: We make sure not to overshoot the next required output time.
        next_output_time = t_coordinate[output_index] if output_index < T_plus_one else final_time
        # Determine time step dt: if the next internal step would overshoot the next output time,
        # set dt to exactly reach it.
        dt = dt_internal
        if current_time + dt > next_output_time:
            dt = next_output_time - current_time
        
        # Update the solution by one time step.
        u = solve_burgers_step(u, dt, dx, nu)
        current_time += dt
        
        internal_step += 1
        if internal_step % 1000 == 0:
            print("Internal step:", internal_step, "Current time:", current_time.item())
        
        # If we have reached or passed the next required time, store the result.
        # (Due to our dt adjustment, we should hit it exactly.)
        if abs(current_time - next_output_time) < 1e-10:
            solution_tensor[:, output_index, :] = u
            print("Recorded solution at t =", current_time.item(), "for output index", output_index)
            output_index += 1
            
            # If we have recorded all outputs, we can exit.
            if output_index >= T_plus_one:
                break
    
    # Convert the solution to numpy before returning.
    solutions = solution_tensor.cpu().numpy()
    return solutions
