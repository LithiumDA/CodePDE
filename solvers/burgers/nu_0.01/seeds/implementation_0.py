
import numpy as np
import torch
import torch.fft

# Try to use GPU if available, otherwise use CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU for computation.")
else:
    device = torch.device('cpu')
    print("Using CPU for computation.")

def compute_rhs_fourier(u_hat_batch, k, nu, ik):
    """Computes the right-hand side of the Burgers' equation in Fourier space.

    RHS = -i*k*FFT(0.5 * IFFT(u_hat)^2) - nu*k^2*u_hat

    Args:
        u_hat_batch (torch.Tensor): Fourier coefficients of u [batch_size, N]. Complex valued.
        k (torch.Tensor): Wave numbers [N]. Real valued.
        nu (float): Viscosity coefficient.
        ik (torch.Tensor): Imaginary unit times wave numbers [N]. Complex valued.

    Returns:
        rhs_hat (torch.Tensor): Fourier coefficients of the RHS [batch_size, N]. Complex valued.
    """
    # Calculate u in physical space
    u_batch = torch.fft.ifft(u_hat_batch, dim=-1)

    # Calculate the non-linear term (u^2 / 2) in physical space
    u_squared_half = 0.5 * u_batch * u_batch

    # Transform the non-linear term to Fourier space
    u_squared_half_hat = torch.fft.fft(u_squared_half, dim=-1)

    # Calculate the derivative of the non-linear term in Fourier space
    # Note the negative sign because the term is moved to the RHS: - d/dx(u^2/2)
    nonlinear_term_hat = -ik * u_squared_half_hat

    # Calculate the diffusion term in Fourier space
    # Note the positive sign because the term is nu * d^2u/dx^2 = -nu * k^2 * u_hat
    diffusion_term_hat = -nu * (k**2) * u_hat_batch

    # Combine terms for the RHS in Fourier space
    rhs_hat = nonlinear_term_hat + diffusion_term_hat

    # Return the complex RHS in Fourier space
    # Important: Ensure the result is complex even if inputs somehow lead to real
    return rhs_hat.to(torch.complex64)


def rk4_step(u_hat_batch, k, nu, ik, dt):
    """Performs a single RK4 step in Fourier space.

    Args:
        u_hat_batch (torch.Tensor): Current Fourier coefficients [batch_size, N].
        k (torch.Tensor): Wave numbers [N].
        nu (float): Viscosity coefficient.
        ik (torch.Tensor): Imaginary unit times wave numbers [N].
        dt (float): Time step size.

    Returns:
        u_hat_next (torch.Tensor): Fourier coefficients after one RK4 step [batch_size, N].
    """
    # k1 = dt * f(u_n)
    k1 = dt * compute_rhs_fourier(u_hat_batch, k, nu, ik)
    
    # k2 = dt * f(u_n + 0.5*k1)
    k2 = dt * compute_rhs_fourier(u_hat_batch + 0.5 * k1, k, nu, ik)
    
    # k3 = dt * f(u_n + 0.5*k2)
    k3 = dt * compute_rhs_fourier(u_hat_batch + 0.5 * k2, k, nu, ik)
    
    # k4 = dt * f(u_n + k3)
    k4 = dt * compute_rhs_fourier(u_hat_batch + k3, k, nu, ik)
    
    # u_{n+1} = u_n + (k1 + 2*k2 + 2*k3 + k4) / 6
    u_hat_next = u_hat_batch + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    
    return u_hat_next


def solver(u0_batch, t_coordinate, nu):
    """Solves the Burgers' equation using a pseudo-spectral method with RK4 time stepping.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N], 
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. 
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        nu (float): Viscosity coefficient. Should be 0.01 for optimization.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    # --- 1. Setup ---
    batch_size, N = u0_batch.shape
    T_plus_1 = len(t_coordinate)
    T = T_plus_1 - 1

    # Convert initial condition to PyTorch tensor and move to device
    # Ensure dtype is float32 for potentially better GPU performance
    u_current_batch = torch.tensor(u0_batch, dtype=torch.float32, device=device)

    # Create spatial grid properties
    L = 1.0 # Domain length assumed to be 1 based on x in (0,1)
    dx = L / N # Spatial step size
    x = torch.linspace(0, L - dx, N, device=device, dtype=torch.float32) # Grid points

    # Calculate wave numbers k for Fourier transforms
    # k = 2 * pi * [0, 1, ..., N/2-1, -N/2, ..., -1] / L
    k = 2.0 * np.pi * torch.fft.fftfreq(N, d=dx).to(device)
    ik = 1j * k # Precompute ik for efficiency

    # Transform initial condition to Fourier space
    # Ensure input to fft is float or complex, output is complex
    u_hat_current_batch = torch.fft.fft(u_current_batch, dim=-1) 

    # --- 2. Time Stepping Setup ---
    # Determine internal time step dt
    # A smaller dt leads to more stability but more computation.
    # Let's base dt on the minimum time interval provided and refine it.
    min_delta_t_required = np.min(np.diff(t_coordinate)) if T > 0 else 1.0 # Avoid division by zero if only t=0 is given
    
    # Heuristic for dt: Ensure stability based on diffusion and CFL-like conditions.
    # For nu=0.01, diffusion stability might require dt ~ dx^2/nu
    # Advection CFL might require dt ~ dx/max(|u|)
    # Using a fixed number of steps between required points is simpler to implement
    # num_internal_steps = 100 # Increase if instability occurs
    # dt = min_delta_t_required / num_internal_steps

    # Alternative dt calculation (more robust):
    # Based roughly on RK4 stability region and dominant terms
    # Max wave number squared: k_max^2 ~ (pi*N/L)^2 = (pi*N)^2
    dt_diff_limit = (dx**2) / (4 * nu) if nu > 1e-9 else float('inf') # Based on Forward Euler diffusion limit (conservative)
    # Estimate max initial speed for CFL limit
    with torch.no_grad():
        max_u0 = torch.max(torch.abs(u_current_batch)).item()
    dt_cfl_limit = dx / (max_u0 + 1e-6) # Based on Forward Euler CFL limit (conservative)
    
    # Choose a dt smaller than these limits, e.g., a fraction
    # Using a slightly less conservative factor due to RK4
    # Let's aim for dt smaller than the minimum required time step too
    suggested_dt = 0.5 * min(dt_diff_limit, dt_cfl_limit) 
    # Further constrain dt to be at most the minimum required interval / some factor (e.g., 10)
    # to avoid overly large steps if stability limits allow.
    dt = min(suggested_dt, min_delta_t_required / 10.0) 
    
    # Ensure dt is positive
    dt = max(dt, 1e-7) # Add a small floor to dt
    
    print(f"Using internal time step dt = {dt:.6e}")
    print(f"Based on dx={dx:.4f}, nu={nu:.4f}, max|u0|={max_u0:.4f}")
    print(f"Stability limits (approx): dt_diff={dt_diff_limit:.4e}, dt_cfl={dt_cfl_limit:.4e}")


    # --- 3. Solution Storage ---
    # Initialize storage tensor on CPU for easier conversion to numpy later
    # Store complex Fourier coefficients internally if memory allows, or store real physical space values
    # Storing real values is required by the output format.
    solutions_torch = torch.zeros((batch_size, T_plus_1, N), dtype=torch.float32, device='cpu')
    
    # Store initial condition (convert back to CPU and real)
    solutions_torch[:, 0, :] = u_current_batch.cpu()

    # --- 4. Time Integration Loop ---
    current_t = 0.0
    for i in range(1, T_plus_1):
        t_target = t_coordinate[i]
        print(f"Integrating from t={current_t:.4f} to t={t_target:.4f}")
        
        # Internal steps to reach t_target
        while current_t < t_target - 1e-9: # Use tolerance for float comparison
            step_dt = min(dt, t_target - current_t)
            
            # Perform one RK4 step
            u_hat_current_batch = rk4_step(u_hat_current_batch, k, nu, ik, step_dt)
            
            # Update current time
            current_t += step_dt

        # --- 5. Store Solution ---
        # Transform back to physical space for storage
        u_physical_batch = torch.fft.ifft(u_hat_current_batch, dim=-1)
        
        # Store the real part of the solution at the required time step
        # Move to CPU before storing in the pre-allocated CPU tensor
        solutions_torch[:, i, :] = u_physical_batch.real.cpu() 
        
        # Optional: Print max value for monitoring stability
        # print(f"  Max value at t={current_t:.4f}: {torch.max(torch.abs(u_physical_batch.real)).item():.4f}")


    print("Solver finished.")
    # Convert final solutions tensor to NumPy array
    solutions = solutions_torch.numpy()
    
    return solutions

