
import numpy as np
import torch

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
    # Use PyTorch device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Convert input initial conditions to a torch tensor on the proper device
    u = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    
    batch_size, N = u.shape
    # Spatial domain details:
    # Assume x in [0,1) with N equidistant points; grid spacing:
    dx = 1.0 / N

    # Create Fourier frequencies for spectral differentiation.
    # Domain length is 1; compute frequencies scaled by 2*pi.
    k = 2 * np.pi * torch.fft.fftfreq(N, d=dx).to(device)
    # Reshape k to (1, N) for broadcasting over batches.
    k = k.view(1, N)

    # Function to compute spatial derivatives using FFT.
    def spectral_derivative(u_tensor, order=1):
        """Computes the spectral derivative of the given order.
        
        Args:
            u_tensor (torch.Tensor): shape [batch_size, N].
            order (int): Derivative order (1 or 2).
        
        Returns:
            torch.Tensor: derivative with shape [batch_size, N].
        """
        # Apply FFT along spatial direction.
        u_hat = torch.fft.fft(u_tensor)
        if order == 1:
            # First derivative: multiply by i*k.
            factor = (1j * k)
        elif order == 2:
            # Second derivative: multiply by -k^2.
            factor = -(k ** 2)
        else:
            raise ValueError("Only first and second derivatives are supported!")
        # Multiply in Fourier space.
        u_hat_deriv = factor * u_hat
        # Inverse FFT back to physical space.
        u_deriv = torch.real(torch.fft.ifft(u_hat_deriv))
        return u_deriv

    # Function for computing the right-hand side of Burgers' equation.
    def burgers_rhs(u_tensor):
        """Computes the time derivative (RHS) of the Burgers' equation.
        
        u_t = - d_dx (0.5 * u^2) + nu * u_xx.
        
        Args:
            u_tensor (torch.Tensor): shape [batch_size, N].
        
        Returns:
            torch.Tensor: time derivative, shape [batch_size, N].
        """
        # Compute nonlinear convective flux and its derivative.
        flux = 0.5 * u_tensor ** 2
        dflux_dx = spectral_derivative(flux, order=1)
        # Compute diffusive term (second spatial derivative).
        u_xx = spectral_derivative(u_tensor, order=2)
        # Combine terms.
        return -dflux_dx + nu * u_xx

    # RK4 integrator for one time step.
    def rk4_step(u_tensor, dt):
        """Performs one RK4 step for the given time step dt.
        
        Args:
            u_tensor (torch.Tensor): current solution [batch_size, N].
            dt (float): time step size.
        
        Returns:
            torch.Tensor: solution after dt.
        """
        k1 = burgers_rhs(u_tensor)
        k2 = burgers_rhs(u_tensor + 0.5 * dt * k1)
        k3 = burgers_rhs(u_tensor + 0.5 * dt * k2)
        k4 = burgers_rhs(u_tensor + dt * k3)
        return u_tensor + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    # Prepare solution storage: [batch_size, T+1, N]
    T_total = len(t_coordinate) - 1  # number of intervals
    solutions = torch.empty((batch_size, T_total + 1, N), dtype=torch.float32, device=device)
    solutions[:, 0, :] = u

    # Safety constant for CFL condition.
    CFL = 0.1
    # Set a target dt in case CFL condition permits larger steps.
    dt_internal_target = 1e-4

    current_time = t_coordinate[0]
    print("Initial time: {}. Starting simulation...".format(current_time))
    
    # Loop over each output time interval.
    for t_idx in range(T_total):
        t_start = t_coordinate[t_idx]
        t_end = t_coordinate[t_idx + 1]
        dt_out = t_end - t_start

        # Compute maximum velocity for CFL condition across the batch.
        u_max_val = u.abs().max().item()
        # Compute CFL-driven dt; add a tiny number to avoid division by zero.
        dt_cfl = CFL * dx / (u_max_val + 1e-6)
        # Use the minimum of our target dt_internal and the CFL condition.
        dt_internal = min(dt_internal_target, dt_cfl)
        
        # Determine number of internal substeps to cover the output interval.
        n_substeps = max(int(np.ceil(dt_out / dt_internal)), 1)
        dt_internal = dt_out / n_substeps  # adjust dt_internal exactly
        
        print("Evolving from t = {:.4f} to t = {:.4f} with {} internal steps (dt_internal = {:.6e}), u_max = {:.6e}".format(
            t_start, t_end, n_substeps, dt_internal, u_max_val))
        
        # Perform internal RK4 steps.
        for i in range(n_substeps):
            u = rk4_step(u, dt_internal)
            # Check for NaNs during computation to break early if needed.
            if torch.isnan(u).any():
                print("NaN detected during integration; aborting further steps.")
                break
        
        current_time = t_end
        solutions[:, t_idx + 1, :] = u
        
        # Output diagnostics.
        u_min = u.min().item()
        u_max = u.max().item()
        print("At t = {:.4f}, u.min() = {:.6e}, u.max() = {:.6e}".format(current_time, u_min, u_max))
        # Early termination if NaNs occur.
        if np.isnan(u_min) or np.isnan(u_max):
            print("NaN values detected in solution. Terminating simulation early.")
            break

    # Convert result to numpy array and return.
    solutions_np = solutions.cpu().detach().numpy()
    print("Simulation completed.")
    return solutions_np
