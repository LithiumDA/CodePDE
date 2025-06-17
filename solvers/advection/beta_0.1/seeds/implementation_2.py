
import numpy as np
import torch # Import PyTorch

# Check if GPU is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU found, using CUDA.")
else:
    device = torch.device("cpu")
    print("GPU not found, using CPU.")

def spectral_step(u_t, dt, k, beta):
    """
    Performs one time step of the advection equation using the spectral method.

    Args:
        u_t (torch.Tensor): Solution at the current time step [batch_size, N].
        dt (float): Time step duration.
        k (torch.Tensor): Wavenumbers [1, N].
        beta (float): Advection speed.

    Returns:
        torch.Tensor: Solution at the next time step [batch_size, N].
    """
    # Compute the Fast Fourier Transform (FFT) along the spatial dimension (last dim)
    u_hat = torch.fft.fft(u_t, dim=-1)

    # Compute the propagator term in Fourier space: exp(-i * beta * k * dt)
    # Ensure k has the correct shape for broadcasting and is on the same device/dtype
    # Note: 1j is the imaginary unit in Python
    propagator = torch.exp(-1j * beta * k * dt)

    # Multiply the Fourier coefficients by the propagator
    u_hat_next = u_hat * propagator

    # Compute the Inverse Fast Fourier Transform (IFFT)
    u_next = torch.fft.ifft(u_hat_next, dim=-1)

    # Return the real part of the solution (numerical errors might introduce small imaginary parts)
    return u_next.real

def solver(u0_batch, t_coordinate, beta):
    """Solves the 1D Advection equation using the Fourier spectral method.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N],
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1].
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        beta (float): Constant advection speed. Specifically considered for beta=0.1.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    # --- 1. Initialization ---
    print("Starting solver...")

    # Get dimensions
    batch_size, N = u0_batch.shape
    num_time_steps = len(t_coordinate)
    T = num_time_steps - 1 # Number of intervals

    # Convert inputs to PyTorch tensors and move to the selected device
    # Use float32 for efficiency on GPUs, and derive complex dtype
    u0_batch_torch = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    t_coordinate_torch = torch.tensor(t_coordinate, dtype=torch.float32, device=device)

    # Define spatial domain parameters (assuming domain is [0, 1])
    L = 1.0 # Length of the spatial domain
    dx = L / N # Spatial step size

    # Calculate wavenumbers k for the spectral method
    # k = 2 * pi * [0, 1, ..., N/2-1, -N/2, ..., -1] / L
    # Use torch.fft.fftfreq for convenience
    # Reshape k to [1, N] for broadcasting with u_hat [batch_size, N]
    k_freq = torch.fft.fftfreq(N, d=dx, device=device)
    k = 2 * np.pi * k_freq
    k = k.reshape(1, N) # Reshape for broadcasting
    # Ensure k is complex for calculations involving 1j
    k = k.to(torch.complex64)


    # Initialize the solutions tensor to store results at each required time step
    # Shape: [batch_size, T+1, N]
    solutions_torch = torch.zeros((batch_size, num_time_steps, N), dtype=torch.float32, device=device)

    # Store the initial condition
    solutions_torch[:, 0, :] = u0_batch_torch
    print(f"Initial condition stored. Shape: {u0_batch_torch.shape}")

    # Set the current solution state
    u_current = u0_batch_torch

    # --- 2. Time Stepping Loop ---
    print(f"Starting time stepping for {T} intervals...")
    for i in range(T):
        # Calculate the time step duration for this interval
        t_current = t_coordinate_torch[i]
        t_next = t_coordinate_torch[i+1]
        dt = (t_next - t_current).item() # Get dt as a float

        # Perform one spectral step to get the solution at t_next
        u_next = spectral_step(u_current, dt, k, beta)

        # Store the solution at t_next
        solutions_torch[:, i+1, :] = u_next

        # Update the current solution for the next iteration
        u_current = u_next

        # Optional: Print progress
        if (i + 1) % max(1, T // 10) == 0 or i == T - 1:
             print(f"Computed step {i+1}/{T}, Time = {t_next.item():.4f}")

    # --- 3. Finalization ---
    print("Simulation finished.")

    # Move the solutions tensor back to CPU and convert to NumPy array
    solutions_np = solutions_torch.cpu().numpy()

    print(f"Returning solutions array with shape: {solutions_np.shape}")
    return solutions_np

