
import numpy as np
import torch

def solver(u0_batch, t_coordinate, beta):
    """Solves the Advection equation for all times in t_coordinate using spectral methods.
    
    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N]
        t_coordinate (np.ndarray): Time coordinates [T+1]
        beta (float): Advection speed
        
    Returns:
        np.ndarray: [batch_size, T+1, N] solution tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, N = u0_batch.shape
    
    # Convert to PyTorch tensors for GPU acceleration
    u0 = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    
    # Compute wavenumbers
    dx = 1.0 / N
    k = 2 * np.pi * torch.fft.fftfreq(N, d=dx).to(device)
    
    # Compute FFT of initial conditions
    fft_u0 = torch.fft.fft(u0, dim=1)  # [batch_size, N]
    
    # Prepare time steps excluding initial time 0
    times = torch.tensor(t_coordinate[1:], dtype=torch.float32, device=device)
    
    # Compute phase factors for each time and wavenumber
    phase_factors = torch.exp(-1j * beta * k * times[:, None])
    
    # Broadcast multiplication across batch and time dimensions
    fft_solutions = fft_u0[:, None, :] * phase_factors[None, :, :]
    
    # Inverse FFT to get spatial solutions
    solutions = torch.fft.ifft(fft_solutions, dim=-1).real  # [batch_size, T, N]
    
    # Concatenate initial condition
    initial = u0[:, None, :]
    all_solutions = torch.cat([initial, solutions], dim=1)
    
    # Convert back to numpy and return
    return all_solutions.cpu().numpy()
