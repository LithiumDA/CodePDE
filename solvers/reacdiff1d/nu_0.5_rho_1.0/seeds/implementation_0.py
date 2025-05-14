
import numpy as np
import torch

def crank_nicolson_matrix(N, dx, dt, nu):
    """Constructs the A and B matrices for the Crank-Nicolson method."""
    r = nu * dt / (2 * dx**2)
    # Identity matrix
    I = torch.eye(N, device='cuda')
    
    # Off-diagonal shifts (periodic boundary)
    off_diag = -r * torch.roll(I, shifts=1, dims=1)
    off_diag += -r * torch.roll(I, shifts=-1, dims=1)
    
    # A and B matrices
    A = (1 + 2*r) * I + off_diag
    B = (1 - 2*r) * I - off_diag
    return A, B

def apply_reaction_term(u, rho, dt):
    """Applies the reaction term using explicit Euler."""
    return u + dt * rho * u * (1 - u)

def solver(u0_batch, t_coordinate, nu, rho):
    """Solves the 1D reaction diffusion equation for all times in t_coordinate."""
    # Extract the dimensions
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1
    
    # Convert to torch tensors for GPU operations
    u0_batch = torch.tensor(u0_batch, dtype=torch.float32, device='cuda')
    
    # Spatial step
    dx = 1.0 / N
    
    # Internal time step for stability
    dt_internal = 0.1 * dx**2 / nu
    num_internal_steps = int(np.ceil((t_coordinate[1] - t_coordinate[0]) / dt_internal))
    dt_internal = (t_coordinate[1] - t_coordinate[0]) / num_internal_steps
    
    # Precompute Crank-Nicolson matrices
    A, B = crank_nicolson_matrix(N, dx, dt_internal, nu)
    A_inv = torch.linalg.inv(A).to('cuda')
    
    # Initialize solution array
    solutions = torch.zeros((batch_size, T+1, N), device='cuda')
    solutions[:, 0, :] = u0_batch
    
    # Time-stepping loop
    for t in range(1, T+1):
        u = solutions[:, t-1, :].clone()
        for _ in range(num_internal_steps):
            u = apply_reaction_term(u, rho, dt_internal)  # Apply reaction term
            u = torch.matmul(B, u.T).T  # Now u has shape (batch_size, N)
            u = torch.matmul(A_inv, u.T).T  # Apply diffusion term, maintaining correct shape
        solutions[:, t, :] = u
    
    # Move result back to CPU
    return solutions.cpu().numpy()

# Example usage for testing
# u0_batch = np.random.rand(10, 1024)
# t_coordinate = np.linspace(0, 1, 11)
# solutions = solver(u0_batch, t_coordinate, nu=0.5, rho=1.0)
# print(solutions.shape)
