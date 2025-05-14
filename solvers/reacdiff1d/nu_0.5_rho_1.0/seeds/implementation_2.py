
import numpy as np
import torch

def finite_difference_matrix(N, nu, dx):
    """Creates the finite difference matrix for the diffusion term with periodic boundaries."""
    diagonal = -2.0 * np.ones(N)
    off_diagonal = np.ones(N - 1)
    D = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    D[0, -1] = D[-1, 0] = 1.0  # Periodic boundary conditions
    return torch.tensor(D * nu / (dx ** 2), dtype=torch.float32)

def reaction_term(u, rho):
    """Computes the reaction term rho * u * (1 - u)."""
    return rho * u * (1 - u)

def solver(u0_batch, t_coordinate, nu, rho):
    """Solves the 1D reaction diffusion equation for all times in t_coordinate."""
    batch_size, N = u0_batch.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Discretization parameters
    dx = 1.0 / N
    dt_internal = 0.01  # Internal time step for stability
    num_internal_steps = int((t_coordinate[1] - t_coordinate[0]) / dt_internal)
    
    # Create finite difference matrix for diffusion
    D = finite_difference_matrix(N, nu, dx).to(device)
    
    # Initialize solutions tensor
    solutions = torch.zeros((batch_size, len(t_coordinate), N), dtype=torch.float32, device=device)
    solutions[:, 0, :] = torch.tensor(u0_batch, dtype=torch.float32, device=device)
    
    # Time integration loop
    for t in range(1, len(t_coordinate)):
        u = solutions[:, t-1, :].clone()
        num_steps = int((t_coordinate[t] - t_coordinate[t-1]) / dt_internal)
        for _ in range(num_steps):
            # Explicit reaction term
            reaction = reaction_term(u, rho)
            
            # Implicit diffusion term (solved using linear algebra)
            I = torch.eye(N, device=device)
            A = I - dt_internal * D
            
            # Expand A to handle batch processing
            A_batch = A.unsqueeze(0).expand(batch_size, -1, -1)
            b = u + dt_internal * reaction
            
            # Solve A * u_new = b for each batch
            u_new = torch.linalg.solve(A_batch, b.unsqueeze(2)).squeeze(2)
            
            # Update u
            u = u_new
        
        # Store solution
        solutions[:, t, :] = u

    return solutions.cpu().numpy()
