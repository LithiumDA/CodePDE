
import numpy as np
import torch

def matvec(a, x):
    """Compute matrix-vector product A*x implicitly using finite differences."""
    B, Nx, Ny = x.shape
    N_full = Nx + 2  # Full grid size including boundaries
    current_a = a[:, 1:-1, 1:-1]  # Interior region of a

    # Compute neighboring coefficients
    east_a = a[:, 2:N_full, 1:-1]
    west_a = a[:, 0:N_full-2, 1:-1]
    north_a = a[:, 1:-1, 2:N_full]
    south_a = a[:, 1:-1, 0:N_full-2]

    # Calculate coefficients for each direction
    coeff_east = (current_a + east_a) / 2.0
    coeff_west = (west_a + current_a) / 2.0
    coeff_north = (current_a + north_a) / 2.0
    coeff_south = (south_a + current_a) / 2.0

    # Center coefficient
    center_coeff = - (west_a + east_a + south_a + north_a + 4.0 * current_a) / 2.0

    # Compute contributions from each direction
    # East contribution (shifted down)
    east_x = x[:, 1:, :]
    east_contribution = torch.zeros_like(x)
    east_contribution[:, :-1, :] = coeff_east[:, :-1, :] * east_x

    # West contribution (shifted up)
    west_x = x[:, :-1, :]
    west_contribution = torch.zeros_like(x)
    west_contribution[:, 1:, :] = coeff_west[:, 1:, :] * west_x

    # North contribution (shifted right)
    north_x = x[:, :, 1:]
    north_contribution = torch.zeros_like(x)
    north_contribution[:, :, :-1] = coeff_north[:, :, :-1] * north_x

    # South contribution (shifted left)
    south_x = x[:, :, :-1]
    south_contribution = torch.zeros_like(x)
    south_contribution[:, :, 1:] = coeff_south[:, :, 1:] * south_x

    # Combine all contributions and add center term
    result = (east_contribution + west_contribution +
              north_contribution + south_contribution +
              center_coeff * x)
    return result

def conjugate_gradient(a, b, x0, max_iter=1000, tol=1e-8):
    """Batched Conjugate Gradient solver for Ax = b."""
    x = x0.clone()
    r = b - matvec(a, x)
    p = r.clone()
    rsold = (r * r).sum(dim=(1, 2), keepdim=True)
    for i in range(max_iter):
        Ap = matvec(a, p)
        pAp = (p * Ap).sum(dim=(1, 2), keepdim=True)
        alpha = rsold / (pAp + 1e-10)  # Avoid division by zero
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = (r * r).sum(dim=(1, 2), keepdim=True)
        if torch.all(rsnew < tol ** 2):
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    return x

def solver(a):
    """Solve the Darcy equation using PyTorch and CG."""
    a = torch.from_numpy(a).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = a.to(device)
    B, N, _ = a.shape
    Nx = N - 2  # Interior grid size

    h = 1.0 / (N - 1)
    b_val = -h * h  # RHS value

    # Create RHS tensor and initial guess
    b = torch.full((B, Nx, Nx), b_val, dtype=torch.float32, device=device)
    x0 = torch.zeros(B, Nx, Nx, dtype=torch.float32, device=device)

    # Solve using conjugate gradient
    x = conjugate_gradient(a, b, x0)

    # Convert back to NumPy and pad boundaries with zeros
    solution = x.cpu().numpy()
    full_solution = np.zeros((B, N, N), dtype=np.float32)
    full_solution[:, 1:-1, 1:-1] = solution

    return full_solution
