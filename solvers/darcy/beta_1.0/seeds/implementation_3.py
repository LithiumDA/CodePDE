
import numpy as np
import torch

def solve_sparse(A_data, A_indices, A_shape, b, max_iter=2000, tol=1e-6):
    """Conjugate Gradient solver with sparse matrix support."""
    x = torch.zeros_like(b)
    r = b - torch.sparse.mm(torch.sparse_coo_tensor(A_indices, A_data, A_shape), x.unsqueeze(1)).squeeze()
    p = r.clone()
    rsold = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = torch.sparse.mm(torch.sparse_coo_tensor(A_indices, A_data, A_shape), p.unsqueeze(1)).squeeze()
        alpha = rsold / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = torch.dot(r, r)
        
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

def solver(a):
    """Solve Darcy equation with corrected sparse matrix construction."""
    batch_size, N, _ = a.shape
    h = 1.0 / (N - 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    solutions = np.zeros((batch_size, N, N), dtype=np.float32)
    
    # Grid indices setup
    i_grid, j_grid = torch.meshgrid(torch.arange(N, device=device), 
                                    torch.arange(N, device=device), indexing='ij')
    k = i_grid * N + j_grid
    interior = (i_grid > 0) & (i_grid < N-1) & (j_grid > 0) & (j_grid < N-1)
    
    for b in range(batch_size):
        a_t = torch.tensor(a[b], dtype=torch.float32, device=device)
        
        # Precompute coefficient matrices
        a_x = 0.5 * (a_t[1:] + a_t[:-1])  # (N-1, N)
        a_y = 0.5 * (a_t[:, 1:] + a_t[:, :-1])  # (N, N-1)

        # Diagonal terms calculation
        diag = torch.zeros(N, N, device=device)
        diag[1:-1, :] = (a_x[:-1] + a_x[1:])/h**2  # x-contrib
        diag[:, 1:-1] += (a_y[:, :-1] + a_y[:, 1:])/h**2  # y-contrib
        diag = -diag

        rows, cols, values = [], [], []
        
        # Interior diagonal terms
        valid = interior
        rows.append(k[valid])
        cols.append(k[valid])
        values.append(diag[valid])

        # Off-diagonal terms with direct index calculation
        directions = [
            (1, 0, a_x, lambda i,j: (i, j)),    # East
            (-1, 0, a_x, lambda i,j: (i-1, j)), # West 
            (0, 1, a_y, lambda i,j: (i, j)),    # North
            (0, -1, a_y, lambda i,j: (i, j-1))  # South
        ]
        
        for di, dj, coeffs, idx_fn in directions:
            valid_neighbor = (i_grid + di >= 0) & (i_grid + di < N) & (j_grid + dj >= 0) & (j_grid + dj < N)
            current_valid = interior & valid_neighbor
            
            # Get indices for valid entries
            i_vals = i_grid[current_valid]
            j_vals = j_grid[current_valid]
            
            # Calculate coefficient indices
            ci, cj = idx_fn(i_vals, j_vals)
            coeff_vals = coeffs[ci, cj] / h**2

            rows.append(k[current_valid])
            cols.append(k[current_valid] + di*N + dj)
            values.append(coeff_vals)

        # Boundary conditions
        boundary = ~interior
        rows.append(k[boundary])
        cols.append(k[boundary])
        values.append(torch.ones(boundary.sum(), device=device))
        
        # Build sparse matrix
        A_indices = torch.stack([torch.cat(rows), torch.cat(cols)])
        A_data = torch.cat(values)
        
        # Right-hand side
        f = torch.zeros(N*N, device=device)
        f[interior.flatten()] = -1.0
        
        # Solve system
        u = solve_sparse(A_data, A_indices, (N*N, N*N), f)
        solutions[b] = u.reshape(N, N).cpu().numpy()
        
        print(f"Batch {b+1}/{batch_size}", end='\r')
    
    print("\nSolver completed with corrected sparse construction.")
    return solutions
