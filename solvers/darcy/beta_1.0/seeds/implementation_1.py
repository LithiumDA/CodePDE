
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from concurrent.futures import ProcessPoolExecutor
import time

def solver(a):
    """Solve the Darcy equation.

    Args:
        a (np.ndarray): Shape [batch_size, N, N], the coefficient of the Darcy equation,
            where N denotes the number of spatial grid points in each direction.

    Returns:
        solutions (np.ndarray): Shape [batch_size, N, N].
    """
    start_time = time.time()
    batch_size, N, _ = a.shape
    
    # Create the solutions array
    solutions = np.zeros((batch_size, N, N), dtype=np.float64)
    
    # Process each batch element
    if batch_size > 1:
        print(f"Solving {batch_size} PDE problems with grid size {N}x{N}")
        # For larger batches, use parallel processing
        with ProcessPoolExecutor() as executor:
            # Create a list of tasks
            futures = [executor.submit(solve_single_problem, a[i], N) for i in range(batch_size)]
            
            # Collect results
            for i, future in enumerate(futures):
                solutions[i] = future.result()
    else:
        # For a single problem, solve directly
        solutions[0] = solve_single_problem(a[0], N)
    
    elapsed_time = time.time() - start_time
    print(f"Total solving time: {elapsed_time:.4f} seconds")
    
    return solutions

def solve_single_problem(a_coeff, N):
    """Solve a single Darcy flow problem.
    
    Args:
        a_coeff (np.ndarray): Shape [N, N], coefficient function
        N (int): Number of grid points in each direction
        
    Returns:
        u (np.ndarray): Shape [N, N], solution
    """
    # Set up the grid
    h = 1.0 / (N - 1)  # Grid spacing
    
    # Construct the sparse matrix for the linear system
    # We'll use the 5-point stencil for the Laplacian
    main_diag = np.zeros(N*N)
    upper_diag = np.zeros(N*N-1)
    lower_diag = np.zeros(N*N-1)
    upper_N_diag = np.zeros(N*N-N)
    lower_N_diag = np.zeros(N*N-N)
    
    # Right-hand side vector
    rhs = np.ones(N*N)
    
    # Apply boundary conditions and set up the linear system
    for i in range(N):
        for j in range(N):
            idx = i*N + j
            
            # Handle boundary conditions
            if i == 0 or i == N-1 or j == 0 or j == N-1:
                main_diag[idx] = 1.0
                rhs[idx] = 0.0  # u = 0 on the boundary
            else:
                # Get coefficients at neighboring points
                a_center = a_coeff[i, j]
                a_west = a_coeff[i, j-1]
                a_east = a_coeff[i, j+1]
                a_north = a_coeff[i-1, j]
                a_south = a_coeff[i+1, j]
                
                # Average coefficients at interfaces
                a_w = 0.5 * (a_center + a_west)
                a_e = 0.5 * (a_center + a_east)
                a_n = 0.5 * (a_center + a_north)
                a_s = 0.5 * (a_center + a_south)
                
                # Set diagonal entries
                main_diag[idx] = (a_w + a_e + a_n + a_s) / (h*h)
                
                # Set off-diagonal entries
                if j > 0:  # West
                    lower_diag[idx-1] = -a_w / (h*h)
                if j < N-1:  # East
                    upper_diag[idx] = -a_e / (h*h)
                if i > 0:  # North
                    lower_N_diag[idx-N] = -a_n / (h*h)
                if i < N-1:  # South
                    upper_N_diag[idx] = -a_s / (h*h)
    
    # Construct the sparse matrix
    diagonals = [main_diag, upper_diag, lower_diag, upper_N_diag, lower_N_diag]
    offsets = [0, 1, -1, N, -N]
    A = sparse.diags(diagonals, offsets, shape=(N*N, N*N), format='csr')
    
    # Solve the linear system
    u_flat = spsolve(A, rhs)
    
    # Reshape the solution to 2D
    u = u_flat.reshape((N, N))
    
    return u
