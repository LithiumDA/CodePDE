import numpy as np
from scipy.fftpack import idct
from scipy.sparse import diags
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, block_diag, bmat


import matplotlib.pyplot as plt

def GRF(alpha, tau, s):
    # Random variables in KL expansion
    xi = np.random.normal(0, 1, (s, s))
    
    # Define the (square root of) eigenvalues of the covariance operator
    K1, K2 = np.meshgrid(np.arange(s), np.arange(s))
    coef = tau**(alpha - 1) * (np.pi**2 * (K1**2 + K2**2) + tau**2)**(-alpha / 2)
    
    # Construct the KL coefficients
    L = s * coef * xi
    L[0, 0] = 0  # Ensure mean is 0
    
    # Perform the inverse discrete cosine transform
    U = idct(idct(L, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    return U


def solve_gwf(coef, F):
    K = len(coef)

    # Create meshgrids for interpolation
    X1, Y1 = np.meshgrid(np.linspace(1/(2*K), (2*K-1)/(2*K), K), 
                          np.linspace(1/(2*K), (2*K-1)/(2*K), K))
    X2, Y2 = np.meshgrid(np.linspace(0, 1, K), np.linspace(0, 1, K))

    # Interpolate coef and F
    interp_coef = RectBivariateSpline(X1[0, :], Y1[:, 0], coef, kx=3, ky=3)
    coef = interp_coef(X2[0, :], Y2[:, 0], grid=True)

    interp_F = RectBivariateSpline(X1[0, :], Y1[:, 0], F, kx=3, ky=3)
    F = interp_F(X2[0, :], Y2[:, 0], grid=True)

    # Trim F for the inner grid
    F = F[1:K-1, 1:K-1]

    N = (K-2)  # Size of inner grid

    # Initialize d with zero sparse matrices instead of None
    d = [[diags([np.zeros(N)], [0], shape=(N, N)) for _ in range(N)] for _ in range(N)]

    # Construct sparse matrix system
    for j in range(1, K-1):
        main_diag = (coef[:-2, j] + coef[1:-1, j])/2 + (coef[2:, j] + coef[1:-1, j])/2 \
                    + (coef[1:-1, j-1] + coef[1:-1, j])/2 + (coef[1:-1, j+1] + coef[1:-1, j])/2
        upper_diag = -(coef[1:-2, j] + coef[2:-1, j]) / 2
        lower_diag = upper_diag

        d[j-1][j-1] = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], shape=(N, N))

        if j != K-2:
            off_diag = - (coef[1:-1, j] + coef[1:-1, j+1]) / 2
            d[j-1][j] = diags([off_diag], offsets=[0], shape=(N, N))
            d[j][j-1] = d[j-1][j]

    # Convert d into a sparse block matrix
    A = bmat(d, format='csr') * (K-1)**2

    # Flatten F correctly for spsolve
    F_flat = F.flatten()

    # Solve the sparse system
    P_inner = spsolve(A, F_flat).reshape((N, N))

    # Construct full solution grid
    P = np.zeros((K, K))
    P[1:-1, 1:-1] = P_inner

    # Interpolate the solution back to original grid
    interp_P = RectBivariateSpline(X2[0, :], Y2[:, 0], P, kx=3, ky=3)
    P = interp_P(X1[0, :], Y1[:, 0], grid=True).T

    return P


def solve_darcy(alpha=2, tau=3, s=256):
    """
    Solve the Darcy equation using the GWF method
    tau and alpha are parameters of the covariance C = tau^(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
    Note that we need alpha > d/2 (here d=2)
    Laplacian has zero Neumann boundary
    alpha and tau control smoothness; the bigger they are, the smoother the function
    """
    # Number of grid points on [0,1]^2
    # i.e., uniform mesh with step h=1/(s-1)

    # Create mesh (only needed for plotting)
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    X, Y = np.meshgrid(x, y)

    # Generate random coefficients from N(0,C)
    norm_a = GRF(alpha, tau, s)

    # Another way to achieve ellipticity is to threshold the coefficients
    thresh_a = np.where(norm_a >= 0, 12, 4)

    # Forcing function, f(x) = 1 
    f = np.ones((s, s))

    # Solve PDE: - div(a(x)*grad(p(x))) = f(x)
    # lognorm_p = solve_gwf(lognorm_a, f)
    thresh_p = solve_gwf(thresh_a, f)

    return thresh_p

if __name__ == "__main__":
    thresh_p = solve_darcy()