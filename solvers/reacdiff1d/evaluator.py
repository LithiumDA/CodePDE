import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import time

from solver import *


### For nRMSE evaluation
def compute_nrmse(u_computed, u_reference):
    """Computes the Normalized Root Mean Squared Error (nRMSE) between the computed solution and reference.

    Args:
        u_computed (np.ndarray): Computed solution [batch_size, len(t_coordinate), N].
        u_reference (np.ndarray): Reference solution [batch_size, len(t_coordinate), N].
    
    Returns:
        nrmse (np.float32): The normalized RMSE value.
    """
    rmse_values = np.sqrt(np.mean((u_computed - u_reference)**2, axis=(1,2)))
    u_true_norm = np.sqrt(np.mean(u_reference**2, axis=(1,2)))
    nrmse = np.mean(rmse_values / u_true_norm)
    return nrmse


### For convergence test
def init(xc, 
         modes: list =["sin", "sinsin", "Gaussian", "react", "possin"], 
         u0=1.0, 
         du=0.1):
    """Initializes one or more 1D scalar functions based on specified modes.

    Args:
        xc (np.ndarray): Cell center coordinates.
        modes (list): List of initial condition types to generate. Options include
                     "sin", "sinsin", "Gaussian", "react", and "possin".
        u0 (float): Base amplitude scaling factor.
        du (float): Secondary amplitude scaling factor for "sinsin" mode.
    
    Returns:
        np.ndarray: Stacked initial conditions with shape [len(modes), len(xc)].
    """
    initial_conditions = []
    for mode in modes:
        assert mode in ["sin", "sinsin", "Gaussian", "react", "possin"], f"mode {mode} not supported!"

        if mode == "sin":  # sinusoidal wave
            u = u0 * np.sin((xc + 1.0) * np.pi)
        elif mode == "sinsin":  # sinusoidal wave
            u = np.sin((xc + 1.0) * np.pi) + du * np.sin((xc + 1.0) * np.pi * 8.0)
        elif mode == "Gaussian":  # for diffusion check
            t0 = 1.0
            u = np.exp(-(xc**2) * np.pi / (4.0 * t0)) / np.sqrt(2.0 * t0)
        elif mode == "react":  # for reaction-diffusion eq.
            logu = -0.5 * (xc - np.pi) ** 2 / (0.25 * np.pi) ** 2
            u = np.exp(logu)
        elif mode == "possin":  # sinusoidal wave
            u = u0 * np.abs(np.sin((xc + 1.0) * np.pi))
        
        initial_conditions.append(u)
    return np.stack(initial_conditions)

def interpolate_solution(u_fine, x_fine, t_fine, x_coarse, t_coarse):
    """
    Interpolates the fine solution onto the coarse grid in both space and time.
    """
    # Interpolate in space
    space_interp_func = interp1d(x_fine, u_fine, axis=2, kind='linear', fill_value="extrapolate")
    # finding the values of the u_fine function over the grid points of x
    u_fine_interp_space = space_interp_func(x_coarse) 

    # Interpolate in time
    time_interp_func = interp1d(t_fine, u_fine_interp_space, axis=1, kind='linear', fill_value="extrapolate")
    # finding the values of the u_fine_interp_sapce function over the grid points of time.
    u_fine_interp = time_interp_func(t_coarse)

    return u_fine_interp

def compute_error(coarse_tuple, fine_tuple):
    """
    Computes the error between coarse and fine grid solutions by interpolating in both space and time.
    """
    u_coarse, x_coarse, t_coarse = coarse_tuple
    u_fine, x_fine, t_fine = fine_tuple
    u_fine_interp = interpolate_solution(u_fine, x_fine, t_fine, x_coarse, t_coarse)

    # Compute L2 norm error
    error = np.mean(np.linalg.norm(u_coarse - u_fine_interp, axis=(1,2))) / np.sqrt(u_coarse.size)
    return error

def get_x_coordinate(x_min, x_max, nx):
    dx = (x_max - x_min) / nx
    xe = np.linspace(x_min, x_max, nx+1)
    
    xc = xe[:-1] + 0.5 * dx
    return xc

def get_t_coordinate(t_min, t_max, nt):
    # t-coordinate
    it_tot = np.ceil((t_max - t_min) / nt) + 1
    tc = np.arange(it_tot + 1) * nt
    return tc

def convergence_test(nu, rho,
                     nxs=[256, 512, 1024, 2048], 
                     dts=[0.01, 0.01, 0.01, 0.01], 
                     t_min=0, t_max=2,
                     x_min=-1, x_max=1):
    print(f"##### Running convergence test for the solver #####")
    us = []
    xcs = []
    tcs = []
    for nx, dt in zip(nxs, dts):
        print(f"**** Spatio resolution {nx} ****")
        tc = get_t_coordinate(t_min, t_max, dt)
        xc = get_x_coordinate(x_min, x_max, nx)
        u0 = init(xc)
        u = solver(u0, tc, nu, rho)
        us.append(np.squeeze(np.array(u)))
        xcs.append(np.array(xc))
        tcs.append(np.array(tc))
        print(f"**** Finished ****")
    
    # now we try to compute error. 
    errors = []
    for i in range(len(nxs) - 1):
        coarse_tuple = (us[i], xcs[i], tcs[i])
        fine_tuple = (us[-1], xcs[-1], tcs[-1])
        error = compute_error(
            coarse_tuple, fine_tuple
        )
        errors.append(error)

    for i in range(len(nxs) - 2):
        rate = np.log(errors[i] / errors[i+1]) / np.log(nxs[i+1] / nxs[i])
        print(f"Error measured at spatio resolution {nxs[i]} is {errors[i]:.3e}")
        print(f"Rate of convergence measured at spatio resolution {nxs[i]} is {rate:.3f}")

    avg_rate = np.mean(
        [np.log(errors[i] / errors[i+1]) / np.log(nxs[i+1] / nxs[i]) for i in range(len(nxs) - 2)]
    )
    return avg_rate

def save_visualization(u_batch_np: np.array, u_ref_np: np.array, save_file_idx=0):
    """
    Save the visualization of u_batch and u_ref in 2D (space vs time).
    """
    difference_np = u_batch_np - u_ref_np
    fig, axs = plt.subplots(3, 1, figsize=(7, 12))

    im1 = axs[0].imshow(u_batch_np, aspect='auto', extent=[0, 1, 1, 0], cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar1.set_label("Predicted values", fontsize=14)
    axs[0].set_xlabel("Spatial Dimension (x)", fontsize=14)
    axs[0].set_ylabel("Temporal Dimension (t)", fontsize=14)
    axs[0].set_title("Computed Solution over Space and Time", fontsize=16)

    im2 = axs[1].imshow(u_ref_np, aspect='auto', extent=[0, 1, 1, 0], cmap='viridis')
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar2.set_label("Reference values", fontsize=14)
    axs[1].set_xlabel("Spatial Dimension (x)", fontsize=14)
    axs[1].set_ylabel("Temporal Dimension (t)", fontsize=14)
    axs[1].set_title("Reference Solution over Space and Time", fontsize=16)

    im3 = axs[2].imshow(difference_np, aspect='auto', extent=[0, 1, 1, 0], cmap='coolwarm')
    cbar3 = fig.colorbar(im3, ax=axs[2])
    cbar3.set_label("Prediction error", fontsize=14)
    axs[2].set_xlabel("Spatial Dimension (x)", fontsize=14)
    axs[2].set_ylabel("Temporal Dimension (t)", fontsize=14)
    axs[2].set_title("Prediction error over Space and Time", fontsize=16)

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(args.save_pth, f'visualization_{save_file_idx}.png'))

def time_min_max(t_coordinate):
    return t_coordinate[0], t_coordinate[-1]

def x_coord_min_max(x_coordinate):
    return x_coordinate[0], x_coordinate[-1]

def load_data(path, is_h5py=True):
    if is_h5py:
        #TODO: make sure this works out
        with h5py.File(path, 'r') as f:
            # Do NOT modify the data loading code
            t_coordinate = np.array(f['t-coordinate'])
            u = np.array(f['tensor'])
            x_coordinate = np.array(f['x-coordinate'])
    else:
        raise NotImplementedError("Only h5py format is supported for now.")

    t_min, t_max = time_min_max(t_coordinate)
    x_min, x_max = time_min_max(x_coordinate)
    return dict(
        tensor=u,
        t_coordinate=t_coordinate,
        x_coordinate=x_coordinate,
        t_min=t_min,
        t_max=t_max,
        x_min=x_min,
        x_max=x_max
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for solving 1D Reaction-Diffusion Equation.")

    parser.add_argument("--save-pth", type=str,
                        default='.',
                        help="The folder to save experimental results.")

    parser.add_argument("--run-id", type=str,
                        default=0,
                        help="The id of the current run.")

    parser.add_argument("--nu", type=float,
                        default=0.5,
                        choices=[0.5, 1.0, 2.0, 5.0],
                        help="The diffusion coefficient.")

    parser.add_argument("--rho", type=float,
                        default=1.0,
                        choices=[1.0, 2.0, 5.0, 10.0],
                        help="The reaction coefficient.")
    parser.add_argument("--dataset-path-for-eval", type=str,
                        default='/usr1/username/data/CodePDE/ReactionDiffusion/ReacDiff_Nu0.5_Rho1.0.hdf5',
                        help="The path to load the dataset.")

    args = parser.parse_args()

    data_dict = load_data(args.dataset_path_for_eval, is_h5py=True)
    u = data_dict['tensor']
    t_coordinate = data_dict['t_coordinate']
    x_coordinate = data_dict['x_coordinate']

    print(f"Loaded data with shape: {u.shape}")
    # t_coordinate contains T+1 time points, i.e., 0, t_1, ..., t_T.

    # Extract test set
    u0 = u[:, 0]
    u_ref = u[:, :]

    # Hyperparameters
    batch_size, N = u0.shape
    nu, rho = args.nu, args.rho

    # Run solver
    print(f"##### Running the solver on the given dataset #####")
    start_time = time.time()
    u_batch = solver(u0, t_coordinate, nu, rho)
    end_time = time.time()
    print(f"##### Finished #####")

    # Evaluation
    nrmse = compute_nrmse(u_batch, u_ref)
    avg_rate = convergence_test(
        nu,
        rho,
        t_min=data_dict['t_min'],
        t_max=data_dict['t_max']/10,  # to save time
        x_min=data_dict['x_min'],
        x_max=data_dict['x_max']
    )
    print(f"Result summary")
    print(
        f"nRMSE: {nrmse:.3e}\t| "
        f"Time: {end_time - start_time:.2f}s\t| "
        f"Average convergence rate: {avg_rate:.3f}\t|"
    )

    # Visualization for the first sample
    save_visualization(u_batch[2], u_ref[2], args.run_id)
