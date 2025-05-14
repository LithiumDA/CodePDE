import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
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

# For convergence test
def convergence_test(a, u_pred, 
                     down_sample_rates=[6, 4, 3, 2],
                     batch_size=8): 
    """Use the test dataset for convergence test."""
    print(f"##### Running convergence test for the solver #####")

    a_fine, u_fine = a[:batch_size], u_pred[:batch_size]

    down_sample_rates = sorted(down_sample_rates, reverse=True)
    errors = []
    for rate in down_sample_rates:
        a_coarse = a_fine[:, ::rate, ::rate]
        u_coarse = solver(a_coarse)
        u_fine_proj = u_fine[:, ::rate, ::rate]
        error = np.mean(
            np.linalg.norm(u_coarse - u_fine_proj, axis=(1,2))
        ) / np.sqrt(u_coarse.size)
        errors.append(error)

    rates = []
    for i in range(len(errors)-1):
        rate = np.log(errors[i] / errors[i+1]) / np.log(down_sample_rates[i] / down_sample_rates[i+1])
        resolution = int(((a.shape[1] - 1)/down_sample_rates[i]) + 1)
        print(f"Rate of convergence measured at spatio resolution {resolution} is {rate:.3f}")
        rates.append(rate)

    avg_rate = sum(rates) / len(rates)
    print(f"Average rate of convergence is {avg_rate:.3f}")
    return avg_rate

def save_visualization(u_batch_np: np.array, u_ref_np: np.array, save_file_idx=0):
    """
    Save the visualization of u_batch and u_ref in 2D (space vs time).
    """
    difference_np = u_batch_np - u_ref_np
    fig, axs = plt.subplots(3, 1, figsize=(4, 12))

    im1 = axs[0].imshow(u_batch_np, aspect='auto', extent=[0, 1, 1, 0], cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar1.set_label("Predicted values", fontsize=14)
    axs[0].set_title("Computed Solution", fontsize=16)

    im2 = axs[1].imshow(u_ref_np, aspect='auto', extent=[0, 1, 1, 0], cmap='viridis')
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar2.set_label("Reference values", fontsize=14)
    axs[1].set_title("Reference Solution", fontsize=16)

    im3 = axs[2].imshow(difference_np, aspect='auto', extent=[0, 1, 1, 0], cmap='coolwarm')
    cbar3 = fig.colorbar(im3, ax=axs[2])
    cbar3.set_label("Prediction error", fontsize=14)
    axs[2].set_title("Prediction error", fontsize=16)

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(args.save_pth, f'visualization_{save_file_idx}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Solving 2D Darcy Equation.")

    parser.add_argument("--save-pth", type=str,
                        default='.',
                        help="The folder to save experimental results.")
    parser.add_argument("--run-id", type=str,
                        default=0,
                        help="The id of the current run.")
    parser.add_argument("--num-samples", type=int,
                        default=100,
                        help="The number of samples to test on.")
    parser.add_argument("--dataset-path-for-eval", type=str,
                        default='/usr1/username/data/CodePDE/Darcy/piececonst_r421_N1024_smooth1_sample100.hdf5',
                        help="The path to load the dataset.")

    args = parser.parse_args()

    with h5py.File(args.dataset_path_for_eval, 'r') as f:
        # Load the data
        u = np.array(f['sol'])[:args.num_samples]
        a = np.array(f['coeff'])[:args.num_samples]

    print(f"Loaded data with shape: {a.shape}")

    # Run solver
    print(f"##### Running the solver on the given dataset #####")
    start_time = time.time()
    u_pred = solver(a) 
    end_time = time.time()

    nrmse = compute_nrmse(u_pred, u)
    avg_rate = convergence_test(a, u_pred)

    print(f"Result summary")
    print(
        f"nRMSE: {nrmse:.3e}\t| "
        f"Time: {end_time - start_time:.2f}s\t| "
        f"Average convergence rate: {avg_rate:.3f}\t|"
    )

    # Visualization for the first sample
    save_visualization(u_pred[2], u[2], args.run_id)
