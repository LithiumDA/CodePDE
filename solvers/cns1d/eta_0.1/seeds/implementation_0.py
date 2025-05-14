
import numpy as np
import torch
import math

def solver(Vx0, density0, pressure0, t_coordinate, eta, zeta):
    """Solves the 1D compressible Navier-Stokes equations for all times in t_coordinate using an IMEX approach.
    
    The convective terms are updated explicitly using a Lax–Friedrichs flux, while the viscous term in the momentum 
    equation is updated implicitly via an FFT solve, allowing us to use a larger time–step.
    
    Args:
        Vx0 (np.ndarray): Initial velocity of shape [batch_size, N] 
            where N is the number of uniformly spaced spatial points.
        density0 (np.ndarray): Initial density [batch_size, N].
        pressure0 (np.ndarray): Initial pressure [batch_size, N].
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. It begins with t_0=0 
            and includes the subsequent time steps.
        eta (float): The shear viscosity coefficient.
        zeta (float): The bulk viscosity coefficient.
        
    Returns:
        Vx_pred (np.ndarray): Velocity solutions with shape [batch_size, len(t_coordinate), N].
        density_pred (np.ndarray): Density solutions with shape [batch_size, len(t_coordinate), N].
        pressure_pred (np.ndarray): Pressure solutions with shape [batch_size, len(t_coordinate), N].
    """
    # Select torch device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert initial conditions to torch tensors.
    density = torch.tensor(density0, dtype=torch.float32, device=device)  # shape [B, N]
    velocity = torch.tensor(Vx0, dtype=torch.float32, device=device)
    pressure = torch.tensor(pressure0, dtype=torch.float32, device=device)
    
    # Physical parameters.
    gamma = 5.0/3.0  # adiabatic index
    mu_eff = eta + zeta + eta/3.0  # effective viscosity for momentum diffusion
    
    # Spatial grid.
    batch_size, N = density.shape
    L = 1.0         # spatial domain size (assumed [0, 1])
    dx = L / N      # grid spacing
    
    # Set up Fourier frequencies for the implicit viscous solve.
    # k = 2*pi*n / L, where n = 0, 1, ..., N/2, -N/2+1, ... -1.
    freqs = torch.fft.fftfreq(N, d=dx).to(device)  # shape [N]
    k = 2 * math.pi * freqs  # wave numbers (real tensor)
    # Reshape k for broadcasting: shape [1, N]
    k = k.view(1, N)
    
    # Time coordinates.
    t_coordinate = np.array(t_coordinate, dtype=np.float32)
    T_out = len(t_coordinate)
    
    # Prepare storage for output (each tensor shape: [batch_size, T_out, N]).
    density_history = torch.empty((batch_size, T_out, N), dtype=torch.float32, device=device)
    velocity_history = torch.empty((batch_size, T_out, N), dtype=torch.float32, device=device)
    pressure_history = torch.empty((batch_size, T_out, N), dtype=torch.float32, device=device)
    
    # Compute initial total energy: E = p/(gamma-1) + 0.5 * density * velocity**2
    energy = pressure/(gamma - 1.0) + 0.5 * density * velocity**2
    # Compute initial momentum.
    momentum = density * velocity
    
    # Save initial state.
    density_history[:, 0, :] = density
    velocity_history[:, 0, :] = velocity
    pressure_history[:, 0, :] = pressure
    
    # Set CFL parameters (choose based on advective constraints only).
    CFL = 0.8  # now we allow a higher CFL because diffusion is treated implicitly.
    
    # Helper functions for periodic boundary shifts.
    def shift_right(f):
        return torch.roll(f, shifts=-1, dims=-1)
    
    def shift_left(f):
        return torch.roll(f, shifts=1, dims=-1)
    
    # Initialize simulation time.
    t_sim = 0.0
    
    # Main time-loop over external output times.
    for out_idx in range(1, T_out):
        t_target = t_coordinate[out_idx]
        print(f"Starting external time step to t = {t_target:.4f} ...")
        
        # Advance simulation until t_sim reaches the current external target time.
        while t_sim < t_target:
            # Recompute primitive variables.
            density = torch.clamp(density, min=1e-6)
            velocity = momentum / density
            kinetic = 0.5 * density * velocity**2
            pressure = (gamma - 1.0) * (energy - kinetic)
            pressure = torch.clamp(pressure, min=1e-6)
            
            # Compute sound speed.
            sound_speed = torch.sqrt(gamma * pressure / density)
            max_speed = torch.max(torch.abs(velocity) + sound_speed).item()
            
            # Determine dt based solely on the advective CFL.
            dt_advective = CFL * dx / (max_speed + 1e-6)
            dt = min(dt_advective, t_target - t_sim)
            
            if dt <= 0 or not np.isfinite(dt):
                print("Encountered non-finite or zero dt, aborting integration step.")
                break
            
            # --- Explicit update using Lax-Friedrichs for convective (hyperbolic) part ---
            # Pack conserved variables: U = [density, momentum, energy]
            U = torch.stack([density, momentum, energy], dim=0)  # shape [3, B, N]
            
            # Compute fluxes F for each conserved variable.
            # Mass flux: F₁ = momentum.
            F1 = momentum
            # Momentum flux: F₂ = (momentum^2/density) + pressure.
            F2 = (momentum**2 / density) + pressure
            # Energy flux:   F₃ = (energy + pressure) * velocity.
            F3 = (energy + pressure) * velocity
            F = torch.stack([F1, F2, F3], dim=0)
            
            # Use periodic shifts for Lax-Friedrichs update.
            U_right = torch.roll(U, shifts=-1, dims=-1)
            U_left  = torch.roll(U, shifts=1, dims=-1)
            F_right = torch.roll(F, shifts=-1, dims=-1)
            F_left  = torch.roll(F, shifts=1, dims=-1)
            
            # Lax-Friedrichs update.
            U_explicit = 0.5*(U_right + U_left) - (dt/(2*dx))*(F_right - F_left)
            
            # Unpack updated conserved variables from explicit (convective) update.
            density_new = U_explicit[0]
            momentum_explicit = U_explicit[1]
            energy = U_explicit[2]  # for energy, diffusion is not present
            
            # --- Implicit update for the viscous term in momentum ---
            # Compute explicit estimate for velocity.
            density_new = torch.clamp(density_new, min=1e-6)
            v_explicit = momentum_explicit / density_new
            
            # Solve the diffusion equation in Fourier space:
            # We want to find v_new such that:
            #   v_new - dt * mu_eff * Δv_new = v_explicit.
            # In Fourier space for each mode, this becomes:
            #   v̂_new = v̂_explicit / (1 + dt * mu_eff * k^2)
            v_fft = torch.fft.fft(v_explicit, dim=-1)
            # Denominator: shape [1, N] broadcasted over batch.
            denom = 1 + dt * mu_eff * (k**2)
            v_new_fft = v_fft / denom
            # Inverse FFT to get the updated velocity.
            v_new = torch.real(torch.fft.ifft(v_new_fft, dim=-1))
            
            # Update momentum using the implicitly updated velocity.
            momentum = density_new * v_new
            
            # Update density.
            density = density_new
            
            # Ensure energy remains consistent.
            velocity = momentum / density
            kinetic = 0.5 * density * velocity**2
            energy = torch.maximum(energy, kinetic + 1e-6)
            pressure = (gamma - 1.0) * (energy - kinetic)
            pressure = torch.clamp(pressure, min=1e-6)
            
            # Increment simulation time.
            t_sim += dt
            if not np.isfinite(t_sim):
                print("Simulation time became non-finite.")
                break
        
        # Record the state at the current external time.
        density_history[:, out_idx, :] = density
        velocity_history[:, out_idx, :] = momentum / density  # recover velocity from momentum and density
        pressure_history[:, out_idx, :] = pressure
        print(f"Recorded state at t = {t_target:.4f} (simulated time {t_sim:.4f}).")
    
    # Convert results to numpy arrays.
    Vx_pred = velocity_history.cpu().numpy()
    density_pred = density_history.cpu().numpy()
    pressure_pred = pressure_history.cpu().numpy()
    
    return Vx_pred, density_pred, pressure_pred
