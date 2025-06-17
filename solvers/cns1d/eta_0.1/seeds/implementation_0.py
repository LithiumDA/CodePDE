
import torch
import numpy as np

def solver(Vx0, density0, pressure0, t_coordinate, eta, zeta):
    """Solves the 1D compressible Navier-Stokes equations with improved stability."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Vx0 = torch.as_tensor(Vx0, dtype=torch.float32, device=device)
    density0 = torch.as_tensor(density0, dtype=torch.float32, device=device)
    pressure0 = torch.as_tensor(pressure0, dtype=torch.float32, device=device)
    t_coordinate = torch.as_tensor(t_coordinate, dtype=torch.float32, device=device)
    
    batch_size, N = Vx0.shape
    T = len(t_coordinate) - 1
    
    # Initialize output arrays
    Vx_pred = torch.zeros((batch_size, T+1, N), device=device)
    density_pred = torch.zeros((batch_size, T+1, N), device=device)
    pressure_pred = torch.zeros((batch_size, T+1, N), device=device)
    
    # Store initial conditions
    Vx_pred[:, 0] = Vx0
    density_pred[:, 0] = density0
    pressure_pred[:, 0] = pressure0
    
    # Constants
    GAMMA = 5/3
    dx = 2.0 / (N - 1)
    HYPERBOLIC_SAFETY = 0.4  # Less restrictive for wave propagation
    VISCOUS_SAFETY = 0.4      # Less restrictive for viscosity
    MIN_DT = 1e-6             # Minimum time step to prevent getting stuck
    
    # Combined viscosity coefficients
    viscosity_coeff = eta
    bulk_viscosity_coeff = zeta + (4/3)*eta

    def periodic_pad(x):
        """Apply periodic padding to tensor."""
        return torch.cat([x[:, -2:-1], x, x[:, 1:2]], dim=1)

    def compute_rhs(Vx, density, pressure):
        """Compute right-hand side of the PDE system with safeguards."""
        # Apply periodic boundary conditions
        Vx_pad = periodic_pad(Vx)
        density_pad = periodic_pad(density)
        pressure_pad = periodic_pad(pressure)
        
        # Compute spatial derivatives
        dv_dx = (Vx_pad[:, 2:] - Vx_pad[:, :-2]) / (2*dx)
        d2v_dx2 = (Vx_pad[:, 2:] - 2*Vx + Vx_pad[:, :-2]) / (dx**2)
        
        # Continuity equation
        rho_v = density * Vx
        rho_v_pad = periodic_pad(rho_v)
        drho_v_dx = (rho_v_pad[:, 2:] - rho_v_pad[:, :-2]) / (2*dx)
        drho_dt = -drho_v_dx
        
        # Momentum equation
        pressure_grad = (pressure_pad[:, 2:] - pressure_pad[:, :-2]) / (2*dx)
        dv_dt = (-Vx * dv_dx - 
                pressure_grad / torch.clamp(density, min=1e-6) +
                (viscosity_coeff + bulk_viscosity_coeff) * d2v_dx2)
        
        # Energy equation
        epsilon = pressure / (GAMMA - 1)
        total_energy = epsilon + 0.5 * density * Vx**2
        energy_flux = (total_energy + pressure) * Vx - Vx * bulk_viscosity_coeff * dv_dx
        
        energy_flux_pad = periodic_pad(energy_flux)
        denergy_flux_dx = (energy_flux_pad[:, 2:] - energy_flux_pad[:, :-2]) / (2*dx)
        dE_dt = -denergy_flux_dx
        
        # Pressure change from energy
        kinetic_energy_change = Vx * density * dv_dt + 0.5 * Vx**2 * drho_dt
        dp_dt = (GAMMA - 1) * (dE_dt - kinetic_energy_change)
        
        return drho_dt, dv_dt, dp_dt

    def rk4_step(Vx, density, pressure, dt):
        """Optimized RK4 step for small time steps."""
        # Stage 1
        k1_rho, k1_v, k1_p = compute_rhs(Vx, density, pressure)
        
        # Stage 2
        temp_Vx = Vx + 0.5*dt*k1_v
        temp_rho = torch.clamp(density + 0.5*dt*k1_rho, min=1e-6)
        temp_p = torch.clamp(pressure + 0.5*dt*k1_p, min=1e-6)
        k2_rho, k2_v, k2_p = compute_rhs(temp_Vx, temp_rho, temp_p)
        
        # Stage 3
        temp_Vx = Vx + 0.5*dt*k2_v
        temp_rho = torch.clamp(density + 0.5*dt*k2_rho, min=1e-6)
        temp_p = torch.clamp(pressure + 0.5*dt*k2_p, min=1e-6)
        k3_rho, k3_v, k3_p = compute_rhs(temp_Vx, temp_rho, temp_p)
        
        # Stage 4
        temp_Vx = Vx + dt*k3_v
        temp_rho = torch.clamp(density + dt*k3_rho, min=1e-6)
        temp_p = torch.clamp(pressure + dt*k3_p, min=1e-6)
        k4_rho, k4_v, k4_p = compute_rhs(temp_Vx, temp_rho, temp_p)
        
        # Combine stages
        Vx_new = Vx + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        density_new = torch.clamp(density + (dt/6) * (k1_rho + 2*k2_rho + 2*k3_rho + k4_rho), min=1e-6)
        pressure_new = torch.clamp(pressure + (dt/6) * (k1_p + 2*k2_p + 2*k3_p + k4_p), min=1e-6)
        
        return Vx_new, density_new, pressure_new
    
    # Main time stepping loop
    current_time = 0.0
    current_Vx = Vx0.clone()
    current_density = density0.clone()
    current_pressure = pressure0.clone()
    
    for i in range(1, T+1):
        target_time = t_coordinate[i].item()
        last_progress = current_time
        
        while current_time < target_time:
            # Enhanced CFL condition with separate safety factors
            sound_speed = torch.sqrt(GAMMA * current_pressure / torch.clamp(current_density, min=1e-6))
            max_wave_speed = torch.max(torch.abs(current_Vx) + sound_speed)
            
            # Hyperbolic time step restriction
            hyperbolic_dt = HYPERBOLIC_SAFETY * dx / (max_wave_speed + 1e-6)
            
            # Viscous time step restriction (less restrictive)
            viscous_dt = VISCOUS_SAFETY * (dx**2) / (2 * (viscosity_coeff + bulk_viscosity_coeff) + 1e-6)
            
            dt = min(hyperbolic_dt, viscous_dt, target_time - current_time)
            dt = max(dt, MIN_DT)  # Enforce minimum time step
            
            current_Vx, current_density, current_pressure = rk4_step(
                current_Vx, current_density, current_pressure, dt)
            current_time += dt
            
            # Progress monitoring
            if current_time - last_progress > 0.01:  # Print every 1% progress
                print(f"Progress: time={current_time:.4f}, max Vx={torch.max(torch.abs(current_Vx)):.4f}, dt={dt:.2e}")
                last_progress = current_time
        
        Vx_pred[:, i] = current_Vx
        density_pred[:, i] = current_density
        pressure_pred[:, i] = current_pressure
    
    return Vx_pred.cpu().numpy(), density_pred.cpu().numpy(), pressure_pred.cpu().numpy()
