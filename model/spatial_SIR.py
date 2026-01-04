import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SPATIAL SIR MODEL USING FINITE DIFFERENCE METHOD (FDM)
# =============================================================================
# TASK: Simulate the spread of an infectious disease using the PDE version
# of the SIR model and explore the impact of transmission and recovery rates
# on spatial dynamics.
#
# GOVERNING EQUATIONS (Partial Differential Equations):
# ∂S/∂t = D_s∇²S - β·S·I/N    (Susceptible: diffusion - infection)
# ∂I/∂t = D_i∇²I + β·S·I/N - γ·I    (Infected: diffusion + infection - recovery)
# ∂R/∂t = γ·I    (Recovered: only from infected)
#
# Where:
# - S, I, R: Population densities (individuals/cm²)
# - β: Transmission rate (day⁻¹)
# - γ: Recovery rate (day⁻¹)
# - N: Total population
# - D_s, D_i: Diffusion coefficients (cm²/day)
# - ∇²: Laplacian operator (second spatial derivative)
# =============================================================================

def run_sir_simulation(beta, gamma, N, Ds, Di, dx, dy, nx, ny, dt, t_max, 
                       verbose=True):
    """
    Run a spatial SIR simulation using explicit Finite Difference Method.
    
    This function implements the 4-step FDM process:
    1. Discretize space and time into a grid
    2. Apply finite difference formulas (forward time, central space)
    3. Update state variables using explicit equations
    4. Enforce boundary conditions and stability constraints
    
    Parameters:
    -----------
    beta : float
        Transmission rate (day⁻¹), controls infection spread speed
    gamma : float
        Recovery rate (day⁻¹), controls how fast infected individuals recover
    N : float
        Total population (constant)
    Ds, Di : float
        Diffusion coefficients for S and I (cm²/day), controls spatial movement
    dx, dy : float
        Spatial step sizes (cm)
    nx, ny : int
        Number of grid points in x and y directions
    dt : float
        Time step size (day)
    t_max : float
        Maximum simulation time (days)
    verbose : bool
        Print progress information
    
    Returns:
    --------
    S_history, I_history, R_history : ndarray
        3D arrays of shape (n_timesteps, nx, ny) containing spatial distributions
    time_points : ndarray
        1D array of time values
    """
    
    # =========================================================================
    # STEP 1: DISCRETIZATION OF SPACE AND TIME
    # =========================================================================
    # "Discretization of Space and Time"
    #
    # We replace continuous variables with discrete approximations:
    # - Continuous time t → discrete time steps n·Δt
    # - Continuous space (x,y) → discrete grid points (i·Δx, j·Δy)
    # - Function notation: U(x,y,t) → U_{i,j}^n
    #
    # Example: I(x,y,t) becomes I_{i,j}^n where:
    #   i = spatial index in x-direction (0 to nx-1)
    #   j = spatial index in y-direction (0 to ny-1)
    #   n = time index (0 to n_steps)
    # =========================================================================
    
    n_steps = int(t_max / dt)  # Total number of time steps
    n_points = nx * ny          # Total number of spatial grid points
    
    # =========================================================================
    # STEP 4: STABILITY CONSTRAINT (Von Neumann Stability Analysis)
    # =========================================================================
    # "Stability"
    #
    # For explicit FDM with diffusion, we must satisfy:
    # Δt ≤ Δx²/(2D) and Δt ≤ Δy²/(2D)
    #
    # This ensures the numerical solution doesn't produce "nonsensical 
    # physical conditions" such as negative populations or explosive growth.
    #
    # Physical interpretation: The time step must be small enough that
    # diffusion can't move population further than one grid cell in one step.
    # =========================================================================
    
    max_dt_x = dx**2 / (2 * max(Ds, Di))  # Maximum stable Δt from x-direction
    max_dt_y = dy**2 / (2 * max(Ds, Di))  # Maximum stable Δt from y-direction
    max_dt = min(max_dt_x, max_dt_y)       # Most restrictive constraint
    
    if verbose:
        print(f"\nSimulation: β={beta:.2f}, γ={gamma:.2f}, N={N}")
        print(f"  Stability check: Δt={dt:.4f}, max_stable_Δt={max_dt:.4f}", end="")
        if dt > max_dt:
            print("UNSTABLE - reduce Δt!")
        else:
            print("Stable")
        print(f"  Basic reproduction number: R₀ = β/γ = {beta/gamma:.2f}")
        if beta/gamma > 1:
            print(f"    → R₀ > 1: Epidemic will occur")
        else:
            print(f"    → R₀ < 1: Disease will die out")
    
    # =========================================================================
    # STEP 4: INITIAL CONDITIONS (ICs)
    # =========================================================================
    # "Initial Conditions"
    #
    # Define S_{i,j}^0, I_{i,j}^0, R_{i,j}^0 (state at time n=0)
    #
    # Strategy: Create a "small drop of infection in the center of the grid"
    # - Most of population is susceptible
    # - Small infected region at center
    # - No recovered individuals initially
    # =========================================================================
    
    # Initialize spatial arrays at time n=0
    S = np.ones((nx, ny)) * (N - 50) / n_points  # S_{i,j}^0: uniformly distributed
    I = np.zeros((nx, ny))                        # I_{i,j}^0: zero everywhere
    R = np.zeros((nx, ny))                        # R_{i,j}^0: zero everywhere
    
    # Create localized infection ("drop" at center)
    center_i, center_j = nx//2, ny//2
    I[center_i-1:center_i+1, center_j-1:center_j+1] = 50 / 4  # 50 total infected
    
    # Storage arrays for visualization (store at regular intervals)
    storage_interval = int(1.0 / dt)  # Store every 1 day
    n_stored = n_steps // storage_interval + 1
    
    S_history = np.zeros((n_stored, nx, ny))
    I_history = np.zeros((n_stored, nx, ny))
    R_history = np.zeros((n_stored, nx, ny))
    time_points = np.zeros(n_stored)
    
    # Store initial condition
    S_history[0] = S.copy()
    I_history[0] = I.copy()
    R_history[0] = R.copy()
    time_points[0] = 0
    
    # =========================================================================
    # MAIN TIME LOOP: EXPLICIT FINITE DIFFERENCE TIME STEPPING
    # =========================================================================
    # We will march forward in time: n=0 → n=1 → n=2 → ... → n=n_steps
    # At each step, we compute the state at time (n+1) from state at time (n)
    # =========================================================================
    
    for step in range(1, n_steps + 1):
        # =====================================================================
        # STEP 2: APPLY THE FINITE DIFFERENCE TOOLKIT
        # =====================================================================
        # "The Finite Difference Toolkit"
        #
        # We need to compute the LAPLACIAN operator ∇²U for both S and I
        #
        # DISCRETE LAPLACIAN FORMULA (2D):
        # ∇²U_{i,j}^n ≈ [U_{i+1,j}^n - 2U_{i,j}^n + U_{i-1,j}^n] / Δx²
        #             + [U_{i,j+1}^n - 2U_{i,j}^n + U_{i,j-1}^n] / Δy²
        #
        # This formula comes from Taylor series expansion and represents
        # how much U differs from its neighbors (curvature in space).
        #
        # Physical meaning: Positive ∇²U means U is locally a minimum
        # (surrounded by higher values), causing diffusion INTO this point.
        # =====================================================================
        
        # Initialize Laplacian arrays
        laplace_S = np.zeros((nx, ny))  # Will store ∇²S for all grid points
        laplace_I = np.zeros((nx, ny))  # Will store ∇²I for all grid points
        
        # Loop over all interior and boundary points
        for i in range(nx):
            for j in range(ny):
                # =============================================================
                # STEP 4: BOUNDARY CONDITIONS (BCs)
                # =============================================================
                # "Boundary Conditions"
                #
                # We implement NEUMANN boundary conditions: ∂U/∂n = 0
                # Physical meaning: "No flux" - no one enters or leaves the area
                #
                # Implementation: At boundaries, we use the boundary value itself
                # instead of a non-existent neighbor. This makes the derivative
                # zero at the boundary.
                #
                # Example: At left edge (i=0), we need U_{-1,j} which doesn't
                # exist. We set U_{-1,j} = U_{0,j}, so:
                # (U_{0,j} - 2U_{0,j} + U_{0,j})/Δx² - (U_{1,j} - 2U_{0,j} + U_{0,j})/Δx²
                # This effectively implements ∂U/∂x|_{x=0} = 0
                # =============================================================
                
                # For S (Susceptible):
                # Get neighboring values, applying Neumann BC at boundaries
                S_left = S[i-1, j] if i > 0 else S[i, j]       # S_{i-1,j} or boundary
                S_right = S[i+1, j] if i < nx-1 else S[i, j]   # S_{i+1,j} or boundary
                S_down = S[i, j-1] if j > 0 else S[i, j]       # S_{i,j-1} or boundary
                S_up = S[i, j+1] if j < ny-1 else S[i, j]      # S_{i,j+1} or boundary
                
                # Apply discrete Laplacian formula
                laplace_S[i, j] = ((S_right - 2*S[i,j] + S_left) / dx**2 + 
                                   (S_up - 2*S[i,j] + S_down) / dy**2)
                
                # For I (Infected):
                # Get neighboring values, applying Neumann BC at boundaries
                I_left = I[i-1, j] if i > 0 else I[i, j]       # I_{i-1,j} or boundary
                I_right = I[i+1, j] if i < nx-1 else I[i, j]   # I_{i+1,j} or boundary
                I_down = I[i, j-1] if j > 0 else I[i, j]       # I_{i,j-1} or boundary
                I_up = I[i, j+1] if j < ny-1 else I[i, j]      # I_{i,j+1} or boundary
                
                # Apply discrete Laplacian formula
                laplace_I[i, j] = ((I_right - 2*I[i,j] + I_left) / dx**2 + 
                                   (I_up - 2*I[i,j] + I_down) / dy**2)
        
        # =====================================================================
        # STEP 3: DISCRETE IMPLEMENTATION OF SIR EQUATIONS
        # =====================================================================
        # "Discrete Implementation"
        #
        # We use FORWARD DIFFERENCE for time derivative:
        # ∂U/∂t ≈ (U^{n+1} - U^n) / Δt
        #
        # Rearranging: U^{n+1} = U^n + Δt · (∂U/∂t)
        #
        # This is the EXPLICIT EULER METHOD: we solve for future state (n+1)
        # directly from current state (n) without solving any equations.
        # =====================================================================
        
        # ---------------------------------------------------------------------
        # SUSCEPTIBLE EQUATION:
        # ∂S/∂t = D_s∇²S - β·S·I/N
        #         ↑       ↑
        #         |       └─ Infection term: S loses members to I
        #         └───────── Diffusion term: S spreads spatially
        #
        # Discrete form:
        # S_{i,j}^{n+1} = S_{i,j}^n + Δt[D_s·∇²S_{i,j}^n - β·S_{i,j}^n·I_{i,j}^n/N]
        # ---------------------------------------------------------------------
        S_new = S + dt * (Ds * laplace_S - beta * S * I / N)
        #       ↑   ↑    ↑               ↑
        #       |   |    |               └─ Infection: -β·S·I/N (loss)
        #       |   |    └───────────────── Diffusion: D_s·∇²S (spreading)
        #       |   └────────────────────── Time step multiplier
        #       └────────────────────────── Current state
        
        # ---------------------------------------------------------------------
        # INFECTED EQUATION:
        # ∂I/∂t = D_i∇²I + β·S·I/N - γ·I
        #         ↑        ↑         ↑
        #         |        |         └─ Recovery term: I loses members to R
        #         |        └─────────── Infection term: I gains from S
        #         └──────────────────── Diffusion term: I spreads spatially
        #
        # Discrete form:
        # I_{i,j}^{n+1} = I_{i,j}^n + Δt[D_i·∇²I_{i,j}^n + β·S_{i,j}^n·I_{i,j}^n/N - γ·I_{i,j}^n]
        # ---------------------------------------------------------------------
        I_new = I + dt * (Di * laplace_I + beta * S * I / N - gamma * I)
        #       ↑   ↑    ↑                ↑                  ↑
        #       |   |    |                |                  └─ Recovery: -γ·I (loss)
        #       |   |    |                └──────────────────── Infection: +β·S·I/N (gain)
        #       |   |    └───────────────────────────────────── Diffusion: D_i·∇²I (spreading)
        #       |   └────────────────────────────────────────── Time step multiplier
        #       └────────────────────────────────────────────── Current state
        
        # ---------------------------------------------------------------------
        # RECOVERED EQUATION:
        # ∂R/∂t = γ·I
        #         ↑
        #         └─ Recovery term: R gains members from I
        #            (No diffusion for R in this model)
        #
        # Discrete form:
        # R_{i,j}^{n+1} = R_{i,j}^n + Δt[γ·I_{i,j}^n]
        # ---------------------------------------------------------------------
        R_new = R + dt * (gamma * I)
        #       ↑   ↑    ↑
        #       |   |    └─ Recovery: γ·I (gain from infected)
        #       |   └────── Time step multiplier
        #       └────────── Current state
        
        # Update state: n → n+1
        S, I, R = S_new, I_new, R_new
        
        # Store results at regular intervals
        if step % storage_interval == 0:
            idx = step // storage_interval
            S_history[idx] = S.copy()
            I_history[idx] = I.copy()
            R_history[idx] = R.copy()
            time_points[idx] = step * dt
    
    return S_history, I_history, R_history, time_points

# =============================================================================
# MAIN SIMULATION SETUP
# =============================================================================
print("=" * 70)
print("SPATIAL SIR MODEL - PARAMETER EXPLORATION")
print("=" * 70)

# =============================================================================
# SIMULATION PARAMETERS (Following Task Specifications)
# =============================================================================
# Task requirements:
# - N: 1000 - 5000
# - β: 0.1 - 0.5 day⁻¹
# - γ: 0.05 - 0.2 day⁻¹
# - D_s, D_i: 0.01 - 0.1 cm²/day
# =============================================================================

# Spatial discretization
nx, ny = 20, 20      # 20×20 grid = 400 spatial points
dx = 1.0             # Δx = 1 cm (spatial resolution)
dy = 1.0             # Δy = 1 cm (spatial resolution)

# Temporal discretization
dt = 0.01            # Δt = 0.01 days (must be small for stability!)
t_max = 100          # Simulate for 100 days

# Fixed parameters (within task-specified ranges)
N = 5000         # Total population (middle of 1000-5000 range)
Ds = 0.05            # Diffusion for S: 0.05 cm²/day (middle of 0.01-0.1)
Di = 0.05            # Diffusion for I: 0.05 cm²/day (middle of 0.01-0.1)

# =============================================================================
# PARAMETER EXPLORATION: IMPACT OF β AND γ ON SPATIAL DYNAMICS
# =============================================================================
# Task: "Explore the impact of transmission and recovery rates on the 
# spatial dynamics of the epidemic"
#
# We test 4 scenarios covering different combinations of β and γ:
# 1. Low β, Low γ: Slow spread, slow recovery
# 2. High β, Low γ: Fast spread, slow recovery (worst case)
# 3. Low β, High γ: Slow spread, fast recovery (best case)
# 4. High β, High γ: Fast spread, fast recovery (moderate case)
# =============================================================================

print("\nRunning parameter exploration:")
print("  Testing combinations of β (transmission) and γ (recovery)")
print()

scenarios = [
    {"beta": 0.1, "gamma": 0.05, "label": "β=0.2, γ=0.1 (R₀=2.0)"},
    {"beta": 0.5, "gamma": 0.05, "label": "β=0.4, γ=0.1 (R₀=4.0)"},
    {"beta": 0.1, "gamma": 0.2, "label": "β=0.2, γ=0.15 (R₀=1.33)"},
    {"beta": 0.5, "gamma": 0.2, "label": "β=0.4, γ=0.15 (R₀=2.67)"},
]

results = []

for scenario in scenarios:
    S_hist, I_hist, R_hist, times = run_sir_simulation(
        beta=scenario["beta"],
        gamma=scenario["gamma"],
        N=N, Ds=Ds, Di=Di,
        dx=dx, dy=dy, nx=nx, ny=ny,
        dt=dt, t_max=t_max,
        verbose=True
    )
    results.append({
        "S": S_hist,
        "I": I_hist,
        "R": R_hist,
        "times": times,
        "beta": scenario["beta"],
        "gamma": scenario["gamma"],
        "label": scenario["label"]
    })

print("\n" + "=" * 70)
print("All simulations complete!")
print("=" * 70)

# =============================================================================
# VISUALIZATION 1: Spatial spread with SEPARATE scales
# =============================================================================
print("\n" + "="*70)
print("GENERATING PLOTS...")
print("="*70)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
time_snapshots = [0, 30, 60, 90]

for row, result in enumerate(results):
    # Use SEPARATE color scale for each row to see patterns clearly
    row_max = np.max(result["I"])
    
    for col, t_idx in enumerate(time_snapshots):
        ax = axes[row, col]
        I_data = result["I"][t_idx]
        
        im = ax.imshow(I_data, origin='lower', cmap='hot', 
                      vmin=0, vmax=row_max)
        
        # Show the actual maximum value
        max_val = np.max(I_data)
        ax.text(0.5, 0.95, f'max={max_val:.2f}', 
                transform=ax.transAxes, 
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)
        
        if col == 0:
            ax.set_ylabel(result["label"], fontsize=9, fontweight='bold')
        if row == 0:
            ax.set_title(f"t={result['times'][t_idx]:.0f} days", 
                        fontsize=10, fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar for each row (rightmost plot)
        if col == 3:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label('I', fontsize=8)

plt.suptitle('Spatial Infection Spread (Each row has its own color scale)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =============================================================================
# VISUALIZATION 2: Time series comparison
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, result in enumerate(results):
    ax = axes[idx // 2, idx % 2]
    
    total_S = np.sum(result["S"], axis=(1, 2))
    total_I = np.sum(result["I"], axis=(1, 2))
    total_R = np.sum(result["R"], axis=(1, 2))
    
    ax.plot(result["times"], total_S, 'b-', linewidth=2, label='S')
    ax.plot(result["times"], total_I, 'r-', linewidth=2, label='I')
    ax.plot(result["times"], total_R, 'g-', linewidth=2, label='R')
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title(result["label"], fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('SIR Dynamics Over Time', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =============================================================================
# VISUALIZATION 3: Direct comparison of infected curves
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot all infected curves together
for result in results:
    total_I = np.sum(result["I"], axis=(1, 2))
    ax1.plot(result["times"], total_I, linewidth=2.5, label=result["label"])

ax1.set_xlabel('Time (days)', fontsize=11)
ax1.set_ylabel('Total Infected', fontsize=11)
ax1.set_title('Infected Population Over Time\n(All scenarios overlaid)', 
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Peak analysis
peak_times = []
peak_values = []
labels = []

for result in results:
    total_I = np.sum(result["I"], axis=(1, 2))
    peak_idx = np.argmax(total_I)
    peak_times.append(result["times"][peak_idx])
    peak_values.append(total_I[peak_idx])
    labels.append(f"R₀={result['beta']/result['gamma']:.1f}")

colors = ['skyblue', 'orange', 'lightgreen', 'pink']
x_pos = np.arange(len(labels))

ax2.bar(x_pos, peak_values, color=colors)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels)
ax2.set_ylabel('Peak Infected', fontsize=11)
ax2.set_title('Peak Infection Magnitude\n(Higher R₀ = Larger outbreak)', 
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Scenario':<20} {'R₀':<8} {'Peak I':<12} {'Peak Time':<12} {'Attack Rate'}")
print("-"*70)

for result in results:
    total_I = np.sum(result["I"], axis=(1, 2))
    peak_I = np.max(total_I)
    peak_time = result["times"][np.argmax(total_I)]
    final_R = np.sum(result["R"][-1])
    attack_rate = 100 * final_R / N
    R0 = result["beta"] / result["gamma"]
    
    label_short = f"β={result['beta']}, γ={result['gamma']}"
    print(f"{label_short:<20} {R0:<8.2f} {peak_I:<12.1f} {peak_time:<12.1f} {attack_rate:.1f}%")

print("="*70)
print("\nKEY OBSERVATIONS:")
print("  • R₀ > 1: Epidemic occurs")
print("  • R₀ < 1: Disease dies out")
print("  • Higher R₀ → Earlier peak, higher magnitude, more total infections")
print("  • β controls infection rate, γ controls recovery rate")
print("="*70)