import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import ndimage
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Callable
import time
from matplotlib import cm, colors
import matplotlib.animation as animation

# Constants for molecule types
ALPHA = "alpha_factor"
IAA = "IAA"
BETA = "beta_estradiol"
GFP = "GFP"
VENUS = "Venus"
BAR1 = "BAR1"
GH3 = "GH3"

# Constants for strain types
ACTIVATION = "activation"
REPRESSION = "repression"

@dataclass
class StrainParameters:
    """Parameters for a single strain."""
    strain_id: str
    input_molecule: str  # ALPHA_FACTOR, AUXIN, or BETA_ESTRADIOL
    regulation_type: str  # ACTIVATION or REPRESSION
    output_molecule: str  # ALPHA_FACTOR, AUXIN, GFP, VENUS, BAR1, or GH3
    
    # Model parameters (from Figure 1 in paper)
    k1: float  # Input binding rate
    d1: float  # Input-binder complex degradation rate
    k2: float  # Max transcription rate
    K: float   # Half-saturation constant
    n: float   # Hill coefficient
    d2: float  # TF degradation rate
    k3: float  # Output production rate
    d3: float  # Output degradation rate
    b: float   # Basal output production rate
    
    # Growth parameters (logistic model with lag phase)
    k: float = 100.0   # Carrying capacity
    r: float = 0.3     # Growth rate
    A: float = 1e-8    # Initial population fraction
    lag: float = 0.0   # Lag time (hours)
    
    def __post_init__(self):
        """Validate parameters."""
        valid_inputs = [ALPHA, IAA, BETA]
        valid_outputs = [ALPHA, IAA, GFP, VENUS, BAR1, GH3]
        valid_regulation = [ACTIVATION, REPRESSION]
        
        if self.input_molecule not in valid_inputs:
            raise ValueError(f"Invalid input molecule: {self.input_molecule}")
        
        if self.output_molecule not in valid_outputs:
            raise ValueError(f"Invalid output molecule: {self.output_molecule}")
            
        if self.regulation_type not in valid_regulation:
            raise ValueError(f"Invalid regulation type: {self.regulation_type}")


class SpatialMultiStrainModel:
    """Model for simulating multiple interacting yeast strains with spatial dynamics."""
    
    def __init__(self, grid_size=(50, 50), dx=0.1, coarse_factor=1):
        """
        Initialize the spatial model.
        
        Args:
            grid_size: Tuple of (height, width) for the 2D grid
            dx: Grid spacing (in mm)
            coarse_factor: Factor to coarsen the grid (1 = original resolution)
        """
        if coarse_factor > 1:
            # Calculate new grid size after coarsening
            self.grid_size = (grid_size[0] // coarse_factor, grid_size[1] // coarse_factor)
            self.dx = dx * coarse_factor  # Adjust spatial step size
        else:
            self.grid_size = grid_size
            self.dx = dx
            
        self.strains = []
        self.initial_molecule_grids = {}
        self.strain_grids = []
        self.time_span = (0, 10)
        
        # Diffusion coefficients (mm²/hour)
        self.diffusion_coefficients = {
            ALPHA: 0.15,  # Alpha factor
            IAA: 0.67,    # Auxin
            BETA: 0.2,    # Beta estradiol
            BAR1: 0.05,   # Enzymes diffuse slower
            GH3: 0.05
        }
        
        # Default growth parameters - will be overridden by strain-specific parameters
        self.default_carrying_capacity = 10.0  # Default carrying capacity
        
        # Initialize with zero concentration grids
        for molecule in [ALPHA, IAA, BETA, GFP, VENUS, BAR1, GH3]:
            self.initial_molecule_grids[molecule] = np.zeros(grid_size)
    
    def add_strain(self, strain_params: StrainParameters, initial_grid: Optional[np.ndarray] = None):
        """
        Add a strain to the model with its spatial distribution.
        
        Args:
            strain_params: Parameters for the strain
            initial_grid: Initial spatial distribution of the strain (if None, all zeros)
        """
        self.strains.append(strain_params)
        
        if initial_grid is None:
            # Default to all zeros grid
            self.strain_grids.append(np.zeros(self.grid_size))
        elif initial_grid.shape != self.grid_size:
            raise ValueError(f"Initial grid shape {initial_grid.shape} must match model grid size {self.grid_size}")
        else:
            self.strain_grids.append(initial_grid.copy())
        
        return self
    
    def place_strain(self, strain_idx: int, row: int, col: int, shape: str = "circle", 
                   radius: int = None, width: int = None, height: int = None, 
                   concentration: float = 1.0):
        """
        Place a colony of a strain at a specific location with specified shape.
        
        Args:
            strain_idx: Index of the strain to place
            row, col: Center coordinates of the colony
            shape: Shape of the colony ("circle" or "rectangle")
            radius: Radius of the colony (for circular shape)
            width: Width of the colony (for rectangular shape)
            height: Height of the colony (for rectangular shape)
            concentration: Concentration/population of the strain
        """
        if strain_idx < 0 or strain_idx >= len(self.strains):
            raise ValueError(f"Invalid strain index: {strain_idx}")
        
        # Validate shape parameter
        shape = shape.lower()
        if shape not in ["circle", "rectangle"]:
            raise ValueError(f"Invalid shape: {shape}. Must be 'circle' or 'rectangle'")
        
        # Create mask based on shape
        if shape == "circle":
            if radius is None:
                raise ValueError("Radius must be specified for circular shape")
            
            # Create a circular colony
            y, x = np.ogrid[-row:self.grid_size[0]-row, -col:self.grid_size[1]-col]
            mask = x*x + y*y <= radius*radius
            
        elif shape == "rectangle":
            if width is None or height is None:
                raise ValueError("Width and height must be specified for rectangular shape")
            
            # Create a rectangular colony
            # Calculate rectangle boundaries
            half_width = width // 2
            half_height = height // 2
            
            # Define rectangle coordinates
            min_row = max(0, row - half_height)
            max_row = min(self.grid_size[0], row + half_height)
            min_col = max(0, col - half_width)
            max_col = min(self.grid_size[1], col + half_width)
            
            # Create mask
            mask = np.zeros(self.grid_size, dtype=bool)
            mask[min_row:max_row, min_col:max_col] = True
        
        # Place the colony on the grid
        self.strain_grids[strain_idx][mask] = concentration
        
        return self
    
    def place_molecule(self, molecule: str, row: int, col: int, shape: str = "circle", 
                     radius: int = None, width: int = None, height: int = None, 
                     concentration: float = 1.0):
        """
        Place a region of a molecule at a specific location with specified shape.
        
        Args:
            molecule: Name of the molecule
            row, col: Center coordinates of the region
            shape: Shape of the region ("circle" or "rectangle")
            radius: Radius of the region (for circular shape)
            width: Width of the region (for rectangular shape)
            height: Height of the region (for rectangular shape)
            concentration: Concentration of the molecule
        """
        if molecule not in self.initial_molecule_grids:
            raise ValueError(f"Unknown molecule: {molecule}")
        
        # Validate shape parameter
        shape = shape.lower()
        if shape not in ["circle", "rectangle"]:
            raise ValueError(f"Invalid shape: {shape}. Must be 'circle' or 'rectangle'")
        
        # Create mask based on shape
        if shape == "circle":
            if radius is None:
                raise ValueError("Radius must be specified for circular shape")
            
            # Create a circular region
            y, x = np.ogrid[-row:self.grid_size[0]-row, -col:self.grid_size[1]-col]
            mask = x*x + y*y <= radius*radius
            
        elif shape == "rectangle":
            if width is None or height is None:
                raise ValueError("Width and height must be specified for rectangular shape")
            
            # Create a rectangular region
            # Calculate rectangle boundaries
            half_width = width // 2
            half_height = height // 2
            
            # Define rectangle coordinates
            min_row = max(0, row - half_height)
            max_row = min(self.grid_size[0], row + half_height)
            min_col = max(0, col - half_width)
            max_col = min(self.grid_size[1], col + half_width)
            
            # Create mask
            mask = np.zeros(self.grid_size, dtype=bool)
            mask[min_row:max_row, min_col:max_col] = True
        
        # Place the molecule on the grid
        self.initial_molecule_grids[molecule][mask] = concentration
        
        return self
    def set_diffusion_coefficient(self, molecule: str, coefficient: float):
        """
        Set the diffusion coefficient for a molecule.
        
        Args:
            molecule: Name of the molecule
            coefficient: Diffusion coefficient (mm²/hour)
        """
        self.diffusion_coefficients[molecule] = coefficient
        return self
    
    def set_simulation_time(self, t_start: float, t_end: float):
        """
        Set the simulation time span.
        
        Args:
            t_start: Start time (hours)
            t_end: End time (hours)
        """
        self.time_span = (t_start, t_end)
        return self
    
    def _build_optimized_spatial_ode_system(self) -> Callable:
        """
        Build an optimized spatial ODE system for the model with competition between strains.
        Uses scipy.ndimage for faster diffusion and focuses computation on active regions.
        
        Returns:
            Function that computes the derivatives for the state variables
        """
        # Grid dimensions
        grid_height, grid_width = self.grid_size
        
        # Number of diffusible molecules
        diffusible_molecules = [ALPHA, IAA, BETA, BAR1, GH3]
        n_diffusible = len(diffusible_molecules)
        
        # Number of fluorescent reporter molecules
        reporter_molecules = [GFP, VENUS]
        n_reporters = len(reporter_molecules)
        
        # Number of state variables per grid cell for each strain
        n_internal_states = 3
        
        # Diffusion coefficient for each molecule
        D_values = [self.diffusion_coefficients.get(molecule, 0.0) for molecule in diffusible_molecules]
        
        # Create index mappings for molecules
        diffusible_indices = {molecule: i for i, molecule in enumerate(diffusible_molecules)}
        reporter_indices = {molecule: i for i, molecule in enumerate(reporter_molecules)}
        
        # Extract growth parameters for each strain
        strain_growth_params = []
        for strain in self.strains:
            strain_growth_params.append({
                'k': strain.k,
                'r': strain.r,
                'lag': strain.lag
            })
        
        # Activity threshold for determining active regions
        activity_threshold = 0.005  # Regions with > 0.5% of max will be considered active
        
        def dydt(t, y):
            """Compute derivatives for all state variables - optimized version."""
            # Initialize derivatives array
            derivatives = np.zeros_like(y)
            
            # Reshape the state array
            state_idx = 0
            
            # Extract diffusible molecule grids
            diffusible_grids = []
            for i in range(n_diffusible):
                grid = y[state_idx:state_idx + grid_height*grid_width].reshape(grid_height, grid_width)
                diffusible_grids.append(grid)
                state_idx += grid_height*grid_width
            
            # Extract reporter molecule grids
            reporter_grids = []
            for i in range(n_reporters):
                grid = y[state_idx:state_idx + grid_height*grid_width].reshape(grid_height, grid_width)
                reporter_grids.append(grid)
                state_idx += grid_height*grid_width
            
            # Extract strain population grids and internal state grids
            strain_pop_grids = []
            strain_internal_states = []
            
            for strain_idx in range(len(self.strains)):
                # Population grid
                pop_grid = y[state_idx:state_idx + grid_height*grid_width].reshape(grid_height, grid_width)
                strain_pop_grids.append(pop_grid)
                state_idx += grid_height*grid_width
                
                # Internal state grids (3 per strain)
                strain_states = []
                for j in range(n_internal_states):
                    state_grid = y[state_idx:state_idx + grid_height*grid_width].reshape(grid_height, grid_width)
                    strain_states.append(state_grid)
                    state_idx += grid_height*grid_width
                
                strain_internal_states.append(strain_states)
            
            # Calculate total population
            total_population = np.zeros((grid_height, grid_width))
            for pop_grid in strain_pop_grids:
                total_population += pop_grid
                
            # Identify active regions to optimize computation
            active_mask = np.zeros((grid_height, grid_width), dtype=bool)
            
            # Mark regions with significant strain population or signaling molecules as active
            for pop_grid in strain_pop_grids:
                if np.max(pop_grid) > 0:
                    active_mask |= (pop_grid > activity_threshold * np.max(pop_grid))
            
            for mol_grid in diffusible_grids:
                if np.max(mol_grid) > 0:
                    active_mask |= (mol_grid > activity_threshold * np.max(mol_grid))
            
            # Dilate the active mask to include neighboring cells
            active_mask = ndimage.binary_dilation(active_mask, iterations=2)
            active_indices = np.where(active_mask)
            
            # Calculate diffusion for each diffusible molecule using ndimage.laplace
            for mol_idx, D in enumerate(D_values):
                if D > 0:  # Only apply diffusion if coefficient is positive
                    grid = diffusible_grids[mol_idx]
                    
                    # Use scipy.ndimage.laplace for much faster diffusion calculation
                    laplacian = ndimage.laplace(grid) / (self.dx**2)
                    
                    # Apply no-flux boundary conditions
                    # (Boundary code from original function)
                    
                    # Apply diffusion
                    diffusion_deriv = D * laplacian
                    
                    # Update derivatives for this molecule
                    start_idx = mol_idx * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] = diffusion_deriv.flatten()
            
            # Optimize strain dynamics calculation using active regions
            for strain_idx, strain_params in enumerate(self.strains):
                # Get strain data
                pop_grid = strain_pop_grids[strain_idx]
                strain_states = strain_internal_states[strain_idx]
                
                # Get growth parameters
                growth_params = strain_growth_params[strain_idx]
                k = growth_params['k']
                r = growth_params['r']
                lag = growth_params['lag']
                
                # Initialize derivative arrays
                pop_deriv = np.zeros_like(pop_grid)
                input_sensing_deriv = np.zeros_like(strain_states[0])
                signal_processing_deriv = np.zeros_like(strain_states[1])
                output_deriv = np.zeros_like(strain_states[2])
                
                # Get input molecule grid
                input_molecule = strain_params.input_molecule
                input_grid = diffusible_grids[diffusible_indices[input_molecule]]
                
                # Only calculate dynamics in active regions if they're a small portion of the grid
                if np.sum(active_mask) < 0.5 * grid_height * grid_width:
                    # Calculate growth only for active regions
                    if t >= lag:
                        for i, j in zip(*active_indices):
                            pop_deriv[i, j] = r * pop_grid[i, j] * (1 - total_population[i, j] / k)
                    
                    # Calculate internal states only for active regions
                    for i, j in zip(*active_indices):
                        # Input sensing
                        input_sensing_deriv[i, j] = strain_params.k1 * input_grid[i, j] - strain_params.d1 * strain_states[0][i, j]
                        
                        # Signal processing with appropriate Hill function
                        x1 = max(0, strain_states[0][i, j])
                        if strain_params.regulation_type == ACTIVATION:
                            hill_term = (x1 ** strain_params.n) / ((strain_params.K ** strain_params.n) + (x1 ** strain_params.n))
                            signal_processing_deriv[i, j] = (strain_params.k2 * hill_term) - (strain_params.d2 * strain_states[1][i, j])
                        else:
                            hill_term = 1 / (1 + ((x1 / strain_params.K) ** strain_params.n))
                            signal_processing_deriv[i, j] = (strain_params.k2 * hill_term) - (strain_params.d2 * strain_states[1][i, j])
                        
                        # Output production
                        output_deriv[i, j] = strain_params.b + strain_params.k3 * strain_states[1][i, j] - strain_params.d3 * strain_states[2][i, j]
                else:
                    # If most of the grid is active, use vectorized operations for better performance
                    if t >= lag:
                        pop_deriv = r * pop_grid * (1 - total_population / k)
                    
                    # Vectorized calculation of internal state derivatives
                    input_sensing_deriv = strain_params.k1 * input_grid - strain_params.d1 * strain_states[0]
                    
                    # Ensure non-negative values
                    x1 = np.maximum(0, strain_states[0])
                    
                    if strain_params.regulation_type == ACTIVATION:
                        x1_n = np.power(x1, strain_params.n)
                        K_n = np.power(strain_params.K, strain_params.n)
                        hill_term = x1_n / (K_n + x1_n)
                        signal_processing_deriv = (strain_params.k2 * hill_term) - (strain_params.d2 * strain_states[1])
                    else:
                        x1_over_K_n = np.power(x1 / strain_params.K, strain_params.n)
                        hill_term = 1 / (1 + x1_over_K_n)
                        signal_processing_deriv = (strain_params.k2 * hill_term) - (strain_params.d2 * strain_states[1])
                    
                    output_deriv = strain_params.b + strain_params.k3 * strain_states[1] - strain_params.d3 * strain_states[2]
                
                # Output rate is proportional to population
                output_rate = pop_grid * output_deriv
                
                # Update molecule grids based on strain output
                output_molecule = strain_params.output_molecule
                
                if output_molecule in diffusible_molecules:
                    mol_idx = diffusible_indices[output_molecule]
                    start_idx = mol_idx * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] += output_rate.flatten()
                elif output_molecule in reporter_molecules:
                    mol_idx = reporter_indices[output_molecule]
                    start_idx = (n_diffusible + mol_idx) * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] += output_rate.flatten()
                
                # Update strain derivatives
                start_idx = (n_diffusible + n_reporters) * grid_height * grid_width + strain_idx * (1 + n_internal_states) * grid_height * grid_width
                derivatives[start_idx:start_idx + grid_height*grid_width] = pop_deriv.flatten()
                derivatives[start_idx + grid_height*grid_width:start_idx + 2*grid_height*grid_width] = input_sensing_deriv.flatten()
                derivatives[start_idx + 2*grid_height*grid_width:start_idx + 3*grid_height*grid_width] = signal_processing_deriv.flatten()
                derivatives[start_idx + 3*grid_height*grid_width:start_idx + 4*grid_height*grid_width] = output_deriv.flatten()
            
            # Handle signal degradation (BAR1 and GH3)
            # (Similar to original code)
            # Handle signal degradation (BAR1 and GH3)
            # BAR1 degrades alpha-factor and GH3 converts IAA to inactive IAA-Asp
            if all(m in diffusible_indices for m in [ALPHA, BAR1]):
                alpha_idx = diffusible_indices[ALPHA]
                bar1_idx = diffusible_indices[BAR1]
                
                alpha_grid = diffusible_grids[alpha_idx]
                bar1_grid = diffusible_grids[bar1_idx]
                
                # BAR1 degradation rate constant (alpha-factor proteolysis)
                # Based on the paper, BAR1 rapidly degrades alpha-factor
                k_bar1 = 0.5  # Degradation rate constant
                
                if np.sum(active_mask) < 0.5 * grid_height * grid_width:
                    # Calculate degradation only for active regions
                    for i, j in zip(*active_indices):
                        if alpha_grid[i, j] > 0 and bar1_grid[i, j] > 0:
                            # Calculate degradation rate (proportional to both concentrations)
                            degradation_rate = k_bar1 * bar1_grid[i, j] * alpha_grid[i, j]
                            
                            # Update alpha-factor derivative
                            start_idx = alpha_idx * grid_height * grid_width + i * grid_width + j
                            derivatives[start_idx] -= degradation_rate
                else:
                    # Vectorized calculation for the entire grid
                    degradation_rates = k_bar1 * bar1_grid * alpha_grid
                    start_idx = alpha_idx * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] -= degradation_rates.flatten()

            # Handle GH3-mediated IAA degradation (conjugation to IAA-Asp)
            if all(m in diffusible_indices for m in [IAA, GH3]):
                iaa_idx = diffusible_indices[IAA]
                gh3_idx = diffusible_indices[GH3]
                
                iaa_grid = diffusible_grids[iaa_idx]
                gh3_grid = diffusible_grids[gh3_idx]
                
                # GH3 conjugation rate constant
                # From the paper, GH3.3 converts IAA to IAA-Asp which is signaling-inactive
                k_gh3 = 0.3  # Conjugation rate constant
                
                if np.sum(active_mask) < 0.5 * grid_height * grid_width:
                    # Calculate conjugation only for active regions
                    for i, j in zip(*active_indices):
                        if iaa_grid[i, j] > 0 and gh3_grid[i, j] > 0:
                            # Calculate conjugation rate (proportional to both concentrations)
                            degradation_rate = k_gh3 * gh3_grid[i, j] * iaa_grid[i, j]
                            
                            # Update IAA derivative
                            start_idx = iaa_idx * grid_height * grid_width + i * grid_width + j
                            derivatives[start_idx] -= degradation_rate
                else:
                    # Vectorized calculation for the entire grid
                    degradation_rates = k_gh3 * gh3_grid * iaa_grid
                    start_idx = iaa_idx * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] -= degradation_rates.flatten()            

            return derivatives
        
        return dydt
        
    def _get_initial_state(self) -> np.ndarray:
        """
        Get the initial state for the simulation.
        
        Returns:
            Array of initial state values (flattened)
        """
        # Grid dimensions
        grid_height, grid_width = self.grid_size
        
        # Diffusible molecules
        diffusible_molecules = [ALPHA, IAA, BETA, BAR1, GH3]
        
        # Reporter molecules
        reporter_molecules = [GFP, VENUS]
        
        # Start with diffusible molecule grids
        initial_state = []
        
        for molecule in diffusible_molecules:
            initial_state.append(self.initial_molecule_grids[molecule].flatten())
        
        # Add reporter molecule grids (initialized to 0)
        for molecule in reporter_molecules:
            initial_state.append(self.initial_molecule_grids[molecule].flatten())
        
        # Add strain population grids and internal state grids
        for strain_idx, strain in enumerate(self.strains):
            # Add population grid
            initial_state.append(self.strain_grids[strain_idx].flatten())
            
            # Add internal state grids (3 per strain, initialized to 0)
            for _ in range(3):
                initial_state.append(np.zeros(grid_height * grid_width))
        
        return np.concatenate(initial_state)
    
    # Only the simulate method needs to be changed in your simulation_v6.py
    # This is the updated simulate method that fixes the index error issue

    def simulate(self, n_time_points: int = 100, use_optimized: bool = True) -> Dict:
        """
        Run the spatial simulation with strain-specific growth parameters.
        
        Args:
            n_time_points: Number of time points to output
            use_optimized: Whether to use the optimized ODE system
            
        Returns:
            Dictionary with simulation results
        """
        grid_height, grid_width = self.grid_size

        print(f"Starting spatial simulation with {len(self.strains)} strains...")
        start_time = time.time()
        
        # Choose which ODE system to use
        system = self._build_optimized_spatial_ode_system()
        
        # Get initial state
        y0 = self._get_initial_state()
        
        # Set more conservative solver parameters to improve stability
        try:
            # Run simulation with more conservative parameters
            sol = solve_ivp(
                fun=system,
                t_span=self.time_span,
                y0=y0,
                method='BDF',  # Better for stiff equations
                rtol=1e-3,     # Relaxed relative tolerance
                atol=1e-5,     # Relaxed absolute tolerance
                max_step=0.1,  # Limit maximum step size
                t_eval=np.linspace(self.time_span[0], self.time_span[1], n_time_points),
                dense_output=True  # Enable dense output for more reliable time point extraction
            )
            
            # Check if solver successfully reached the end time
            if not sol.success:
                print(f"Warning: Solver did not converge: {sol.message}")
                
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            # Create a minimal solution with at least one time point
            print("Creating minimal solution...")
            # Create a basic solution structure
            sol = type('obj', (object,), {
                't': np.array([self.time_span[0]]),
                'y': np.zeros((len(y0), 1)),
                'success': False,
                'message': str(e)
            })
            sol.y[:, 0] = y0  # Set initial state
            
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")

        # Diffusible molecules
        diffusible_molecules = [ALPHA, IAA, BETA, BAR1, GH3]
        n_diffusible = len(diffusible_molecules)
        
        # Reporter molecules
        reporter_molecules = [GFP, VENUS]
        n_reporters = len(reporter_molecules)
        
        # Extract results
        results = {
            't': sol.t,  # Use actual time points from solution
            'molecule_grids': {},
            'population_grids': [],
            'strain_state_grids': []
        }
        
        # Get actual number of time points (may be less than requested)
        actual_n_time_points = len(sol.t)
        print(f"Simulation produced {actual_n_time_points} time points (requested: {n_time_points})")
        
        # Process simulation results
        state_idx = 0
        
        # Extract diffusible molecule grids
        for molecule in diffusible_molecules:
            molecule_data = []
            for t_idx in range(actual_n_time_points):
                grid = sol.y[state_idx:state_idx + grid_height*grid_width, t_idx].reshape(grid_height, grid_width)
                molecule_data.append(grid)
            
            results['molecule_grids'][molecule] = molecule_data
            state_idx += grid_height*grid_width
        
        # Extract reporter molecule grids
        for molecule in reporter_molecules:
            molecule_data = []
            for t_idx in range(actual_n_time_points):
                grid = sol.y[state_idx:state_idx + grid_height*grid_width, t_idx].reshape(grid_height, grid_width)
                molecule_data.append(grid)
            
            results['molecule_grids'][molecule] = molecule_data
            state_idx += grid_height*grid_width
        
        # Extract strain population grids and internal state grids
        for strain_idx in range(len(self.strains)):
            # Extract population grid
            pop_data = []
            for t_idx in range(actual_n_time_points):
                grid = sol.y[state_idx:state_idx + grid_height*grid_width, t_idx].reshape(grid_height, grid_width)
                pop_data.append(grid)
            
            results['population_grids'].append(pop_data)
            state_idx += grid_height*grid_width
            
            # Extract internal state grids (3 per strain)
            strain_states = []
            for _ in range(3):  # input_sensing, signal_processing, output
                state_data = []
                for t_idx in range(actual_n_time_points):
                    grid = sol.y[state_idx:state_idx + grid_height*grid_width, t_idx].reshape(grid_height, grid_width)
                    state_data.append(grid)
                
                strain_states.append(state_data)
                state_idx += grid_height*grid_width
            
            results['strain_state_grids'].append(strain_states)
        
        return results

    def create_animation(self, results: Dict, molecule: str = None, strain_idx: int = None, 
                       time_indices: List[int] = None, interval: int = 200, cmap: str = 'viridis',
                       vmin: float = None, vmax: float = None):
        """
        Create an animation of a molecule or strain population over time.
        
        Args:
            results: Simulation results from the simulate method
            molecule: Name of molecule to animate (if None, use strain_idx)
            strain_idx: Index of strain to animate (if None, use molecule)
            time_indices: List of time indices to animate (if None, use all)
            interval: Time between frames in milliseconds
            cmap: Colormap to use
            vmin, vmax: Min and max values for colormap

        Returns:
            matplotlib.animation.FuncAnimation object
        """
        if molecule is None and strain_idx is None:
            raise ValueError("Either molecule or strain_idx must be specified")
        
        # Get time points
        t = results['t']
        
        if time_indices is None:
            time_indices = range(len(t))
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set up initial plot
        if molecule is not None:
            data = results['molecule_grids'][molecule]
            title = f"{molecule} - t = {t[0]:.2f} h"
        else:
            data = results['population_grids'][strain_idx]
            title = f"{self.strains[strain_idx].strain_id} Population - t = {t[0]:.2f} h"
        
        # Determine vmin and vmax if not provided
        if vmin is None:
            vmin = min(np.min(data[idx]) for idx in time_indices)
        if vmax is None:
            vmax = max(np.max(data[idx]) for idx in time_indices)
        
        # Create the initial plot
        im = ax.imshow(data[time_indices[0]], cmap=cmap, interpolation='nearest',
                    vmin=vmin, vmax=vmax)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Animation update function
        def update(frame):
            idx = time_indices[frame]
            im.set_data(data[idx])
            
            if molecule is not None:
                ax.set_title(f"{molecule} - t = {t[idx]:.2f} h")
            else:
                ax.set_title(f"{self.strains[strain_idx].strain_id} Population - t = {t[idx]:.2f} h")
            
            return [im]
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(time_indices),
                                    interval=interval, blit=True)
        
        plt.tight_layout()
        return anim
        
    def plot_spatial_results(self, results: Dict, time_idx: int = -1, 
                           molecules: List[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot spatial results at a specific time point.
        
        Args:
            results: Simulation results from the simulate method
            time_idx: Time index to plot (default: last time point)
            molecules: List of molecules to plot (default: all)
            figsize: Figure size (width, height) in inches
        """
        if molecules is None:
            molecules = list(results['molecule_grids'].keys())
        
        n_molecules = len(molecules)
        n_strains = len(self.strains)
        
        # Calculate grid layout
        n_plots = n_molecules + n_strains
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # Get time point
        time_point = results['t'][time_idx]
        
        # Plot molecule grids
        for i, molecule in enumerate(molecules):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            grid = results['molecule_grids'][molecule][time_idx]
            
            # Create heatmap
            im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Set title
            ax.set_title(f"{molecule} (t = {time_point:.2f} h)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        
        # Plot strain population grids
        for i, strain in enumerate(self.strains):
            row, col = (i + n_molecules) // n_cols, (i + n_molecules) % n_cols
            ax = axes[row, col]
            
            grid = results['population_grids'][i][time_idx]
            
            # Create heatmap
            im = ax.imshow(grid, cmap='plasma', interpolation='nearest')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Set title
            ax.set_title(f"{strain.strain_id} Population (t = {time_point:.2f} h)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        
        # Hide unused subplots
        for i in range(n_plots, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        return fig

def plot_strain_growth(results, model, figsize=(12, 8), outputdir = None, average_over_space=True, specific_locations=None):
    """
    Plot the growth of each strain over time from simulation results.
    
    Args:
        results: Simulation results from the simulate method
        model: The SpatialMultiStrainModel instance
        figsize: Figure size (width, height) in inches
        average_over_space: If True, plot the average population across the entire grid
                           If False, plot the total population across the entire grid
        specific_locations: List of (row, col) coordinates to plot strain growth at specific locations
                           If None, only the spatial average/total is plotted
    
    Returns:
        matplotlib figure instance
    """
    t = results['t']
    n_strains = len(model.strains)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color map for strains
    colors = plt.cm.tab10.colors
    
    # Plot average/total growth for each strain
    for strain_idx, strain in enumerate(model.strains):
        strain_id = strain.strain_id
        populations = results['population_grids'][strain_idx]
        
        if average_over_space:
            # Calculate mean population across the grid for each time point
            values = [np.mean(pop) for pop in populations]
            label = f"{strain_id} (Avg)"
        else:
            # Calculate total population across the grid for each time point
            values = [np.sum(pop) for pop in populations]
            label = f"{strain_id} (Total)"
        
        # Plot the strain growth
        ax.plot(t, values, label=label, color=colors[strain_idx % len(colors)], 
                linewidth=2.5, marker='', alpha=0.8)
    
    # Plot growth at specific locations if provided
    if specific_locations:
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'd', 'x']
        
        for loc_idx, (row, col) in enumerate(specific_locations):
            for strain_idx, strain in enumerate(model.strains):
                strain_id = strain.strain_id
                populations = results['population_grids'][strain_idx]
                
                # Extract population at this specific location over time
                values = [pop[row, col] for pop in populations]
                
                # Use different line style for different locations
                ls = linestyles[loc_idx % len(linestyles)]
                marker = markers[loc_idx % len(markers)]
                
                # Plot with thinner lines and markers for specific locations
                ax.plot(t, values, 
                        label=f"{strain_id} at ({row},{col})", 
                        color=colors[strain_idx % len(colors)],
                        linestyle=ls, 
                        marker=marker, 
                        markersize=5,
                        markevery=max(1, len(t)//10),  # Show marker every few points
                        linewidth=1.5,
                        alpha=0.6)
    
    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (hours)', fontsize=12)
    
    if average_over_space:
        ax.set_ylabel('Average Population Density', fontsize=12)
        ax.set_title('Average Strain Growth Over Time', fontsize=14)
    else:
        ax.set_ylabel('Total Population', fontsize=12)
        ax.set_title('Total Strain Growth Over Time', fontsize=14)
    
    # Add legend with a reasonable number of columns
    n_items = n_strains * (1 + len(specific_locations) if specific_locations else 1)
    n_cols = max(1, min(3, n_items // 4 + 1))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=n_cols)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    return fig

# Example usage:
# fig = plot_strain_growth(results, model)
# fig = plot_strain_growth(results, model, average_over_space=False)  # For total rather than average
# fig = plot_strain_growth(results, model, specific_locations=[(20, 20), (30, 40)])  # With specific locations

def create_growth_dashboard(results, model, time_points=None, figsize=(15, 12)):
    """
    Create a comprehensive dashboard to visualize strain growth and behavior.
    
    Args:
        results: Simulation results from the simulate method
        model: The SpatialMultiStrainModel instance
        time_points: List of time indices to display spatial distributions
                    If None, 4 evenly spaced time points will be selected
        figsize: Figure size for the entire dashboard
        
    Returns:
        matplotlib figure instance
    """
    t = results['t']
    n_strains = len(model.strains)
    
    # Select time points if not provided
    if time_points is None:
        time_points = [0, len(t)//3, 2*len(t)//3, -1]  # Beginning, 1/3, 2/3, end
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2)
    
    # Plot 1: Average strain growth
    ax1 = fig.add_subplot(gs[0, 0])
    for strain_idx, strain in enumerate(model.strains):
        strain_id = strain.strain_id
        populations = results['population_grids'][strain_idx]
        
        # Calculate mean population across the grid for each time point
        values = [np.mean(pop) for pop in populations]
        
        # Plot the strain growth
        ax1.plot(t, values, label=f"{strain_id}", linewidth=2)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Average Population Density')
    ax1.set_title('Average Strain Growth Over Time')
    ax1.legend()
    
    # Plot 2: Total strain growth
    ax2 = fig.add_subplot(gs[0, 1])
    for strain_idx, strain in enumerate(model.strains):
        strain_id = strain.strain_id
        populations = results['population_grids'][strain_idx]
        
        # Calculate total population across the grid for each time point
        values = [np.sum(pop) for pop in populations]
        
        # Plot the strain growth
        ax2.plot(t, values, label=f"{strain_id}", linewidth=2)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Total Population')
    ax2.set_title('Total Strain Population Over Time')
    ax2.legend()
    
    # Plot 3: Relative proportions (percentage of total)
    ax3 = fig.add_subplot(gs[1, 0])
    total_by_time = np.zeros(len(t))
    
    # Calculate total population at each time point
    for strain_idx in range(n_strains):
        populations = results['population_grids'][strain_idx]
        for t_idx, pop in enumerate(populations):
            total_by_time[t_idx] += np.sum(pop)
    
    # Plot proportion for each strain
    for strain_idx, strain in enumerate(model.strains):
        strain_id = strain.strain_id
        populations = results['population_grids'][strain_idx]
        
        # Calculate proportion of total at each time point
        proportions = []
        for t_idx, pop in enumerate(populations):
            if total_by_time[t_idx] > 0:
                proportions.append(100 * np.sum(pop) / total_by_time[t_idx])
            else:
                proportions.append(0)
        
        # Plot the strain proportion
        ax3.plot(t, proportions, label=f"{strain_id}", linewidth=2)
    
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Percentage of Total Population')
    ax3.set_title('Relative Strain Proportions Over Time')
    ax3.set_ylim(0, 100)
    ax3.legend()
    
    # Plot 4: Growth rates
    ax4 = fig.add_subplot(gs[1, 1])
    for strain_idx, strain in enumerate(model.strains):
        strain_id = strain.strain_id
        populations = results['population_grids'][strain_idx]
        
        # Calculate average population at each time point
        avg_pop = [np.mean(pop) for pop in populations]
        
        # Calculate growth rate: (dP/dt)/P
        growth_rates = []
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            if avg_pop[i-1] > 0:
                rate = (avg_pop[i] - avg_pop[i-1]) / (dt * avg_pop[i-1])
                growth_rates.append(rate)
            else:
                growth_rates.append(0)
        
        # Plot growth rate (skip first point since we don't have a rate for it)
        ax4.plot(t[1:], growth_rates, label=f"{strain_id}", linewidth=2)
    
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Relative Growth Rate (1/hour)')
    ax4.set_title('Strain Growth Rates Over Time')
    ax4.legend()
    
    # Plot 5: Spatial distribution at selected time points
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    ax5.set_title('Spatial Distribution of Strains at Selected Time Points')
    
    # Create small subplots for each time point
    inner_gs = gs[2, :].subgridspec(1, len(time_points), wspace=0.1)
    
    for i, t_idx in enumerate(time_points):
        inner_ax = fig.add_subplot(inner_gs[0, i])
        
        # Create a composite image showing all strains
        composite = np.zeros(model.grid_size + (3,))  # RGB image
        
        # Normalize each strain to [0, 1] and assign a color
        for strain_idx, strain in enumerate(model.strains):
            pop = results['population_grids'][strain_idx][t_idx]
            max_val = np.max(pop) if np.max(pop) > 0 else 1
            normalized = pop / max_val
            
            # Assign color (cycle through primary and secondary colors)
            if strain_idx % 6 == 0:  # Red
                composite[:, :, 0] += normalized
            elif strain_idx % 6 == 1:  # Green
                composite[:, :, 1] += normalized
            elif strain_idx % 6 == 2:  # Blue
                composite[:, :, 2] += normalized
            elif strain_idx % 6 == 3:  # Yellow
                composite[:, :, 0] += normalized
                composite[:, :, 1] += normalized
            elif strain_idx % 6 == 4:  # Magenta
                composite[:, :, 0] += normalized
                composite[:, :, 2] += normalized
            else:  # Cyan
                composite[:, :, 1] += normalized
                composite[:, :, 2] += normalized
        
        # Clip values to [0, 1]
        composite = np.clip(composite, 0, 1)
        
        # Display the composite image
        inner_ax.imshow(composite)
        inner_ax.set_title(f"t = {t[t_idx]:.2f} h")
        inner_ax.set_xticks([])
        inner_ax.set_yticks([])
    
    plt.tight_layout()
    return fig


def create_strain_library():
        """
        Create a library of all strains from the paper with growth parameters.
        
        Returns:
            Dictionary mapping strain IDs to StrainParameters objects
        """
        # Growth parameters dictionary
        growth_params = {
            'beta->alpha': {'strain': 'beta->alpha','k': 10.156865998540457,'r': 1.3877019835236286,'A': 3.811240700492937e-08,'lag': 8.065932810614516,'doubling_time': 0.4994928225150461,'r_squared': 0.9978592780369944},
            'alpha->venus': {'strain': 'alpha->venus','k': 10.156881939374474,'r': 1.3876456633006105,'A': 3.814254538133409e-08,'lag': 8.06591253743179,'doubling_time': 0.4995130953756935,'r_squared': 0.9978586598580573},
            'alpha->alpha': {'strain': 'alpha->alpha','k': 10.030632161041991,'r': 2.0645209329326293,'A': 1.0705887173990125e-08,'lag': 7.994289828061679,'doubling_time': 0.33574238434838116,'r_squared': 0.9999048003122031},
            'alpha->IAA': {'strain': 'alpha->IAA','k': 10.040333303759686,'r': 1.577804152831291,'A': 6.43804253236298e-09,'lag': 7.973971461454354,'doubling_time': 0.4393112917823972,'r_squared': 0.9999163020148677},
            'beta->IAA': {'strain': 'beta->IAA',
                'k': 10.040307128578494,
                'r': 1.9629605025697323,
                'A': 4.000104513358863e-08,
                'lag': 8.156733057057686,
                'doubling_time': 0.35311315721968883,
                'r_squared': 0.999744136703164},
            'IAA->GFP': {'strain': 'IAA->GFP',
                'k': 10.252654230445089,
                'r': 0.42549511085479685,
                'A': 0.0002582042194400119,
                'lag': 7.805876037701137,
                'doubling_time': 1.6290367688772023,
                'r_squared': 0.9955579879121793},
            'IAA->IAA': {'strain': 'IAA->IAA',
                'k': 10.009222723768396,
                'r': 0.9071483893517285,
                'A': 1.338045537808413e-09,
                'lag': 7.970174930108886,
                'doubling_time': 0.7640945943312384,
                'r_squared': 0.9999395888033465},
            'IAA->alpha': {'strain': 'IAA->alpha',
                'k': 15.947503133990207,
                'r': 0.6144061444836784,
                'A': 2.082679648386623e-08,
                'lag': 10.04798809953061,
                'doubling_time': 1.1281579567249245,
                'r_squared': 0.9904120220822631}}
                    
        strains = {}
        strains['IAA-|GFP'] = StrainParameters(
            strain_id='IAA-|GFP',
            input_molecule=IAA,
            regulation_type=REPRESSION,
            output_molecule=GFP,
            k1=1.76e6, d1=3.13e2, k2=8.29e5, K=2.57e2, n=0.89,
            d2=4.41e5, k3=2.12e5, d3=7.46e4, b=8.68e2,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA-|2xGFP'] = StrainParameters(
            strain_id='IAA-|2xGFP',
            input_molecule=IAA,
            regulation_type=REPRESSION,
            output_molecule=GFP,
            k1=0.69, d1=4.16e2, k2=2.77e2, K=1.79, n=1.011,
            d2=11.40, k3=0.0011, d3=0.49, b=0.0049,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA->GFP'] = StrainParameters(
            strain_id='IAA->GFP',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=GFP,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=3.15e4, d3=1.66e6, b=1.46e4,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha-|GFP'] = StrainParameters(
            strain_id='alpha-|GFP',
            input_molecule=ALPHA,
            regulation_type=REPRESSION,
            output_molecule=GFP,
            k1=6.05, d1=59.54, k2=10.61, K=0.67, n=0.80,
            d2=0.82, k3=3.66e-4, d3=1.10, b=0.0032,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->venus'] = StrainParameters(
            strain_id='alpha->venus',
            input_molecule=ALPHA,
            regulation_type=ACTIVATION,
            output_molecule=VENUS,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=1.62e4, d3=2.15e6, b=5.96e3,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta->GFP'] = StrainParameters(
            strain_id='beta->GFP',
            input_molecule=BETA,
            regulation_type=ACTIVATION,
            output_molecule=GFP,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=0.089, d3=0.16, b=2.155e-4,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta-|GFP'] = StrainParameters(
            strain_id='beta-|GFP',
            input_molecule=BETA,
            regulation_type=REPRESSION,
            output_molecule=GFP,
            k1=0.89, d1=0.17, k2=174.32, K=6.65, n=2.18,
            d2=0.56, k3=1.5e-4, d3=0.55, b=3.77e-4,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->IAA'] = StrainParameters(
            strain_id='alpha->IAA',
            input_molecule=ALPHA,
            regulation_type=ACTIVATION,
            output_molecule=IAA,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=566.24, d3=0.575, b=55.83,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->IAA_NL'] = StrainParameters(
            strain_id='alpha->IAA_NL',
            input_molecule=ALPHA,
            regulation_type=ACTIVATION,
            output_molecule=IAA,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=566.24, d3=0.575, b=0,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta->BAR1'] = StrainParameters(
            strain_id='beta->BAR1',
            input_molecule=BETA,
            regulation_type=ACTIVATION,
            output_molecule=BAR1,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=26.038, d3=1.92e-6, b=0.35,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta->IAA'] = StrainParameters(
            strain_id='beta->IAA',
            input_molecule=BETA,
            regulation_type=ACTIVATION,
            output_molecule=IAA,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=3.48e3, d3=0.16, b=0.21,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta->alpha'] = StrainParameters(
            strain_id='beta->alpha',
            input_molecule=BETA,
            regulation_type=ACTIVATION,
            output_molecule=ALPHA,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=121.6, d3=0.062, b=0.14,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA->alpha'] = StrainParameters(
            strain_id='IAA->alpha',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=ALPHA,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=2.285, d3=0.28, b=0.74,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )
        strains['IAA->alpha_NL'] = StrainParameters(
            strain_id='IAA->alpha_NL',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=ALPHA,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=2.285, d3=0.28, b=0,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )
        strains['beta->GH3'] = StrainParameters(
            strain_id='beta->GH3',
            input_molecule=BETA,
            regulation_type=ACTIVATION,
            output_molecule=GH3,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=109.96, d3=36.71, b=1.6e-4,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA->BAR1'] = StrainParameters(
            strain_id='IAA->BAR1',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=BAR1,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=1.89, d3=1.83e-13, b=0.365,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->GH3'] = StrainParameters(
            strain_id='alpha->GH3',
            input_molecule=ALPHA,
            regulation_type=ACTIVATION,
            output_molecule=GH3,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=582.32, d3=368.67, b=1.17e-14,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha-|IAA'] = StrainParameters(
            strain_id='alpha-|IAA',
            input_molecule=ALPHA,
            regulation_type=REPRESSION,
            output_molecule=IAA,
            k1=6.05, d1=59.54, k2=10.61, K=0.67, n=0.80,
            d2=0.82, k3=4.09, d3=8.74e-11, b=47.78,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta-|IAA'] = StrainParameters(
            strain_id='beta-|IAA',
            input_molecule=BETA,
            regulation_type=REPRESSION,
            output_molecule=IAA,
            k1=0.89, d1=0.17, k2=174.32, K=6.65, n=2.18,
            d2=0.56, k3=2.024e11, d3=4.29e10, b=1.85e9,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta-|alpha'] = StrainParameters(
            strain_id='beta-|alpha',
            input_molecule=BETA,
            regulation_type=REPRESSION,
            output_molecule=ALPHA,
            k1=0.89, d1=0.17, k2=174.32, K=6.65, n=2.18,
            d2=0.56, k3=0.077, d3=1.77, b=0.18,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA-|alpha'] = StrainParameters(
            strain_id='IAA-|alpha',
            input_molecule=IAA,
            regulation_type=REPRESSION,
            output_molecule=ALPHA,
            k1=1.76e6, d1=3.13e2, k2=8.29e5, K=2.57e2, n=0.89,
            d2=4.41e5, k3=471.73, d3=0.41, b=1.34,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->alpha'] = StrainParameters(
            strain_id='alpha->alpha',
            input_molecule=ALPHA,
            regulation_type=ACTIVATION,
            output_molecule=ALPHA,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=419.52, d3=2.10e4, b=2.32e4,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->alpha_NL'] = StrainParameters(
            strain_id='alpha->alpha_NL',
            input_molecule=ALPHA,
            regulation_type=ACTIVATION,
            output_molecule=ALPHA,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=419.52, d3=2.10e4, b=0,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA->IAA'] = StrainParameters(
            strain_id='IAA->IAA',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=IAA,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=775.05, d3=0.84, b=780.90,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )
        
        strains['IAA->IAA_NL'] = StrainParameters(
            strain_id='IAA->IAA_NL',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=IAA,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=775.05, d3=0.84, b=0,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha-|BAR1'] = StrainParameters(
            strain_id='alpha-|BAR1',
            input_molecule=ALPHA,
            regulation_type=REPRESSION,
            output_molecule=BAR1,
            k1=6.05, d1=59.54, k2=10.61, K=0.67, n=0.80,
            d2=0.82, k3=0.0014, d3=3.74e7, b=1.096e8,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA-|GH3'] = StrainParameters(
            strain_id='IAA-|GH3',
            input_molecule=IAA,
            regulation_type=REPRESSION,
            output_molecule=GH3,
            k1=1.76e6, d1=3.13e2, k2=8.29e5, K=2.57e2, n=0.89,
            d2=4.41e5, k3=225.51, d3=42.46, b=3.14e-6,
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        return strains
        
def plot_molecule_concentrations(results, molecules=None):
    """
    Plot the average concentration of each molecule over time.
    
    Args:
        results: Simulation results
        molecules: List of molecules to plot (if None, plot all)
        
    Returns:
        matplotlib figure
    """
    if molecules is None:
        molecules = list(results['molecule_grids'].keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each molecule's average concentration
    for molecule in molecules:
        if molecule in results['molecule_grids']:
            # Calculate average concentration over the grid for each time point
            molecule_grids = results['molecule_grids'][molecule]
            avg_conc = [np.mean(grid) for grid in molecule_grids]
            
            # Plot
            ax.plot(results['t'], avg_conc, label=f'{molecule}', linewidth=2, marker='o', markersize=4)
    
    # Add labels and legend
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Average Concentration', fontsize=12)
    ax.set_title('Molecule Concentrations Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

