import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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
    
    def __init__(self, grid_size=(50, 50), dx=0.1):
        """
        Initialize the spatial model.
        
        Args:
            grid_size: Tuple of (height, width) for the 2D grid
            dx: Grid spacing (in mm)
        """
        self.grid_size = grid_size
        self.dx = dx
        self.strains = []
        self.initial_molecule_grids = {}
        self.strain_grids = []  # List of initial population grids for each strain
        self.time_span = (0, 10)  # Default simulation time (hours)
        
        # Diffusion coefficients (mm²/hour)
        self.diffusion_coefficients = {
            ALPHA: 0.15,  # Alpha factor
            IAA: 0.67,    # Auxin
            BETA: 0.2,    # Beta estradiol
            BAR1: 0.05,   # Enzymes diffuse slower
            GH3: 0.05
        }
        
        # Default growth parameters - will be overridden by strain-specific parameters
        self.default_carrying_capacity = 100.0  # Default carrying capacity
        
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
    
    def _build_spatial_ode_system_with_competition(self) -> Callable:
        """
        Build the spatial ODE system for the model with competition between strains.
        Strains will compete for a shared carrying capacity at each spatial location.
        Uses strain-specific growth parameters.
        
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
        # (3 internal states: input sensing, signal processing, output)
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
                'k': strain.k,          # Carrying capacity
                'r': strain.r,          # Growth rate
                'A': strain.A,          # Initial population fraction (not used here)
                'lag': strain.lag       # Lag time (hours)
            })
        
        # Calculate the total number of state variables
        n_states = (n_diffusible + n_reporters) * grid_height * grid_width  # Molecule grids
        n_states += len(self.strains) * (grid_height * grid_width + n_internal_states * grid_height * grid_width)  # Strain grids + internal states
        
        def dydt(t, y):
            """
            Compute derivatives for all state variables.
            
            Args:
                t: Current time
                y: Current state values (flattened)
                
            Returns:
                Array of derivatives (flattened)
            """
            # Initialize derivatives array
            derivatives = np.zeros_like(y)
            
            # Reshape the state array to get the molecule and strain grids
            # First, extract the diffusible molecule grids
            diffusible_grids = []
            state_idx = 0
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
                # Extract population grid
                pop_grid = y[state_idx:state_idx + grid_height*grid_width].reshape(grid_height, grid_width)
                strain_pop_grids.append(pop_grid)
                state_idx += grid_height*grid_width
                
                # Extract internal state grids (3 per strain)
                strain_states = []
                for j in range(n_internal_states):
                    state_grid = y[state_idx:state_idx + grid_height*grid_width].reshape(grid_height, grid_width)
                    strain_states.append(state_grid)
                    state_idx += grid_height*grid_width
                
                strain_internal_states.append(strain_states)
            
            # Calculate diffusion for each diffusible molecule
            for mol_idx, D in enumerate(D_values):
                if D > 0:  # Only apply diffusion if coefficient is positive
                    grid = diffusible_grids[mol_idx]
                    
                    # Apply finite difference method for diffusion
                    # Central difference for interior points
                    laplacian = np.zeros_like(grid)
                    
                    # Interior points
                    laplacian[1:-1, 1:-1] = (
                        grid[:-2, 1:-1] +  # Top
                        grid[2:, 1:-1] +   # Bottom
                        grid[1:-1, :-2] +  # Left
                        grid[1:-1, 2:] -   # Right
                        4 * grid[1:-1, 1:-1]  # Center
                    ) / (self.dx**2)
                    
                    # No-flux boundary conditions
                    # Top and bottom boundaries
                    laplacian[0, 1:-1] = (grid[1, 1:-1] - grid[0, 1:-1]) / (self.dx**2)
                    laplacian[-1, 1:-1] = (grid[-2, 1:-1] - grid[-1, 1:-1]) / (self.dx**2)
                    
                    # Left and right boundaries
                    laplacian[1:-1, 0] = (grid[1:-1, 1] - grid[1:-1, 0]) / (self.dx**2)
                    laplacian[1:-1, -1] = (grid[1:-1, -2] - grid[1:-1, -1]) / (self.dx**2)
                    
                    # Corner points (average of adjacent boundary points)
                    laplacian[0, 0] = (laplacian[0, 1] + laplacian[1, 0]) / 2
                    laplacian[0, -1] = (laplacian[0, -2] + laplacian[1, -1]) / 2
                    laplacian[-1, 0] = (laplacian[-2, 0] + laplacian[-1, 1]) / 2
                    laplacian[-1, -1] = (laplacian[-2, -1] + laplacian[-1, -2]) / 2
                    
                    # Apply diffusion
                    diffusion_deriv = D * laplacian
                    
                    # Update derivatives for this molecule
                    start_idx = mol_idx * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] = diffusion_deriv.flatten()
            
            # Calculate the total population at each grid location
            total_population = np.zeros((grid_height, grid_width))
            for pop_grid in strain_pop_grids:
                total_population += pop_grid
            
            # Calculate strain dynamics and their effects on molecule concentrations
            for strain_idx, strain_params in enumerate(self.strains):
                # Get population grid and internal state grids
                pop_grid = strain_pop_grids[strain_idx]
                strain_states = strain_internal_states[strain_idx]  # [input_sensing, signal_processing, output]
                
                # Get strain-specific growth parameters
                growth_params = strain_growth_params[strain_idx]
                k = growth_params['k']          # Carrying capacity
                r = growth_params['r']          # Growth rate
                lag = growth_params['lag']      # Lag time
                
                # Calculate logistic growth with lag phase
                # If t < lag, no growth occurs
                # If t >= lag, logistic growth occurs
                if t < lag:
                    # During lag phase, no growth
                    pop_deriv = np.zeros_like(pop_grid)
                else:
                    # After lag phase, logistic growth with competition
                    # dp/dt = r * p * (1 - total_pop/K)
                    pop_deriv = r * pop_grid * (1 - total_population / k)
                
                # Get input molecule grid
                input_molecule = strain_params.input_molecule
                if input_molecule in diffusible_molecules:
                    input_grid = diffusible_grids[diffusible_indices[input_molecule]]
                else:
                    raise ValueError(f"Input molecule {input_molecule} not found in diffusible molecules")
                
                # Calculate internal state derivatives
                # Input sensing (x₁)
                input_sensing_deriv = strain_params.k1 * input_grid - strain_params.d1 * strain_states[0]
                strain_states = np.maximum(0, strain_states)
                
                # Signal processing (x₂)
                if strain_params.regulation_type == ACTIVATION:
                    # Activation: x₁ activates x₂ production
                    x1_n = np.power(strain_states[0], strain_params.n)
                    K_n = np.power(strain_params.K, strain_params.n)
                    hill_term = x1_n / (K_n + x1_n)
                    signal_processing_deriv = (strain_params.k2 * hill_term) - (strain_params.d2 * strain_states[1])
                else:
                    # Repression: x₁ represses x₂ production
                    x1_over_K_n = np.power(strain_states[0] / strain_params.K, strain_params.n)
                    hill_term = 1 / (1 + x1_over_K_n)
                    signal_processing_deriv = (strain_params.k2 * hill_term) - (strain_params.d2 * strain_states[1])
                
                # Output production (x₃)
                output_deriv = strain_params.b + strain_params.k3 * strain_states[1] - strain_params.d3 * strain_states[2]
                
                # Update molecule grids based on strain output
                output_molecule = strain_params.output_molecule
                
                # Output rate is proportional to population
                output_rate = pop_grid * (strain_params.b + strain_params.k3 * strain_states[1] - strain_params.d3 * strain_states[2])
                
                if output_molecule in diffusible_molecules:
                    # Update diffusible molecule derivatives
                    mol_idx = diffusible_indices[output_molecule]
                    start_idx = mol_idx * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] += output_rate.flatten()
                elif output_molecule in reporter_molecules:
                    # Update reporter molecule derivatives
                    mol_idx = reporter_indices[output_molecule]
                    start_idx = (n_diffusible + mol_idx) * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] += output_rate.flatten()
                    
                # Update population derivatives
                start_idx = (n_diffusible + n_reporters) * grid_height * grid_width + strain_idx * (1 + n_internal_states) * grid_height * grid_width
                derivatives[start_idx:start_idx + grid_height*grid_width] = pop_deriv.flatten()
                
                # Update internal state derivatives
                derivatives[start_idx + grid_height*grid_width:start_idx + 2*grid_height*grid_width] = input_sensing_deriv.flatten()
                derivatives[start_idx + 2*grid_height*grid_width:start_idx + 3*grid_height*grid_width] = signal_processing_deriv.flatten()
                derivatives[start_idx + 3*grid_height*grid_width:start_idx + 4*grid_height*grid_width] = output_deriv.flatten()
            
            # Apply effect of BAR1 and GH3 on alpha factor and auxin
            if BAR1 in diffusible_molecules and ALPHA in diffusible_molecules:
                bar1_grid = diffusible_grids[diffusible_indices[BAR1]]
                alpha_grid = diffusible_grids[diffusible_indices[ALPHA]]
                
                if np.any(bar1_grid > 0):
                    # BAR1 degrades alpha factor
                    bar1_effect = 0.1 * bar1_grid * alpha_grid
                    
                    # Update alpha factor derivatives
                    alpha_idx = diffusible_indices[ALPHA]
                    start_idx = alpha_idx * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] -= bar1_effect.flatten()
            
            if GH3 in diffusible_molecules and IAA in diffusible_molecules:
                gh3_grid = diffusible_grids[diffusible_indices[GH3]]
                iaa_grid = diffusible_grids[diffusible_indices[IAA]]
                
                if np.any(gh3_grid > 0):
                    # GH3 degrades auxin
                    gh3_effect = 0.1 * gh3_grid * iaa_grid
                    
                    # Update auxin derivatives
                    iaa_idx = diffusible_indices[IAA]
                    start_idx = iaa_idx * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] -= gh3_effect.flatten()
            
            # Basic degradation for all signaling molecules
            for molecule, idx in diffusible_indices.items():
                if molecule in [BAR1, GH3]:
                    # Apply basic degradation
                    start_idx = idx * grid_height * grid_width
                    derivatives[start_idx:start_idx + grid_height*grid_width] -= 0.05 * diffusible_grids[idx].flatten()
            
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
    
    def simulate(self, n_time_points: int = 100) -> Dict:
        """
        Run the spatial simulation with strain-specific growth parameters.
        
        Args:
            n_time_points: Number of time points to output
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Starting spatial simulation with {len(self.strains)} strains...")
        start_time = time.time()
        
        # Grid dimensions
        grid_height, grid_width = self.grid_size
        
        # Build ODE system with competition and strain-specific growth parameters
        system = self._build_spatial_ode_system_with_competition()
        
        # Get initial state
        y0 = self._get_initial_state()
        
        # Run simulation
        sol = solve_ivp(
            fun=system,
            t_span=self.time_span,
            y0=y0,
            method='RK45',             # Using RK45 (Dormand-Prince method)
            rtol=1e-3,                 # Relative tolerance (increase for speed)
            atol=1e-6,                 # Absolute tolerance 
            t_eval=np.linspace(self.time_span[0], self.time_span[1], n_time_points)
        )
        
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
            't': sol.t,
            'molecule_grids': {},
            'population_grids': [],
            'strain_state_grids': []
        }
        
        # Process simulation results
        state_idx = 0
        
        # Extract diffusible molecule grids
        for molecule in diffusible_molecules:
            molecule_data = []
            for t_idx in range(n_time_points):
                grid = sol.y[state_idx:state_idx + grid_height*grid_width, t_idx].reshape(grid_height, grid_width)
                molecule_data.append(grid)
            
            results['molecule_grids'][molecule] = molecule_data
            state_idx += grid_height*grid_width
        
        # Extract reporter molecule grids
        for molecule in reporter_molecules:
            molecule_data = []
            for t_idx in range(n_time_points):
                grid = sol.y[state_idx:state_idx + grid_height*grid_width, t_idx].reshape(grid_height, grid_width)
                molecule_data.append(grid)
            
            results['molecule_grids'][molecule] = molecule_data
            state_idx += grid_height*grid_width
        
        # Extract strain population grids and internal state grids
        for strain_idx in range(len(self.strains)):
            # Extract population grid
            pop_data = []
            for t_idx in range(n_time_points):
                grid = sol.y[state_idx:state_idx + grid_height*grid_width, t_idx].reshape(grid_height, grid_width)
                pop_data.append(grid)
            
            results['population_grids'].append(pop_data)
            state_idx += grid_height*grid_width
            
            # Extract internal state grids (3 per strain)
            strain_states = []
            for _ in range(3):  # input_sensing, signal_processing, output
                state_data = []
                for t_idx in range(n_time_points):
                    grid = sol.y[state_idx:state_idx + grid_height*grid_width, t_idx].reshape(grid_height, grid_width)
                    state_data.append(grid)
                
                strain_states.append(state_data)
                state_idx += grid_height*grid_width
            
            results['strain_state_grids'].append(strain_states)
        
        return results

def example_strain_competition():
    """
    Example demonstrating competition between strains with different growth parameters.
    
    This example places multiple strains in overlapping spatial regions to observe how
    strains with different growth parameters compete for the shared carrying capacity.
    """
    strain_library = create_strain_library()
    
    # Create model
    model = SpatialMultiStrainModel(grid_size=(50, 50), dx=0.1)
    
    # Set simulation time to capture lag phases and complete growth dynamics
    model.set_simulation_time(0, 36)
    
    # Select strains with different growth characteristics
    # Order by growth rate (r) from highest to lowest
    fast_strain = strain_library['alpha->alpha']    # r ≈ 2.49
    medium_strain = strain_library['beta->alpha']   # r ≈ 1.59 
    slow_strain = strain_library['IAA->GFP']        # r ≈ 0.43
    
    # Add strains to the model
    model.add_strain(fast_strain)    # Strain index 0
    model.add_strain(medium_strain)  # Strain index 1
    model.add_strain(slow_strain)    # Strain index 2
    
    # Create three regions:
    # 1. Region with only fast strain
    # 2. Region with medium and slow strains
    # 3. Region with all three strains competing
    
    # Region 1: Only fast strain (top left)
    model.place_strain(0, row=10, col=10, radius=7, concentration=1.0)
    
    # Region 2: Medium and slow strains (top right)
    model.place_strain(1, row=10, col=40, radius=7, concentration=1.0)
    model.place_strain(2, row=10, col=40, radius=7, concentration=1.0)
    
    # Region 3: All three strains (bottom center)
    model.place_strain(0, row=35, col=25, radius=7, concentration=1.0)
    model.place_strain(1, row=35, col=25, radius=7, concentration=1.0)
    model.place_strain(2, row=35, col=25, radius=7, concentration=1.0)
    
    # Add initial molecule concentrations to trigger behaviors
    model.place_molecule(ALPHA, row=10, col=10, radius=9, concentration=100.0)  # Region 1
    model.place_molecule(BETA, row=10, col=40, radius=9, concentration=50.0)    # Region 2
    model.place_molecule(IAA, row=10, col=40, radius=9, concentration=1000.0)   # Region 2
    
    # Molecules for region 3 (all strains)
    model.place_molecule(ALPHA, row=35, col=25, radius=9, concentration=100.0)
    model.place_molecule(BETA, row=35, col=25, radius=9, concentration=50.0)
    model.place_molecule(IAA, row=35, col=25, radius=9, concentration=1000.0)
    
    # Run simulation
    results = model.simulate(n_time_points=120)
    
    # Create visualization of the competition dynamics
    # Sample times to observe competition at different stages
    time_indices = [0, 30, 60, 90, 119]  # Beginning, 25%, 50%, 75%, end
    
    for time_idx in time_indices:
        time_point = results['t'][time_idx]
        print(f"Plotting competition results at t = {time_point:.2f} hours")
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot each strain
        for strain_idx in range(3):
            strain = model.strains[strain_idx]
            pop_grid = results['population_grids'][strain_idx][time_idx]
            
            # Plot on shared axes
            im = axes[strain_idx].imshow(pop_grid, cmap='viridis', vmin=0, vmax=model.strains[strain_idx].k,
                                        interpolation='nearest')
            
            plt.colorbar(im, ax=axes[strain_idx])
            
            # Add title with growth parameters
            axes[strain_idx].set_title(f"{strain.strain_id}\n" + 
                                      f"r = {strain.r:.2f}, lag = {strain.lag:.2f}h")
            axes[strain_idx].set_xlabel("X")
            axes[strain_idx].set_ylabel("Y")
        
        fig.suptitle(f"Competition at t = {time_point:.2f} hours", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # Create plots showing competition dynamics over time
    # Define three regions to track
    regions = [
        (slice(5, 15), slice(5, 15)),      # Region 1: top left
        (slice(5, 15), slice(35, 45)),     # Region 2: top right
        (slice(30, 40), slice(20, 30))     # Region 3: bottom center
    ]
    region_names = ["Region 1 (Fast only)", "Region 2 (Medium & Slow)", "Region 3 (All strains)"]
    
    # Create figure for growth curves in the different regions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for region_idx, (region_slice, region_name) in enumerate(zip(regions, region_names)):
        ax = axes[region_idx]
        
        # Plot each strain's average population in this region
        for strain_idx in range(3):
            strain = model.strains[strain_idx]
            populations = results['population_grids'][strain_idx]
            
            # Calculate average population in this region over time
            avg_pop = [np.mean(pop[region_slice]) for pop in populations]
            
            # Plot the strain growth curve
            ax.plot(results['t'], avg_pop, 
                    label=f"{strain.strain_id} (r={strain.r:.2f})", 
                    linewidth=2)
        
        # Add vertical lines for lag phases
        for strain_idx, strain in enumerate(model.strains[:3]):
            ax.axvline(x=strain.lag, color=f'C{strain_idx}', linestyle='--', alpha=0.5)
        
        # Calculate and plot total population
        total_pop = np.zeros_like(results['t'])
        for strain_idx in range(3):
            populations = results['population_grids'][strain_idx]
            for t_idx, pop in enumerate(populations):
                total_pop[t_idx] += np.mean(pop[region_slice])
        
        ax.plot(results['t'], total_pop, 'k--', label="Total population", linewidth=1.5)
        
        # Add carrying capacity reference line
        ax.axhline(y=model.strains[0].k, color='gray', linestyle=':', 
                  label=f"Carrying capacity (k={model.strains[0].k:.1f})")
        
        # Customize plot
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('Average Population Density', fontsize=10)
        ax.set_title(f"{region_name}\nStrain Competition Dynamics", fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        
        # Set y-axis limit a bit above carrying capacity
        ax.set_ylim(0, model.strains[0].k * 1.1)
    
    plt.tight_layout()
    plt.show()
    
    return results

    # The visualization functions remain the same


def create_strain_library():
    """
    Create a library of all strains from the paper with growth parameters.
    
    Returns:
        Dictionary mapping strain IDs to StrainParameters objects
    """
    # Growth parameters dictionary
    growth_params = {
        'beta->alpha': {'k': 101.89883870394586, 'r': 1.5910963028004033, 'A': 2.2704007163391717e-08, 'lag': 8.122765202090607},
        'alpha->venus': {'k': 101.89856092475966, 'r': 1.591129021157073, 'A': 2.2693810194718495e-08, 'lag': 8.122769309094306},
        'alpha->alpha': {'k': 100.0911089941252, 'r': 2.491985116331938, 'A': 1.0520282725483878e-09, 'lag': 7.695291121111312},
        'alpha->IAA': {'k': 100.18163801781024, 'r': 1.784158848520821, 'A': 5.434662380597071e-09, 'lag': 7.983240474337439},
        'beta->IAA': {'k': 100.05637394521705, 'r': 2.4267020233248986, 'A': 4.400807476684393e-09, 'lag': 8.128368528029435},
        'IAA->GFP': {'k': 102.52637659915042, 'r': 0.4254980744873518, 'A': 0.002833302911672342, 'lag': 8.024233468750994},
        'IAA->IAA': {'k': 100.0689455692613, 'r': 0.9316534320715953, 'A': 8.204277164775148e-09, 'lag': 7.975381367685189},
        'IAA->alpha': {'k': 156.60733118093165, 'r': 0.670893841196142, 'A': 3.4690886521760006e-08, 'lag': 10.126559816714993}
    }
    
    strains = {}
    
    # 3. IAA->GFP (Auxin activating GFP)
    strains['IAA->GFP'] = StrainParameters(
        strain_id='IAA->GFP',
        input_molecule=IAA,
        regulation_type=ACTIVATION,
        output_molecule=GFP,
        k1=8.95e4, d1=0.082, k2=1.73e7, K=1.4e7, n=0.836,
        d2=1.88e7, k3=3.15e4, d3=1.66e6, b=1.46e4,
        k=growth_params['IAA->GFP']['k'],
        r=growth_params['IAA->GFP']['r'],
        A=growth_params['IAA->GFP']['A'],
        lag=growth_params['IAA->GFP']['lag']
    )

    # 5. ALPHA->VENUS (Alpha activating VENUS)
    strains['alpha->venus'] = StrainParameters(
        strain_id='alpha->venus',
        input_molecule=ALPHA,
        regulation_type=ACTIVATION,
        output_molecule=VENUS,
        k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
        d2=1.56e5, k3=1.62e4, d3=2.15e6, b=5.96e3,
        k=growth_params['alpha->venus']['k'],
        r=growth_params['alpha->venus']['r'],
        A=growth_params['alpha->venus']['A'],
        lag=growth_params['alpha->venus']['lag']
    )

    # 12. beta->IAA (beta activating IAA)
    strains['beta->IAA'] = StrainParameters(
        strain_id='beta->IAA',
        input_molecule=BETA,
        regulation_type=ACTIVATION,
        output_molecule=IAA,
        k1=50.66, d1=25.86, k2=11, K=57.12, n=1.26,
        d2=110.43, k3=3.48e3, d3=0.16, b=0.21,
        k=growth_params['beta->IAA']['k'],
        r=growth_params['beta->IAA']['r'],
        A=growth_params['beta->IAA']['A'],
        lag=growth_params['beta->IAA']['lag']
    )
    
    # 13. beta->alpha (beta activating alpha)
    strains['beta->alpha'] = StrainParameters(
        strain_id='beta->alpha',
        input_molecule=BETA,
        regulation_type=ACTIVATION,
        output_molecule=ALPHA,
        k1=50.66, d1=25.86, k2=11, K=57.12, n=1.26,
        d2=110.43, k3=121.6, d3=0.062, b=0.14,
        k=growth_params['beta->alpha']['k'],
        r=growth_params['beta->alpha']['r'],
        A=growth_params['beta->alpha']['A'],
        lag=growth_params['beta->alpha']['lag']
    )

    # ALPHA->IAA (Alpha activating IAA)
    strains['alpha->IAA'] = StrainParameters(
        strain_id='alpha->IAA',
        input_molecule=ALPHA,
        regulation_type=ACTIVATION,
        output_molecule=IAA,
        k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
        d2=1.56e5, k3=566.24, d3=0.575, b=55.83,
        k=growth_params['alpha->IAA']['k'],
        r=growth_params['alpha->IAA']['r'],
        A=growth_params['alpha->IAA']['A'],
        lag=growth_params['alpha->IAA']['lag']
    )
    
    # ALPHA->ALPHA (Alpha activating ALPHA)
    strains['alpha->alpha'] = StrainParameters(
        strain_id='alpha->alpha',
        input_molecule=ALPHA,
        regulation_type=ACTIVATION,
        output_molecule=ALPHA,
        k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
        d2=1.56e5, k3=419.52, d3=2.1e4, b=2.32e4,
        k=growth_params['alpha->alpha']['k'],
        r=growth_params['alpha->alpha']['r'],
        A=growth_params['alpha->alpha']['A'],
        lag=growth_params['alpha->alpha']['lag']
    )
    
    # IAA->alpha (Auxin activating alpha)
    strains['IAA->alpha'] = StrainParameters(
        strain_id='IAA->alpha',
        input_molecule=IAA,
        regulation_type=ACTIVATION,
        output_molecule=ALPHA,
        k1=8.95e4, d1=0.082, k2=1.73e7, K=1.4e7, n=0.836,
        d2=1.88e7, k3=2.285, d3=0.28, b=0.74,
        k=growth_params['IAA->alpha']['k'],
        r=growth_params['IAA->alpha']['r'],
        A=growth_params['IAA->alpha']['A'],
        lag=growth_params['IAA->alpha']['lag']
    )
    
    # IAA->IAA (Auxin activating IAA)
    strains['IAA->IAA'] = StrainParameters(
        strain_id='IAA->IAA',
        input_molecule=IAA,
        regulation_type=ACTIVATION,
        output_molecule=IAA,
        k1=8.95e4, d1=0.082, k2=1.73e7, K=1.4e7, n=0.836,
        d2=1.88e7, k3=775.05, d3=0.84, b=780.90,
        k=growth_params['IAA->IAA']['k'],
        r=growth_params['IAA->IAA']['r'],
        A=growth_params['IAA->IAA']['A'],
        lag=growth_params['IAA->IAA']['lag']
    )

def plot_strain_growth(results, model, figsize=(12, 8), average_over_space=True, specific_locations=None):
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
