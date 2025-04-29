import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Callable
import time

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


class MultiStrainModel:
    """Model for simulating multiple interacting yeast strains with growth dynamics."""
    
    def __init__(self):
        """Initialize the model."""
        self.strains = []
        self.strain_concentrations = []
        self.initial_molecule_concentrations = {}
        self.time_span = (0, 10)  # Default simulation time (hours)
        
        # Growth parameters (logistic growth model)
        self.growth_rate = 0.3     # Default growth rate (per hour)
        self.carrying_capacity = 100.0  # Default carrying capacity
        
        # Initialize with default concentrations
        for molecule in [ALPHA, IAA, BETA, GFP, VENUS, BAR1, GH3]:
            self.initial_molecule_concentrations[molecule] = 0.0
    
    def add_strain(self, strain_params: StrainParameters, concentration: float = 1.0):
        """
        Add a strain to the model.
        
        Args:
            strain_params: Parameters for the strain
            concentration: Initial concentration/population of the strain (default: 1.0)
        """
        self.strains.append(strain_params)
        self.strain_concentrations.append(concentration)
        return self
    
    def set_growth_parameters(self, growth_rate: float, carrying_capacity: float):
        """
        Set the growth parameters for all strains.
        
        Args:
            growth_rate: Growth rate (per hour)
            carrying_capacity: Maximum population size
        """
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        return self
    
    def set_initial_concentration(self, molecule: str, concentration: float):
        """
        Set the initial concentration of a molecule.
        
        Args:
            molecule: Name of the molecule
            concentration: Initial concentration
        """
        if molecule not in self.initial_molecule_concentrations:
            raise ValueError(f"Unknown molecule: {molecule}")
        
        self.initial_molecule_concentrations[molecule] = concentration
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
    
    def _get_strain_state_indices(self) -> Dict[int, List[int]]:
        """
        Get the indices of state variables for each strain.
        
        Returns:
            Dictionary mapping strain index to list of state variable indices
        """
        strain_indices = {}
        
        # Each strain has 3 internal state variables + 1 for population size
        state_idx = len(self.initial_molecule_concentrations)
        
        for i in range(len(self.strains)):
            # 4 state variables per strain: population + 3 internal states
            strain_indices[i] = [state_idx, state_idx + 1, state_idx + 2, state_idx + 3]
            state_idx += 4
        
        return strain_indices
    
    def _build_ode_system(self) -> Callable:
        """
        Build the ODE system for the model.
        
        Returns:
            Function that computes the derivatives for the state variables
        """
        # Number of molecule types
        n_molecules = len(self.initial_molecule_concentrations)
        
        # Number of state variables per strain (4: population + input sensing, signal processing, output)
        n_states_per_strain = 4
        
        # Total number of state variables
        n_states = n_molecules + len(self.strains) * n_states_per_strain
        
        # Get indices for strain state variables
        strain_indices = self._get_strain_state_indices()
        
        # Convert molecule names to indices
        molecule_indices = {name: i for i, name in enumerate(self.initial_molecule_concentrations.keys())}
        
        def dydt(t, y):
            """
            Compute derivatives for all state variables.
            
            Args:
                t: Current time
                y: Current state values
                
            Returns:
                Array of derivatives
            """
            derivatives = np.zeros(n_states)
            
            # Get current molecule concentrations
            molecule_conc = y[:n_molecules]
            
            # Calculate strain dynamics and their effects on molecule concentrations
            for i, strain in enumerate(self.strains):
                # Get population size and strain states
                pop_idx = strain_indices[i][0]
                population = y[pop_idx]
                strain_states = y[strain_indices[i][1]:strain_indices[i][1] + 3]
                
                # Calculate population growth (logistic model)
                # dp/dt = r * p * (1 - p/K)
                derivatives[pop_idx] = self.growth_rate * population * (1 - population / self.carrying_capacity)
                
                # Get input molecule concentration
                input_idx = molecule_indices[strain.input_molecule]
                input_conc = molecule_conc[input_idx]
                
                # Strain dynamics - scaled by population size
                # Input sensing (x₁)
                derivatives[strain_indices[i][1]] = strain.k1 * input_conc - strain.d1 * strain_states[0]
                
                # Signal processing (x₂)
                if strain.regulation_type == ACTIVATION:
                    # Activation: x₁ activates x₂ production
                    hill_term = (strain_states[0]**strain.n) / (strain.K + strain_states[0]**strain.n)
                    derivatives[strain_indices[i][2]] = (strain.k2 * hill_term) - (strain.d2 * strain_states[1])
                else:
                    # Repression: x₁ represses x₂ production (x₂ starts high and decreases)
                    hill_term = 1 / (1 + (strain_states[0]/strain.K)**strain.n)
                    derivatives[strain_indices[i][2]] = (strain.k2 * hill_term) - (strain.d2 * strain_states[1])
                
                # Output production (x₃)
                derivatives[strain_indices[i][3]] = strain.b + strain.k3 * strain_states[1] - strain.d3 * strain_states[2]
                
                # Output affects molecule concentrations
                output_idx = molecule_indices[strain.output_molecule]
                
                if strain.output_molecule in [ALPHA, IAA, BAR1, GH3]:
                    # Secreted output affects the shared medium
                    # Rate is proportional to strain population
                    derivatives[output_idx] += population * (strain.b + strain.k3 * strain_states[1] - strain.d3 * strain_states[2])
            
            # Apply effect of BAR1 and GH3 on alpha factor and auxin
            bar1_idx = molecule_indices[BAR1]
            gh3_idx = molecule_indices[GH3]
            alpha_idx = molecule_indices[ALPHA]
            auxin_idx = molecule_indices[IAA]
            
            # BAR1 degrades alpha factor
            if molecule_conc[bar1_idx] > 0:
                bar1_effect = 0.1 * molecule_conc[bar1_idx] * molecule_conc[alpha_idx]
                derivatives[alpha_idx] -= bar1_effect
            
            # GH3 degrades auxin
            if molecule_conc[gh3_idx] > 0:
                gh3_effect = 0.1 * molecule_conc[gh3_idx] * molecule_conc[auxin_idx]
                derivatives[auxin_idx] -= gh3_effect
            
            # Basic degradation for all signaling molecules
            # This simulates natural degradation/dilution
            for molecule, idx in molecule_indices.items():
                if molecule in [ALPHA, IAA, BAR1, GH3]:
                    derivatives[idx] -= 0.05 * molecule_conc[idx]  # Basic degradation rate
            
            return derivatives
        
        return dydt
    
    def _get_initial_state(self) -> np.ndarray:
        """
        Get the initial state for the simulation.
        
        Returns:
            Array of initial state values
        """
        # Initial molecule concentrations
        initial_state = list(self.initial_molecule_concentrations.values())
        
        # Initial strain states (4 states per strain: population + 3 internal states)
        for i in range(len(self.strains)):
            # Add population size first, then initialize other states to 0
            initial_state.append(self.strain_concentrations[i])  # Initial population
            initial_state.extend([0.0, 0.0, 0.0])  # Internal states initialized to 0
        
        return np.array(initial_state)
    
    def simulate(self, n_points: int = 100) -> Dict:
        """
        Run the simulation.
        
        Args:
            n_points: Number of time points to output
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Starting simulation with {len(self.strains)} strains...")
        start_time = time.time()
        
        # Build ODE system
        system = self._build_ode_system()
        
        # Get initial state
        y0 = self._get_initial_state()
        
        # Run simulation
        sol = solve_ivp(
            fun=system,
            t_span=self.time_span,
            y0=y0,
            method='LSODA',
            t_eval=np.linspace(self.time_span[0], self.time_span[1], n_points)
        )
        
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        
        # Extract results
        molecule_names = list(self.initial_molecule_concentrations.keys())
        n_molecules = len(molecule_names)
        
        results = {
            't': sol.t,
            'molecules': {
                name: sol.y[i] for i, name in enumerate(molecule_names)
            },
            'strain_states': {},
            'populations': {}
        }
        
        # Extract strain states and populations
        strain_indices = self._get_strain_state_indices()
        for i, strain in enumerate(self.strains):
            idx = strain_indices[i]
            
            # Store population separately
            results['populations'][strain.strain_id] = sol.y[idx[0]]
            
            # Store other strain states
            results['strain_states'][strain.strain_id] = {
                'input_sensing': sol.y[idx[1]],
                'signal_processing': sol.y[idx[2]],
                'output': sol.y[idx[3]]
            }
        
        return results

    def plot_results(self, results: Dict, plot_type: str = 'molecules', figsize: Tuple[int, int] = (12, 8)):
        """
        Plot simulation results.
        
        Args:
            results: Simulation results from the simulate method
            plot_type: Type of plot ('molecules', 'strain_states', 'populations', or 'all')
            figsize: Figure size (width, height) in inches
        """
        t = results['t']
        
        if plot_type == 'molecules' or plot_type == 'all':
            # Plot molecule concentrations
            plt.figure(figsize=figsize)
            
            for name, values in results['molecules'].items():
                if name in [ALPHA, IAA, GFP, VENUS]:  # Only plot key molecules
                    plt.plot(t, values, label=name)
            
            plt.xlabel('Time (hours)')
            plt.ylabel('Concentration')
            plt.title('Molecule Concentrations Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        if plot_type == 'populations' or plot_type == 'all':
            # Plot strain populations
            plt.figure(figsize=figsize)
            
            for strain_id, population in results['populations'].items():
                plt.plot(t, population, label=f"{strain_id}")
            
            plt.xlabel('Time (hours)')
            plt.ylabel('Population Size')
            plt.title('Strain Populations Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        if plot_type == 'strain_states' or plot_type == 'all':
            # Plot strain internal states
            fig, axs = plt.subplots(len(self.strains), 3, figsize=figsize)
            
            # Handle case with only one strain
            if len(self.strains) == 1:
                axs = np.array([axs])
            
            for i, strain in enumerate(self.strains):
                strain_states = results['strain_states'][strain.strain_id]
                
                # Plot input sensing
                axs[i, 0].plot(t, strain_states['input_sensing'])
                axs[i, 0].set_title(f'{strain.strain_id} - Input Sensing')
                
                # Plot signal processing
                axs[i, 1].plot(t, strain_states['signal_processing'])
                axs[i, 1].set_title(f'{strain.strain_id} - Signal Processing')
                
                # Plot output
                axs[i, 2].plot(t, strain_states['output'])
                axs[i, 2].set_title(f'{strain.strain_id} - Output')
            
            plt.tight_layout()
            plt.show()


def create_strain_library():
    """
    Create a library of all 24 strains from the paper.
    
    Returns:
        Dictionary mapping strain IDs to StrainParameters objects
    """
    strains = {}
    
    # Parameters are based on Figure 1 in the paper
    # These are example parameters and may need adjustment to match the actual paper values
    
    # 3. IAA->GFP (Auxin activating GFP)
    strains['IAA->GFP'] = StrainParameters(
        strain_id='IAA->GFP',
        input_molecule=IAA,
        regulation_type=ACTIVATION,
        output_molecule=GFP,
        k1=8.95e4, d1=0.082, k2=1.73e7, K=1.4e7, n=0.836,
        d2=1.88e7, k3=3.15e4, d3=1.66e6, b=1.46e4
    )

    # 5. ALPHA->VENUS (Alpha activating VENUS)
    strains['alpha->venus'] = StrainParameters(
        strain_id='alpha->venus',
        input_molecule=ALPHA,
        regulation_type=ACTIVATION,
        output_molecule=VENUS,
        k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
        d2=1.56e5, k3=1.62e4, d3=2.15e6, b=5.96e3
    )

    # 12. beta->IAA (beta activating IAA)
    strains['beta->IAA'] = StrainParameters(
        strain_id='beta->IAA',
        input_molecule=BETA,
        regulation_type=ACTIVATION,
        output_molecule=IAA,
        k1=50.66, d1=25.86, k2=11, K=57.12, n=1.26,
        d2=110.43, k3=3.48e3, d3=0.16, b=0.21
    )
    
    # 13. beta->alpha (beta activating alpha)
    strains['beta->alpha'] = StrainParameters(
        strain_id='beta->alpha',
        input_molecule=BETA,
        regulation_type=ACTIVATION,
        output_molecule=ALPHA,
        k1=50.66, d1=25.86, k2=11, K=57.12, n=1.26,
        d2=110.43, k3=121.6, d3=0.062, b=0.14
    )

    # 23. IAA->IAA (Auxin activating IAA)
    strains['IAA->IAA'] = StrainParameters(
        strain_id='IAA->IAA',
        input_molecule=IAA,
        regulation_type=ACTIVATION,
        output_molecule=IAA,
        k1=8.95e4, d1=0.082, k2=1.73e7, K=1.4e7, n=0.836,
        d2=1.88e7, k3=775.05, d3=0.84, b=780.90
    )

    # 22. ALPHA->ALPHA (Alpha activating ALPHA)
    strains['alpha->alpha'] = StrainParameters(
        strain_id='alpha->alpha',
        input_molecule=ALPHA,
        regulation_type=ACTIVATION,
        output_molecule=ALPHA,
        k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
        d2=1.56e5, k3=419.52, d3=2.1e4, b=2.32e4
    ) 
    return strains


def test_bistable_switch():
    """Test the simulator with a bistable switch example (similar to Figure 3 in the paper)."""
    # Create strain library
    strain_library = create_strain_library()
    
    # Create a model
    model = MultiStrainModel()
    
    # Add strains to the model with different initial populations
    model.add_strain(strain_library['alpha->alpha'], concentration=2.0)  # α-factor positive feedback
    model.add_strain(strain_library['IAA->IAA'], concentration=5.0)      # IAA positive feedback
    model.add_strain(strain_library['IAA->GFP'], concentration=1.0)      # Reporter strain
    
    # Set growth parameters
    model.set_growth_parameters(growth_rate=0.25, carrying_capacity=50.0)
    
    # Set initial concentrations
    model.set_initial_concentration(ALPHA, 5.0)  # Initial alpha factor
    model.set_initial_concentration(IAA, 1.0)    # Initial IAA
    
    # Set simulation time
    model.set_simulation_time(0, 24)
    
    # Run simulation
    results = model.simulate(n_points=200)
    
    # Plot results
    model.plot_results(results, plot_type='all')
    
    return results


def test_signal_relay():
    """Test the simulator with a signal relay circuit example."""
    # Create strain library
    strain_library = create_strain_library()
    
    # Create a model
    model = MultiStrainModel()
    
    # Add strains to the model with different initial populations
    model.add_strain(strain_library['beta->alpha'], concentration=3.0)  # Sender strain
    model.add_strain(strain_library['alpha->alpha'], concentration=5.0) # Relay strain
    model.add_strain(strain_library['alpha->venus'], concentration=1.0) # Receiver strain
    
    # Set growth parameters
    model.set_growth_parameters(growth_rate=0.2, carrying_capacity=30.0)
    
    # Set initial concentrations
    model.set_initial_concentration(BETA, 10.0)  # Add beta-estradiol input
    
    # Set simulation time
    model.set_simulation_time(0, 15)
    
    # Run simulation
    results = model.simulate(n_points=150)
    
    # Plot results
    model.plot_results(results, plot_type='all')
    
    return results


def test_simulation():
    """Test the simulator with a simple example."""
    # Create strain library
    strain_library = create_strain_library()
    
    # Create a model
    model = MultiStrainModel()
    
    # Add strains to the model
    model.add_strain(strain_library['beta->alpha'], concentration=1.0)
    model.add_strain(strain_library['alpha->venus'], concentration=1.0)
    
    # Set initial concentrations
    model.set_initial_concentration(BETA, 10.0)  # Add beta-estradiol
    
    # Set simulation time
    model.set_simulation_time(0, 12)
    
    # Run simulation
    results = model.simulate(n_points=100)
    
    # Plot results
    model.plot_results(results, plot_type='all')