import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Callable
import time
from matplotlib import cm, colors
import copy
import os

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
    k: float = 10.0   # Carrying capacity
    r: float = 0.3    # Growth rate
    A: float = 1e-8   # Initial population fraction
    lag: float = 0.0  # Lag time (hours)
    
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


class WellMixedSimulation:
    """Model for simulating multiple interacting yeast strains in a well-mixed culture."""
    
    def __init__(self):
        """Initialize the well-mixed simulation model."""
        self.strains = []
        self.strain_ratios = []  # Relative abundance of each strain
        self.initial_molecules = {}  # Initial concentrations of signaling molecules
        self.time_span = (0, 24)  # Default simulation time (hours)
        self.total_density = 1.0  # Total initial cell density (normalized)
        
        # Uniform growth parameters for all strains
        self.growth_rate = 0.3     # Growth rate (1/hour)
        self.carrying_capacity = 10.0  # Carrying capacity
        self.lag_time = 0.0        # Lag time (hours)
        
        # Degradation rates for signaling molecules
        self.degradation_rates = {
            ALPHA: 0.1,  # Alpha factor degradation rate (1/hour)
            IAA: 0.05,   # Auxin degradation rate (1/hour)
            BETA: 0.02,  # Beta estradiol degradation rate (1/hour)
            BAR1: 0.05,  # BAR1 protein degradation rate (1/hour)
            GH3: 0.05    # GH3 protein degradation rate (1/hour)
        }
        
        # Initialize with zero initial concentrations
        for molecule in [ALPHA, IAA, BETA, GFP, VENUS, BAR1, GH3]:
            self.initial_molecules[molecule] = 0.0
    
    def add_strain(self, strain_params: StrainParameters, ratio: float = 1.0):
        """
        Add a strain to the simulation with its relative abundance.
        
        Args:
            strain_params: Parameters for the strain
            ratio: Relative abundance of this strain (will be normalized)
        """
        self.strains.append(copy.deepcopy(strain_params))
        self.strain_ratios.append(ratio)
        
        return self
    
    def set_molecule_concentration(self, molecule: str, concentration: float):
        """
        Set the initial concentration of a signaling molecule.
        
        Args:
            molecule: Name of the molecule
            concentration: Initial concentration
        """
        if molecule not in self.initial_molecules:
            raise ValueError(f"Unknown molecule: {molecule}")
        
        self.initial_molecules[molecule] = concentration
        return self
    
    def set_degradation_rate(self, molecule: str, rate: float):
        """
        Set the degradation rate for a molecule.
        
        Args:
            molecule: Name of the molecule
            rate: Degradation rate (1/hour)
        """
        if molecule not in self.degradation_rates:
            raise ValueError(f"Unknown molecule for degradation: {molecule}")
        
        self.degradation_rates[molecule] = rate
        return self
    def set_growth_parameters(self, growth_rate: float, carrying_capacity: float, lag_time: float = 0.0):
        """
        Set uniform growth parameters for all strains.
        
        Args:
            growth_rate: Growth rate (1/hour)
            carrying_capacity: Carrying capacity 
            lag_time: Lag time (hours)
        """
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.lag_time = lag_time
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
    
    def set_total_density(self, density: float):
        """
        Set the total initial cell density.
        
        Args:
            density: Total initial cell density (normalized)
        """
        self.total_density = density
        return self
        
    def set_growth_parameters(self, growth_rate: float, carrying_capacity: float, lag_time: float = 0.0):
        """
        Set uniform growth parameters for all strains.
        
        Args:
            growth_rate: Growth rate (1/hour)
            carrying_capacity: Carrying capacity 
            lag_time: Lag time (hours)
        """
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.lag_time = lag_time
        return self
    
    def _build_ode_system(self) -> Callable:
        """
        Build the ODE system for the well-mixed model.
        
        Returns:
            Function that computes the derivatives for the state variables
        """
        n_strains = len(self.strains)
        
        # Normalize strain ratios
        total_ratio = sum(self.strain_ratios)
        normalized_ratios = [ratio / total_ratio for ratio in self.strain_ratios]
        
        # Initialize strain populations
        initial_populations = [self.total_density * ratio for ratio in normalized_ratios]
        
        # Track molecules and their indices
        molecules = [ALPHA, IAA, BETA, GFP, VENUS, BAR1, GH3]
        molecule_indices = {molecule: i for i, molecule in enumerate(molecules)}
        n_molecules = len(molecules)
        
        # States for each strain: population and 3 internal states
        # (input sensing, signal processing, output)
        n_strain_states = 4  # population + 3 internal states
        
        # Calculate total number of states
        n_states = n_molecules + n_strains * n_strain_states
        
        def dydt(t, y):
            """
            Compute derivatives for all state variables.
            
            Args:
                t: Current time
                y: Current state values
                
            Returns:
                Array of derivatives
            """
            # Initialize derivatives array
            derivatives = np.zeros_like(y)
            
            # Extract molecule concentrations
            molecule_concs = y[:n_molecules]
            
            # Extract strain populations and internal states
            strain_states = []
            for i in range(n_strains):
                start_idx = n_molecules + i * n_strain_states
                # Get population and 3 internal states for this strain
                strain_states.append(y[start_idx:start_idx + n_strain_states])
            
            # Calculate total population
            total_population = sum(states[0] for states in strain_states)
            
            # Calculate strain dynamics and their effects on molecule concentrations
            for strain_idx, strain_params in enumerate(self.strains):
                # Get strain states (population and internal states)
                population = strain_states[strain_idx][0]
                input_sensing = strain_states[strain_idx][1]
                signal_processing = strain_states[strain_idx][2]
                output_production = strain_states[strain_idx][3]
                
                # Get input molecule concentration
                input_molecule = strain_params.input_molecule
                input_conc = molecule_concs[molecule_indices[input_molecule]]
                
                # Calculate growth using logistic model with competition
                # Use uniform growth parameters for all strains
                if t < self.lag_time:
                    # During lag phase, no growth
                    population_deriv = 0
                else:
                    # After lag phase, logistic growth with competition
                    population_deriv = self.growth_rate * population * (1 - total_population / self.carrying_capacity)
                
                # Calculate internal state derivatives
                # Input sensing
                input_sensing_deriv = strain_params.k1 * input_conc - strain_params.d1 * input_sensing
                
                # Signal processing
                if strain_params.regulation_type == ACTIVATION:
                    # Activation: input activates signal processing
                    hill_term = (input_sensing ** strain_params.n) / \
                                (strain_params.K ** strain_params.n + input_sensing ** strain_params.n)
                    signal_processing_deriv = (strain_params.k2 * hill_term) - \
                                              (strain_params.d2 * signal_processing)
                else:
                    # Repression: input represses signal processing
                    hill_term = 1 / (1 + (input_sensing / strain_params.K) ** strain_params.n)
                    signal_processing_deriv = (strain_params.k2 * hill_term) - \
                                              (strain_params.d2 * signal_processing)
                
                # Output production
                output_deriv = strain_params.b + strain_params.k3 * signal_processing - \
                               strain_params.d3 * output_production
                
                # Update strain derivatives
                start_idx = n_molecules + strain_idx * n_strain_states
                derivatives[start_idx] = population_deriv
                derivatives[start_idx + 1] = input_sensing_deriv
                derivatives[start_idx + 2] = signal_processing_deriv
                derivatives[start_idx + 3] = output_deriv
                
                # Update molecule derivatives based on strain output
                output_molecule = strain_params.output_molecule
                output_rate = population * (strain_params.b + strain_params.k3 * signal_processing)
                
                # Add strain's contribution to molecule concentration
                if output_molecule in molecule_indices:
                    derivatives[molecule_indices[output_molecule]] += output_rate
            
            # Apply degradation for signaling molecules
            for molecule, idx in molecule_indices.items():
                if molecule in self.degradation_rates:
                    derivatives[idx] -= self.degradation_rates[molecule] * molecule_concs[idx]
            
            # Apply effect of BAR1 on alpha factor
            if BAR1 in molecule_indices and ALPHA in molecule_indices:
                bar1_idx = molecule_indices[BAR1]
                alpha_idx = molecule_indices[ALPHA]
                bar1_conc = molecule_concs[bar1_idx]
                alpha_conc = molecule_concs[alpha_idx]
                
                # BAR1 degrades alpha factor
                bar1_effect = 0.5 * bar1_conc * alpha_conc
                derivatives[alpha_idx] -= bar1_effect
            
            # Apply effect of GH3 on auxin
            if GH3 in molecule_indices and IAA in molecule_indices:
                gh3_idx = molecule_indices[GH3]
                iaa_idx = molecule_indices[IAA]
                gh3_conc = molecule_concs[gh3_idx]
                iaa_conc = molecule_concs[iaa_idx]
                
                # GH3 degrades auxin
                gh3_effect = 0.5 * gh3_conc * iaa_conc
                derivatives[iaa_idx] -= gh3_effect
            
            return derivatives
            
        return dydt
    
    def _get_initial_state(self) -> np.ndarray:
        """
        Get the initial state for the simulation.
        
        Returns:
            Array of initial state values
        """
        n_strains = len(self.strains)
        
        # Normalize strain ratios
        total_ratio = sum(self.strain_ratios)
        normalized_ratios = [ratio / total_ratio for ratio in self.strain_ratios]
        
        # Initialize strain populations
        initial_populations = [self.total_density * ratio for ratio in normalized_ratios]
        
        # Track molecules and their indices
        molecules = [ALPHA, IAA, BETA, GFP, VENUS, BAR1, GH3]
        
        # Start with molecule concentrations
        initial_state = []
        for molecule in molecules:
            initial_state.append(self.initial_molecules[molecule])
        
        # Add strain population and internal states (initialized to 0)
        for i, strain in enumerate(self.strains):
            # Add strain population
            initial_state.append(initial_populations[i])
            
            # Add 3 internal states initialized to 0
            for _ in range(3):
                initial_state.append(0.0)
        
        return np.array(initial_state)
    
    def simulate(self, n_time_points: int = 100) -> Dict:
        """
        Run the well-mixed simulation.
        
        Args:
            n_time_points: Number of time points to output
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Starting well-mixed simulation with {len(self.strains)} strains...")
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
            method='RK45',             # Using RK45 (Dormand-Prince method)
            rtol=1e-3,                 # Relative tolerance (increase for speed)
            atol=1e-6,                 # Absolute tolerance 
            t_eval=np.linspace(self.time_span[0], self.time_span[1], n_time_points)
        )
        
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        
        # Define molecules and their indices
        molecules = [ALPHA, IAA, BETA, GFP, VENUS, BAR1, GH3]
        molecule_indices = {molecule: i for i, molecule in enumerate(molecules)}
        n_molecules = len(molecules)
        
        # Extract results
        results = {
            't': sol.t,
            'molecules': {},
            'populations': [],
            'internal_states': []
        }
        
        # Extract molecule concentrations
        for molecule in molecules:
            idx = molecule_indices[molecule]
            results['molecules'][molecule] = sol.y[idx]
        
        # Extract strain populations and internal states
        n_strain_states = 4  # population + 3 internal states
        for i, strain in enumerate(self.strains):
            # Extract population
            start_idx = n_molecules + i * n_strain_states
            results['populations'].append(sol.y[start_idx])
            
            # Extract internal states
            strain_internal = []
            for j in range(1, 4):  # 3 internal states
                strain_internal.append(sol.y[start_idx + j])
            
            results['internal_states'].append(strain_internal)
        
        return results
    
    def plot_results(self, results: Dict, figsize: Tuple[int, int] = (15, 10), 
                   output_dir: str = None, filename_prefix: str = "simulation"):
        """
        Plot simulation results.
        
        Args:
            results: Simulation results from the simulate method
            figsize: Figure size (width, height) in inches
            output_dir: Directory to save plots (if None, plots are displayed)
            filename_prefix: Prefix for saved plot filenames
        """
        t = results['t']
        molecules = [ALPHA, IAA, BETA, GFP, VENUS, BAR1, GH3]
        
        # Create output directory if it doesn't exist
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot 1: Molecule concentrations
        fig1, ax1 = plt.subplots(figsize=figsize)
        for molecule in molecules:
            if np.any(results['molecules'][molecule] > 0):
                ax1.plot(t, results['molecules'][molecule], label=molecule, linewidth=2)
        
        ax1.set_xlabel('Time (hours)', fontsize=12)
        ax1.set_ylabel('Concentration', fontsize=12)
        ax1.set_title('Signaling Molecule Concentrations', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_molecules.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        # Plot 2: Strain populations
        fig2, ax2 = plt.subplots(figsize=figsize)
        for i, strain in enumerate(self.strains):
            ax2.plot(t, results['populations'][i], label=strain.strain_id, linewidth=2)
        
        ax2.set_xlabel('Time (hours)', fontsize=12)
        ax2.set_ylabel('Population', fontsize=12)
        ax2.set_title('Strain Populations', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_populations.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        # Plot 3: Reporter molecules (GFP, VENUS)
        fig3, ax3 = plt.subplots(figsize=figsize)
        for molecule in [GFP, VENUS]:
            if np.any(results['molecules'][molecule] > 0):
                ax3.plot(t, results['molecules'][molecule], label=molecule, linewidth=2)
        
        ax3.set_xlabel('Time (hours)', fontsize=12)
        ax3.set_ylabel('Concentration', fontsize=12)
        ax3.set_title('Reporter Molecule Concentrations', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_reporters.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        return [fig1, fig2, fig3]
    
    def test_xor_gate(self, alpha_concs=[0, 10], beta_concs=[0, 10], 
                     time_points=100, plot=True, output_dir=None):
        """
        Test an XOR gate by running simulations with all combinations of inputs.
        
        Args:
            alpha_concs: List of alpha factor concentrations to test
            beta_concs: List of beta estradiol concentrations to test
            time_points: Number of time points for simulation
            plot: Whether to plot results
            output_dir: Directory to save plots (if None, plots are displayed)
            
        Returns:
            Dictionary with XOR gate test results
        """
        # Store results for each input combination
        xor_results = {
            'input_combinations': [],
            'reporter_levels': {},
            'reporter_molecules': [GFP, VENUS],
            'final_values': {},
            'simulations': {}
        }
        
        # Run simulations for all input combinations
        for alpha_conc in alpha_concs:
            for beta_conc in beta_concs:
                # Set input concentrations
                self.set_molecule_concentration(ALPHA, alpha_conc)
                self.set_molecule_concentration(BETA, beta_conc)
                
                # Run simulation
                sim_results = self.simulate(time_points)
                
                # Store input combination
                input_combo = (alpha_conc, beta_conc)
                xor_results['input_combinations'].append(input_combo)
                
                # Store simulation results
                xor_results['simulations'][input_combo] = sim_results
                
                # Store reporter levels
                for reporter in xor_results['reporter_molecules']:
                    if reporter not in xor_results['reporter_levels']:
                        xor_results['reporter_levels'][reporter] = {}
                    
                    xor_results['reporter_levels'][reporter][input_combo] = sim_results['molecules'][reporter]
                    
                    # Store final value
                    if reporter not in xor_results['final_values']:
                        xor_results['final_values'][reporter] = {}
                    
                    xor_results['final_values'][reporter][input_combo] = sim_results['molecules'][reporter][-1]
                
                # Generate plot for this combination
                if plot:
                    filename_prefix = f"xor_alpha{alpha_conc}_beta{beta_conc}"
                    self.plot_results(sim_results, output_dir=output_dir, filename_prefix=filename_prefix)
        
        # Plot summary of XOR gate behavior
        if plot:
            self._plot_xor_summary(xor_results, output_dir)
        
        return xor_results
    
    def _plot_xor_summary(self, xor_results, output_dir=None):
        """Plot summary of XOR gate behavior."""
        # Create figure for summary
        fig, axes = plt.subplots(1, len(xor_results['reporter_molecules']), figsize=(15, 6))
        if len(xor_results['reporter_molecules']) == 1:
            axes = [axes]
        
        # For each reporter molecule
        for i, reporter in enumerate(xor_results['reporter_molecules']):
            ax = axes[i]
            
            # Get unique input values
            alpha_values = sorted(set(combo[0] for combo in xor_results['input_combinations']))
            beta_values = sorted(set(combo[1] for combo in xor_results['input_combinations']))
            
            # Create grid for heatmap
            grid = np.zeros((len(alpha_values), len(beta_values)))
            
            # Fill grid with final reporter values
            for a_idx, alpha in enumerate(alpha_values):
                for b_idx, beta in enumerate(beta_values):
                    combo = (alpha, beta)
                    if combo in xor_results['final_values'][reporter]:
                        grid[a_idx, b_idx] = xor_results['final_values'][reporter][combo]
            
            # Create heatmap
            im = ax.imshow(grid, cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax, label=f'{reporter} Concentration')
            
            # Set labels
            ax.set_xticks(np.arange(len(beta_values)))
            ax.set_yticks(np.arange(len(alpha_values)))
            ax.set_xticklabels(beta_values)
            ax.set_yticklabels(alpha_values)
            ax.set_xlabel('Beta-estradiol Concentration')
            ax.set_ylabel('Alpha Factor Concentration')
            ax.set_title(f'XOR Gate Response ({reporter})')
            
            # Add text annotations with values
            for a_idx, alpha in enumerate(alpha_values):
                for b_idx, beta in enumerate(beta_values):
                    text = ax.text(b_idx, a_idx, f'{grid[a_idx, b_idx]:.2f}',
                                ha="center", va="center", color="w" if grid[a_idx, b_idx] > grid.max()/2 else "k")
        
        plt.tight_layout()
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, "xor_gate_summary.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        # Create line plots for each input combination
        fig, axes = plt.subplots(len(xor_results['reporter_molecules']), 1, figsize=(10, 10))
        if len(xor_results['reporter_molecules']) == 1:
            axes = [axes]
        
        # For each reporter molecule
        for i, reporter in enumerate(xor_results['reporter_molecules']):
            ax = axes[i]
            
            # Plot reporter level for each input combination
            for combo in xor_results['input_combinations']:
                alpha_conc, beta_conc = combo
                ax.plot(xor_results['simulations'][combo]['t'], 
                        xor_results['reporter_levels'][reporter][combo],
                        label=f'Alpha={alpha_conc}, Beta={beta_conc}')
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel(f'{reporter} Concentration')
            ax.set_title(f'XOR Gate Dynamics ({reporter})')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, "xor_gate_dynamics.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()


def create_strain_library():
    """
    Create a library of all strains from the paper with growth parameters.
    
    Returns:
        Dictionary mapping strain IDs to StrainParameters objects
    """
    strains = {}
    
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

    # ALPHA->IAA (Alpha activating IAA)
    strains['alpha->IAA'] = StrainParameters(
        strain_id='alpha->IAA',
        input_molecule=ALPHA,
        regulation_type=ACTIVATION,
        output_molecule=IAA,
        k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
        d2=1.56e5, k3=566.24, d3=0.575, b=55.83
    )
    
    # ALPHA->ALPHA (Alpha activating ALPHA)
    strains['alpha->alpha'] = StrainParameters(
        strain_id='alpha->alpha',
        input_molecule=ALPHA,
        regulation_type=ACTIVATION,
        output_molecule=ALPHA,
        k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
        d2=1.56e5, k3=419.52, d3=2.1e4, b=2.32e4
    )
    
    # IAA->alpha (Auxin activating alpha)
    strains['IAA->alpha'] = StrainParameters(
        strain_id='IAA->alpha',
        input_molecule=IAA,
        regulation_type=ACTIVATION,
        output_molecule=ALPHA,
        k1=8.95e4, d1=0.082, k2=1.73e7, K=1.4e7, n=0.836,
        d2=1.88e7, k3=2.285, d3=0.28, b=0.74
    )
    
    # IAA->IAA (Auxin activating IAA)
    strains['IAA->IAA'] = StrainParameters(
        strain_id='IAA->IAA',
        input_molecule=IAA,
        regulation_type=ACTIVATION,
        output_molecule=IAA,
        k1=8.95e4, d1=0.082, k2=1.73e7, K=1.4e7, n=0.836,
        d2=1.88e7, k3=775.05, d3=0.84, b=780.90
    )
    
    return strains