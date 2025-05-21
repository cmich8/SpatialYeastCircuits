from simulation_v7 import *

class WellMixedExperiment:
    """
    Class for running experiments with well-mixed strains distributed across the entire grid.
    """
    
    def __init__(self, grid_size=(50, 50), dx=0.1, simulation_time=(0, 48), n_time_points=100):
        """
        Initialize the experiment with grid parameters.
        
        Args:
            grid_size: Tuple of (height, width) for the 2D grid
            dx: Grid spacing (in mm)
            simulation_time: Tuple of (start_time, end_time) in hours
            n_time_points: Number of time points for simulation output
        """
        self.model = SpatialMultiStrainModel(grid_size=grid_size, dx=dx)
        self.strain_library = create_strain_library()
        self.strains_to_include = []
        self.strain_concentrations = []
        self.molecule_concentrations = {}
        self.model.set_simulation_time(simulation_time[0], simulation_time[1])
        self.n_time_points = n_time_points
        
    def add_strain(self, strain_id, concentration=0.01):
        """
        Add a strain to the experiment.
        
        Args:
            strain_id: ID of the strain from the strain library
            concentration: Starting concentration of the strain
        """
        if strain_id not in self.strain_library:
            raise ValueError(f"Strain {strain_id} not found in strain library")
            
        self.strains_to_include.append(strain_id)
        self.strain_concentrations.append(concentration)
        return self
        
    def add_molecule(self, molecule_id, concentration):
        """
        Add a signaling molecule to the experiment.
        
        Args:
            molecule_id: ID of the molecule (ALPHA, IAA, BETA, etc.)
            concentration: Starting concentration of the molecule
        """
        self.molecule_concentrations[molecule_id] = concentration
        return self
        
    def setup_model(self):
        """
        Set up the model with the specified strains and molecules.
        """
        # Add each strain to the model
        for i, strain_id in enumerate(self.strains_to_include):
            strain_params = self.strain_library[strain_id]
            self.model.add_strain(strain_params)
            
            # Create uniform grid with the specified concentration
            grid_height, grid_width = self.model.grid_size
            strain_grid = np.ones((grid_height, grid_width)) * self.strain_concentrations[i]
            
            # Replace the strain grid with our uniform concentration grid
            self.model.strain_grids[i] = strain_grid
        
        # Add molecules
        for molecule_id, concentration in self.molecule_concentrations.items():
            grid_height, grid_width = self.model.grid_size
            molecule_grid = np.ones((grid_height, grid_width)) * concentration
            
            # Replace the molecule grid with our uniform concentration grid
            self.model.initial_molecule_grids[molecule_id] = molecule_grid
            
        return self
    
    def run_simulation(self, use_optimized=True):
        """
        Run the simulation.
        
        Args:
            use_optimized: Whether to use the optimized ODE system
            
        Returns:
            Simulation results
        """
        self.setup_model()
        self.results = self.model.simulate(n_time_points=self.n_time_points, use_optimized=use_optimized)
        return self.results
    
    def visualize_results(self, output_dir=None, experiment_name="wellmixed", 
                         create_animations=True, molecules_to_visualize=None):
        """
        Visualize the simulation results.
        
        Args:
            output_dir: Directory to save visualization files
            experiment_name: Name of the experiment for file naming
            create_animations: Whether to create animations
            molecules_to_visualize: List of molecules to visualize (if None, visualize all)
            
        Returns:
            Dictionary of visualization objects
        """
        if not hasattr(self, 'results'):
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
            
        visualizations = {}
        
        # Determine which molecules to visualize
        if molecules_to_visualize is None:
            molecules_to_visualize = list(self.results['molecule_grids'].keys())
        
        # Create spatial results plot
        figsr = self.model.plot_spatial_results(self.results, time_idx=-1, molecules=molecules_to_visualize)
        visualizations['spatial_results'] = figsr
        
        if output_dir:
            figsr.savefig(f'{output_dir}/{experiment_name}_spatialresults.png')
        
        # Create growth dashboard
        fig_dashboard = create_growth_dashboard(self.results, self.model)
        visualizations['growth_dashboard'] = fig_dashboard
        
        if output_dir:
            fig_dashboard.savefig(f'{output_dir}/{experiment_name}_growthdashboard.png')
        
        # Create strain growth plot
        fig_growth = plot_strain_growth(self.results, self.model)
        visualizations['strain_growth'] = fig_growth
        
        if output_dir:
            fig_growth.savefig(f'{output_dir}/{experiment_name}_straingrowth.png')
        
        # Create animations for each molecule
        if create_animations:
            animations = {}
            for molecule in molecules_to_visualize:
                if molecule in self.results['molecule_grids']:
                    anim = self.model.create_animation(self.results, molecule=molecule, 
                                                     interval=100, cmap='viridis')
                    animations[molecule] = anim
                    
                    if output_dir:
                        anim.save(f'{output_dir}/{experiment_name}_{molecule}_heatmap.gif', writer='ffmpeg', fps=5)
            
            # Create animations for each strain
            for i, strain_id in enumerate(self.strains_to_include):
                anim = self.model.create_animation(self.results, strain_idx=i, 
                                                 interval=100, cmap='plasma')
                animations[f'strain_{strain_id}'] = anim
                
                if output_dir:
                    anim.save(f'{output_dir}/{experiment_name}_{strain_id}_population.gif', writer='ffmpeg', fps=5)
                    
            visualizations['animations'] = animations
        
        return visualizations

# Example experiments:

def run_alpha_iaa_experiment(output_dir=None):
    """
    Run an experiment with alpha and IAA signaling strains well-mixed.
    """
    # Create experiment
    experiment = WellMixedExperiment(grid_size=(50, 50), dx=0.1,cour simulation_time=(0, 48), n_time_points=10)
    
    # Add strains
    experiment.add_strain('beta->alpha', concentration=0.005)  # Sender
    experiment.add_strain('alpha->IAA', concentration=0.01)    # Signal converter
    experiment.add_strain('IAA->GFP', concentration=0.01)      # Receiver
    
    # Add beta-estradiol input
    experiment.add_molecule(BETA, 10.0)  # 10 nM
    
    # Run simulation
    results = experiment.run_simulation()
    
    # Visualize results
    visualizations = experiment.visualize_results(
        output_dir=output_dir, 
        experiment_name="alpha_iaa_wellmixed",
        molecules_to_visualize=[ALPHA, IAA, BETA, GFP]
    )
    
    return experiment, results, visualizations

def run_feedforward_loop_experiment(output_dir=None):
    """
    Run an experiment with a feedforward loop circuit.
    """
    # Create experiment
    experiment = WellMixedExperiment(grid_size=(50, 50), dx=0.1, simulation_time=(0, 72), n_time_points=100)
    
    # Add strains
    experiment.add_strain('beta->alpha', concentration=0.01)    # Initial signal generator
    experiment.add_strain('alpha->alpha', concentration=0.005)  # Positive feedback
    experiment.add_strain('alpha->venus', concentration=0.02)   # Output reporter
    
    # Add beta-estradiol input
    experiment.add_molecule(BETA, 5.0)  # 5 nM
    
    # Run simulation
    results = experiment.run_simulation()
    
    # Visualize results
    visualizations = experiment.visualize_results(
        output_dir=output_dir, 
        experiment_name="feedforward_wellmixed",
        molecules_to_visualize=[ALPHA, BETA, VENUS]
    )
    
    return experiment, results, visualizations

def run_oscillator_experiment(output_dir=None):
    """
    Attempt to create an oscillator circuit (may or may not oscillate).
    """
    # Create experiment
    experiment = WellMixedExperiment(grid_size=(50, 50), dx=0.1, simulation_time=(0, 96), n_time_points=200)
    
    # Add strains to form a potential oscillator
    experiment.add_strain('beta->alpha', concentration=0.01)    # Initial signal generator
    experiment.add_strain('alpha->IAA', concentration=0.01)     # Signal conversion (activation)
    experiment.add_strain('IAA->alpha', concentration=0.01)     # Feedback (potentially negative with delays)
    experiment.add_strain('alpha->venus', concentration=0.01)   # Output reporter for alpha
    experiment.add_strain('IAA->GFP', concentration=0.01)       # Output reporter for IAA
    
    # Add beta-estradiol input
    experiment.add_molecule(BETA, 20.0)  # 20 nM
    
    # Run simulation with more time points for smoother curves
    results = experiment.run_simulation()
    
    # Visualize results
    visualizations = experiment.visualize_results(
        output_dir=output_dir, 
        experiment_name="oscillator_wellmixed",
        molecules_to_visualize=[ALPHA, IAA, BETA, VENUS, GFP]
    )
    
    return experiment, results, visualizations

def run_concentration_gradient_experiment(output_dir=None):
    """
    Run experiments with different beta-estradiol concentrations.
    """
    results_dict = {}
    visualizations_dict = {}
    
    # Test different beta-estradiol concentrations
    beta_concentrations = [0.0, 1.0, 5.0, 10.0, 50.0]
    
    for beta_conc in beta_concentrations:
        # Create experiment
        experiment = WellMixedExperiment(grid_size=(50, 50), dx=0.1, simulation_time=(0, 48), n_time_points=50)
        
        # Add strains
        experiment.add_strain('beta->alpha', concentration=0.01)  # Sender
        experiment.add_strain('alpha->venus', concentration=0.01) # Receiver
        
        # Add beta-estradiol input
        experiment.add_molecule(BETA, beta_conc)
        
        # Run simulation
        results = experiment.run_simulation()
        results_dict[beta_conc] = results
        
        # Visualize results
        visualizations = experiment.visualize_results(
            output_dir=output_dir, 
            experiment_name=f"beta_{beta_conc}nM_wellmixed",
            molecules_to_visualize=[ALPHA, BETA, VENUS]
        )
        visualizations_dict[beta_conc] = visualizations
    
    # Create a comparative plot of Venus production for different beta concentrations
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for beta_conc, results in results_dict.items():
        # Get Venus concentration over time (spatial average)
        venus_grids = results['molecule_grids'][VENUS]
        venus_avg = [np.mean(grid) for grid in venus_grids]
        
        # Plot Venus concentration
        ax.plot(results['t'], venus_avg, label=f'Beta: {beta_conc} nM', linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Average Venus Concentration')
    ax.set_title('Venus Production vs Beta-Estradiol Concentration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_dir:
        fig.savefig(f'{output_dir}/beta_concentration_comparison.png')
    
    return results_dict, visualizations_dict, fig

# To run all experiments:
if __name__ == "__main__":
    output_dir = "./wellmixed_results"
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run experiments
    print("Running alpha-IAA experiment...")
    alpha_iaa_exp, alpha_iaa_results, alpha_iaa_viz = run_alpha_iaa_experiment(output_dir)
    
    print("Running feedforward loop experiment...")
    ffl_exp, ffl_results, ffl_viz = run_feedforward_loop_experiment(output_dir)
    
    print("Running oscillator experiment...")
    osc_exp, osc_results, osc_viz = run_oscillator_experiment(output_dir)
    
    print("Running concentration gradient experiment...")
    conc_results, conc_viz, conc_fig = run_concentration_gradient_experiment(output_dir)
    
    print("All experiments completed!")