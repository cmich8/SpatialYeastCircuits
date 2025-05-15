from simulation_v7 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import os

def run_wellmixed_experiment(
    strains_to_include=None,
    strain_concentrations=None,
    molecules=None,
    grid_size=(20, 20),
    dx=0.5,
    coarse_factor=5,
    simulation_time=(0,96),
    n_time_points=48,
    output_dir="./wellmixed_v2_results"
):
    """
    Run a well-mixed experiment with specified strains and molecules.
    
    Args:
        strains_to_include: List of strain IDs to include
        strain_concentrations: List of starting concentrations for each strain
        molecules: Dict mapping molecule IDs to their starting concentrations
        grid_size: Tuple of (height, width) for the 2D grid before coarsening
        dx: Grid spacing (in mm) before coarsening
        coarse_factor: Factor to coarsen the grid (reduces computational load)
        simulation_time: Tuple of (start_time, end_time) in hours
        n_time_points: Number of time points for simulation output
        output_dir: Directory to save output files
        
    Returns:
        model, results, figures
    """
    # Validate inputs
    if strains_to_include is None:
        strains_to_include = ['beta->alpha', 'alpha->venus']
    
    if strain_concentrations is None:
        strain_concentrations = [0.01] * len(strains_to_include)
    
    if molecules is None:
        molecules = {BETA: 10.0}  # Default: 10 nM beta-estradiol
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Creating model with grid size {grid_size} and coarse factor {coarse_factor}...")
    
    # Create a model with the specified coarse factor
    model = SpatialMultiStrainModel(grid_size=grid_size, dx=dx, coarse_factor=coarse_factor)
    
    # Load strain library
    strain_library = create_strain_library()
    
    # Add strains to the model
    for i, strain_id in enumerate(strains_to_include):
        if strain_id not in strain_library:
            print(f"Warning: Strain {strain_id} not found in library. Skipping.")
            continue
        
        strain_params = strain_library[strain_id]
        model.add_strain(strain_params)
        
        # Create a uniform grid with the specified concentration
        grid_height, grid_width = model.grid_size
        concentration = strain_concentrations[i]
        strain_grid = np.ones((grid_height, grid_width)) * concentration
        
        # Replace the strain grid
        model.strain_grids[i] = strain_grid
        
        print(f"Added strain {strain_id} with uniform concentration {concentration}")
    
    # Add molecules
    for molecule_id, concentration in molecules.items():
        grid_height, grid_width = model.grid_size
        molecule_grid = np.ones((grid_height, grid_width)) * concentration
        
        # Replace the molecule grid
        model.initial_molecule_grids[molecule_id] = molecule_grid
        
        print(f"Added molecule {molecule_id} with uniform concentration {concentration}")
    
    # Set simulation time
    model.set_simulation_time(simulation_time[0], simulation_time[1])
    
    # Run simulation with robust parameters
    try:
        print(f"Starting simulation for time span {simulation_time}...")
        start_time = time.time()
        
        # Extract the ODE system and initial state
        system = model._build_optimized_spatial_ode_system()
        y0 = model._get_initial_state()
        
        # Run with robust solver settings
        sol = solve_ivp(
            fun=system,
            t_span=model.time_span,
            y0=y0,
            method='LSODA',        # More robust solver for stiff equations
            rtol=1e-2,             # Relaxed relative tolerance
            atol=1e-4,             # Relaxed absolute tolerance
            max_step=2.0,          # Limit maximum step size
            t_eval=np.linspace(model.time_span[0], model.time_span[1], n_time_points)
        )
        
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        
        if not sol.success:
            print(f"Solver warning: {sol.message}")
        
        # Process the results
        results = process_simulation_results(model, sol, n_time_points)
        
        # Create visualizations
        print("Creating visualizations...")
        figures = []
        
        # Determine molecules to visualize
        molecules_to_visualize = list(molecules.keys())
        for strain in model.strains:
            if strain.output_molecule not in molecules_to_visualize:
                molecules_to_visualize.append(strain.output_molecule)
        
        # Plot spatial results
        fig1 = model.plot_spatial_results(results, time_idx=-1, molecules=molecules_to_visualize)
        figures.append(fig1)
        if output_dir:
            fig1.savefig(f'{output_dir}/spatial_results.png')
        
        # Plot growth dashboard
        fig2 = create_growth_dashboard(results, model)
        figures.append(fig2)
        if output_dir:
            fig2.savefig(f'{output_dir}/growth_dashboard.png')
        
        # Plot strain growth
        fig3 = plot_strain_growth(results, model)
        figures.append(fig3)
        if output_dir:
            fig3.savefig(f'{output_dir}/strain_growth.png')
        
        # Plot average molecule concentrations over time
        fig4 = plot_molecule_concentrations(results, molecules_to_visualize)
        figures.append(fig4)
        if output_dir:
            fig4.savefig(f'{output_dir}/molecule_concentrations.png')
        
        print("Experiment completed successfully!")
        return model, results, figures
        
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return model, None, None

def process_simulation_results(model, sol, n_time_points):
    """
    Process simulation results from solve_ivp into the expected format.
    
    Args:
        model: SpatialMultiStrainModel instance
        sol: Solution object from solve_ivp
        n_time_points: Number of time points
    
    Returns:
        Dictionary with processed results
    """
    grid_height, grid_width = model.grid_size
    
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
    for strain_idx in range(len(model.strains)):
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

# Example usage
if __name__ == "__main__":
    # Example 1: Simple two-strain system
    model1, results1, figures1 = run_wellmixed_experiment(
        strains_to_include=['beta->alpha'],
        strain_concentrations=[0.1],
        molecules={BETA: 100.0},
        grid_size=(50, 50),
        coarse_factor=2,
        output_dir="./example1_results"
    )
    
    # Example 2: Three-strain relay circuit
    #model2, results2, figures2 = run_wellmixed_experiment(
    #    strains_to_include=['beta->alpha', 'alpha->IAA', 'IAA->GFP'],
    #    strain_concentrations=[0.1, 0.1, 0.1],
    #    molecules={BETA: 100.0},
    #    grid_size=(50,50),
    #    coarse_factor=2,
    #    output_dir="./example2_results"
    #)
    
    # Example 3: Varying beta-estradiol concentrations
    #for beta_conc in [10, 100, 1000]:
    #    output_dir = f"./beta_{beta_conc}_results"
    #    model3, results3, figures3 = run_wellmixed_experiment(
    #        strains_to_include=['beta->alpha', 'alpha->venus'],
    #        strain_concentrations=[0.1, 0.1],
    #        molecules={BETA: beta_conc},
    ##        grid_size=(50, 50),
    #        coarse_factor=1,
    #        output_dir=output_dir
    #    )
    
    print("All experiments completed!")