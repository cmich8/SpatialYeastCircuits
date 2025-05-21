from fixed_wellmixed_production import run_wellmixed_experiment
from simulation_v7 import *

# Run a well-mixed experiment with the Turing pattern strains
def create_turing_pattern():
    # Run experiment with simple configuration
    model, results, figures = run_wellmixed_experiment(
        strains_to_include=[
            'alpha->alpha',   # Activator self-promotion (alpha-factor promotes itself)
            'alpha->IAA',     # Activator produces inhibitor (alpha produces IAA)
            'IAA-|alpha'      # Inhibitor suppresses activator (IAA represses alpha)
        ],
        strain_concentrations=[0.2, 0.15, 0.1],  # Concentration ratios
        molecules={
            ALPHA: 5.0,      # Initial alpha-factor concentration
            IAA: 0.5         # Initial IAA concentration
        },
        grid_size=(20, 20),  # Larger grid for pattern formation
        dx=0.5,              # Smaller spatial step
        coarse_factor=1,      # No coarsening
        simulation_time=(0, 72),  # Simulation time
        n_time_points=36,    # Number of time points
        output_dir="./turing_pattern_results"
    )
    
    return model, results, figures

# Run the experiment
model, results, figures = create_turing_pattern()