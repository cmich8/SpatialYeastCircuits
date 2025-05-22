from simulation_v7 import *
import os
import json
import datetime
import logging

#########
# Multi-colony, multi-molecule experiment script
# Supports:
# - Arbitrary number of colonies of different strains
# - Multiple starting molecule distributions
# - Comprehensive experiment documentation
#########

class Colony:
    """Class representing a yeast colony with position, shape, and strain information."""
    
    def __init__(self, strain_id, position, shape="circle", radius=5, width=10, height=10, concentration=0.5):
        """
        Initialize a colony.
        
        Args:
            strain_id: Identifier for the strain (must exist in strain library)
            position: (row, col) position of the colony center
            shape: "circle" or "rectangle"
            radius: Radius if shape is circle
            width: Width if shape is rectangle
            height: Height if shape is rectangle
            concentration: Initial cell concentration
        """
        self.strain_id = strain_id
        self.position = position
        self.shape = shape.lower()
        self.radius = radius
        self.width = width
        self.height = height
        self.concentration = concentration
        
    def to_dict(self):
        """Convert colony to dictionary for metadata."""
        return {
            "strain_id": self.strain_id,
            "position": self.position,
            "shape": self.shape,
            "dimensions": {
                "radius": self.radius if self.shape == "circle" else None,
                "width": self.width if self.shape == "rectangle" else None,
                "height": self.height if self.shape == "rectangle" else None
            },
            "concentration": self.concentration
        }

class MoleculeDistribution:
    """Class representing a starting distribution of a signaling molecule."""
    
    def __init__(self, molecule_type, position, shape="circle", radius=5, width=10, height=10, concentration=100):
        """
        Initialize a molecule distribution.
        
        Args:
            molecule_type: Type of molecule (e.g., ALPHA, BETA, IAA)
            position: (row, col) position of the distribution center
            shape: "circle" or "rectangle"
            radius: Radius if shape is circle
            width: Width if shape is rectangle
            height: Height if shape is rectangle
            concentration: Initial molecule concentration
        """
        self.molecule_type = molecule_type
        self.position = position
        self.shape = shape.lower()
        self.radius = radius
        self.width = width
        self.height = height
        self.concentration = concentration
        
    def to_dict(self):
        """Convert molecule distribution to dictionary for metadata."""
        return {
            "molecule_type": self.molecule_type,
            "position": self.position,
            "shape": self.shape,
            "dimensions": {
                "radius": self.radius if self.shape == "circle" else None,
                "width": self.width if self.shape == "rectangle" else None,
                "height": self.height if self.shape == "rectangle" else None
            },
            "concentration": self.concentration
        }

def multi_colony_experiment(experiment_name, output_base_dir, 
                          grid_size=(70, 20), dx=1.0, coarse_factor=1,
                          colonies=None, molecule_distributions=None,
                          simulation_time=48, n_time_points=10,
                          molecules_to_visualize=None,
                          notes=""):
    """
    Run a multi-colony experiment with configurable parameters.
    
    Args:
        experiment_name: Name of the experiment for file output
        output_base_dir: Base directory for output files
        grid_size: Tuple of (height, width) for the 2D grid
        dx: Grid spacing (in mm)
        coarse_factor: Factor to coarsen the grid
        
        colonies: List of Colony objects defining strain positions and shapes
        molecule_distributions: List of MoleculeDistribution objects for initial molecules
        
        simulation_time: Total simulation time (hours)
        n_time_points: Number of time points to output
        molecules_to_visualize: List of molecule types to visualize (default: [ALPHA, BETA, VENUS])
        notes: Additional notes about the experiment
        
    Returns:
        Tuple of (simulation results, model)
    """
    # Set default values if not provided
    if colonies is None:
        colonies = [
            Colony("beta->alpha", (30, 10)),
            Colony("alpha->venus", (40, 10))
        ]
    
    if molecule_distributions is None:
        # Default to beta-estradiol at first colony position
        molecule_distributions = [
            MoleculeDistribution(BETA, colonies[0].position, "rectangle", width=10, height=10)
        ]
    
    if molecules_to_visualize is None:
        molecules_to_visualize = [ALPHA, BETA, VENUS]
    
    # Create experiment-specific output directory
    experiment_dir = os.path.join(output_base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create a log file for this experiment
    log_file = os.path.join(experiment_dir, "experiment_log.txt")
    metadata_file = os.path.join(experiment_dir, "experiment_metadata.json")
    
    # Setup logging to both console and file
    logger = setup_logger(experiment_name, log_file)
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Output directory: {experiment_dir}")
    logger.info(f"Original grid size: {grid_size}, dx: {dx}, coarse factor: {coarse_factor}")
    
    # Calculate the actual grid size after coarsening
    actual_height = grid_size[0] // coarse_factor
    actual_width = grid_size[1] // coarse_factor
    actual_grid_size = (actual_height, actual_width)
    
    logger.info(f"Actual grid size after coarsening: {actual_grid_size}")
    
    # Create strain library
    strain_library = create_strain_library()
    
    # Create model with direct actual grid size (bypass internal coarsening)
    model = SpatialMultiStrainModel(grid_size=actual_grid_size, dx=dx * coarse_factor, coarse_factor=1)
    
    # Verify the model's grid size matches what we expect
    logger.info(f"Model grid size: {model.grid_size}")
    
    # Initialize metadata
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "notes": notes,
        "grid_parameters": {
            "original_grid_size": grid_size,
            "dx": dx,
            "coarse_factor": coarse_factor,
            "actual_grid_size": actual_grid_size,
            "actual_dx": dx * coarse_factor
        },
        "colonies": [],
        "molecule_distributions": [],
        "simulation_parameters": {
            "time_span": [0, simulation_time],
            "n_time_points": n_time_points,
            "molecules_to_visualize": molecules_to_visualize
        },
        "strain_parameters": {}
    }
    
    # Add all strains used in the experiment to the model
    strain_indices = {}  # Maps strain_id to index in model
    for i, colony in enumerate(colonies):
        if colony.strain_id not in strain_indices:
            # Only add each strain type once
            if colony.strain_id not in strain_library:
                logger.error(f"Unknown strain ID: {colony.strain_id}")
                raise ValueError(f"Unknown strain ID: {colony.strain_id}")
            
            strain_indices[colony.strain_id] = len(strain_indices)
            model.add_strain(strain_library[colony.strain_id])
            
            # Add strain parameters to metadata
            metadata["strain_parameters"][colony.strain_id] = vars(strain_library[colony.strain_id])
    
    # Process each colony and add to the model
    for colony_idx, colony in enumerate(colonies):
        # Scale position and dimensions
        scaled_position = (
            min(actual_height - 1, colony.position[0] // coarse_factor),
            min(actual_width - 1, colony.position[1] // coarse_factor)
        )
        
        # Colony dimensions scaled according to coarse factor
        if colony.shape == "circle":
            scaled_radius = max(1, colony.radius // coarse_factor)
            scaled_dimensions = {"radius": scaled_radius}
            logger.info(f"Colony {colony_idx}: strain={colony.strain_id}, position={scaled_position}, shape={colony.shape}, radius={scaled_radius}")
        elif colony.shape == "rectangle":
            scaled_width = max(1, min(colony.width // coarse_factor, actual_width - 1))
            scaled_height = max(1, min(colony.height // coarse_factor, actual_height - 1))
            scaled_dimensions = {"width": scaled_width, "height": scaled_height}
            logger.info(f"Colony {colony_idx}: strain={colony.strain_id}, position={scaled_position}, shape={colony.shape}, width={scaled_width}, height={scaled_height}")
        else:
            logger.warning(f"Unsupported shape {colony.shape} for colony {colony_idx}. Using circle.")
            colony.shape = "circle"
            scaled_radius = max(1, colony.radius // coarse_factor)
            scaled_dimensions = {"radius": scaled_radius}
        
        # Add to metadata
        colony_data = colony.to_dict()
        colony_data["scaled_position"] = scaled_position
        colony_data["scaled_dimensions"] = scaled_dimensions
        colony_data["strain_index"] = strain_indices[colony.strain_id]
        metadata["colonies"].append(colony_data)
        
        # Place the colony on the model if the position is valid
        if scaled_position[0] < actual_height and scaled_position[1] < actual_width:
            strain_idx = strain_indices[colony.strain_id]
            
            if colony.shape == "circle":
                model.place_strain(strain_idx, 
                                  row=scaled_position[0], 
                                  col=scaled_position[1], 
                                  shape="circle", 
                                  radius=scaled_dimensions["radius"], 
                                  concentration=colony.concentration)
            elif colony.shape == "rectangle":
                model.place_strain(strain_idx, 
                                  row=scaled_position[0], 
                                  col=scaled_position[1], 
                                  shape="rectangle", 
                                  width=scaled_dimensions["width"], 
                                  height=scaled_dimensions["height"], 
                                  concentration=colony.concentration)
        else:
            logger.warning(f"Colony {colony_idx} position {scaled_position} is outside grid {actual_grid_size}")
    
    # Process each molecule distribution
    for mol_idx, mol_dist in enumerate(molecule_distributions):
        # Scale position and dimensions
        scaled_position = (
            min(actual_height - 1, mol_dist.position[0] // coarse_factor),
            min(actual_width - 1, mol_dist.position[1] // coarse_factor)
        )
        
        # Molecule dimensions scaled according to coarse factor
        if mol_dist.shape == "circle":
            scaled_radius = max(1, mol_dist.radius // coarse_factor)
            scaled_dimensions = {"radius": scaled_radius}
            logger.info(f"Molecule {mol_idx}: type={mol_dist.molecule_type}, position={scaled_position}, shape={mol_dist.shape}, radius={scaled_radius}")
        elif mol_dist.shape == "rectangle":
            scaled_width = max(1, min(mol_dist.width // coarse_factor, actual_width - 1))
            scaled_height = max(1, min(mol_dist.height // coarse_factor, actual_height - 1))
            scaled_dimensions = {"width": scaled_width, "height": scaled_height}
            logger.info(f"Molecule {mol_idx}: type={mol_dist.molecule_type}, position={scaled_position}, shape={mol_dist.shape}, width={scaled_width}, height={scaled_height}")
        else:
            logger.warning(f"Unsupported shape {mol_dist.shape} for molecule {mol_idx}. Using rectangle.")
            mol_dist.shape = "rectangle"
            scaled_width = max(1, min(mol_dist.width // coarse_factor, actual_width - 1))
            scaled_height = max(1, min(mol_dist.height // coarse_factor, actual_height - 1))
            scaled_dimensions = {"width": scaled_width, "height": scaled_height}
        
        # Add to metadata
        mol_data = mol_dist.to_dict()
        mol_data["scaled_position"] = scaled_position
        mol_data["scaled_dimensions"] = scaled_dimensions
        metadata["molecule_distributions"].append(mol_data)
        
        # Place the molecule on the model if the position is valid
        if scaled_position[0] < actual_height and scaled_position[1] < actual_width:
            if mol_dist.shape == "circle":
                model.place_molecule(mol_dist.molecule_type, 
                                   row=scaled_position[0], 
                                   col=scaled_position[1], 
                                   shape="circle", 
                                   radius=scaled_dimensions["radius"], 
                                   concentration=mol_dist.concentration)
            elif mol_dist.shape == "rectangle":
                model.place_molecule(mol_dist.molecule_type, 
                                   row=scaled_position[0], 
                                   col=scaled_position[1], 
                                   shape="rectangle", 
                                   width=scaled_dimensions["width"], 
                                   height=scaled_dimensions["height"], 
                                   concentration=mol_dist.concentration)
        else:
            logger.warning(f"Molecule {mol_idx} position {scaled_position} is outside grid {actual_grid_size}")
    
    # Save initial metadata to file
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Set simulation time
    model.set_simulation_time(0, simulation_time)
    
    try:
        # Run simulation
        logger.info(f"Running simulation with {n_time_points} time points...")
        results = model.simulate(n_time_points=n_time_points)
        
        # Update metadata with actual time points
        metadata["simulation_results"] = {
            "actual_time_points": len(results['t']),
            "time_values": results['t'].tolist(),
            "success": True
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Generate plots and save results
        logger.info("Generating plots...")
        
        # Plot spatial results for specified molecules
        try:
            figsr = model.plot_spatial_results(results, time_idx=-1, molecules=molecules_to_visualize)
            figsr.savefig(os.path.join(experiment_dir, "spatial_results.png"))
            plt.close(figsr)
        except Exception as e:
            logger.error(f"Error generating spatial results plot: {str(e)}")
        
        # Create animations if we have more than one time point
        if len(results['t']) > 1:
            try:
                # Create animations for each visualized molecule
                for molecule in molecules_to_visualize:
                    mol_name = molecule.lower().replace('_', '')
                    mol_anim = model.create_animation(results, molecule=molecule, interval=100, cmap='viridis')
                    mol_anim.save(os.path.join(experiment_dir, f"{mol_name}_heatmap.gif"), writer='ffmpeg', fps=5)
            except Exception as e:
                logger.error(f"Error generating animations: {str(e)}")
        else:
            logger.warning("Not enough time points for animations")
        
        # Create growth dashboard
        try:
            fig = create_growth_dashboard(results, model)
            fig.savefig(os.path.join(experiment_dir, "growth_dashboard.png"))
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error generating growth dashboard: {str(e)}")
        
        # Plot growth by location
        try:
            # Use colony positions as specific locations to track
            specific_locations = [colony_data["scaled_position"] for colony_data in metadata["colonies"]]
            fig2 = plot_strain_growth(results, model, specific_locations=specific_locations)
            fig2.savefig(os.path.join(experiment_dir, "growth_by_location.png"))
            plt.close(fig2)
        except Exception as e:
            logger.error(f"Error generating strain growth plot: {str(e)}")
            
        # Create a README.md file with experiment description
        create_experiment_readme(experiment_dir, metadata)
            
        logger.info(f"Experiment {experiment_name} completed successfully")
        return results, model
    
    except Exception as e:
        logger.error(f"Experiment {experiment_name} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Update metadata with failure info
        metadata["simulation_results"] = {
            "success": False,
            "error": str(e)
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        # Return None for both results and model to indicate failure
        return None, model

def setup_logger(name, log_file):
    """Set up a logger that writes to both file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(log_file)
    
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    # Prevent duplicate messages
    logger.propagate = False
    
    return logger

def create_experiment_readme(experiment_dir, metadata):
    """Create a README.md file with experiment description."""
    readme_path = os.path.join(experiment_dir, "README.md")
    
    with open(readme_path, 'w') as f:
        f.write(f"# Experiment: {metadata['experiment_name']}\n\n")
        
        if metadata['notes']:
            f.write(f"## Notes\n{metadata['notes']}\n\n")
        
        f.write(f"## Configuration\n\n")
        f.write(f"* **Grid size**: {metadata['grid_parameters']['original_grid_size']} (coarsened to {metadata['grid_parameters']['actual_grid_size']})\n")
        f.write(f"* **Spatial resolution (dx)**: {metadata['grid_parameters']['dx']} mm (coarsened to {metadata['grid_parameters']['actual_dx']} mm)\n")
        f.write(f"* **Simulation time**: {metadata['simulation_parameters']['time_span'][1]} hours\n")
        f.write(f"* **Number of time points**: {metadata['simulation_parameters']['n_time_points']}\n\n")
        
        f.write(f"## Colony Configuration\n\n")
        for i, colony in enumerate(metadata["colonies"]):
            f.write(f"### Colony {i+1} ({colony['strain_id']})\n")
            f.write(f"* **Position**: {colony['position']} (scaled to {colony['scaled_position']})\n")
            f.write(f"* **Shape**: {colony['shape']}\n")
            if colony['shape'] == "circle":
                f.write(f"* **Radius**: {colony['dimensions']['radius']} (scaled to {colony['scaled_dimensions']['radius']})\n")
            else:
                f.write(f"* **Width**: {colony['dimensions']['width']} (scaled to {colony['scaled_dimensions']['width']})\n")
                f.write(f"* **Height**: {colony['dimensions']['height']} (scaled to {colony['scaled_dimensions']['height']})\n")
            f.write(f"* **Concentration**: {colony['concentration']}\n\n")
        
        f.write(f"## Initial Molecule Distributions\n\n")
        for i, mol in enumerate(metadata["molecule_distributions"]):
            f.write(f"### Distribution {i+1} ({mol['molecule_type']})\n")
            f.write(f"* **Position**: {mol['position']} (scaled to {mol['scaled_position']})\n")
            f.write(f"* **Shape**: {mol['shape']}\n")
            if mol['shape'] == "circle":
                f.write(f"* **Radius**: {mol['dimensions']['radius']} (scaled to {mol['scaled_dimensions']['radius']})\n")
            else:
                f.write(f"* **Width**: {mol['dimensions']['width']} (scaled to {mol['scaled_dimensions']['width']})\n")
                f.write(f"* **Height**: {mol['dimensions']['height']} (scaled to {mol['scaled_dimensions']['height']})\n")
            f.write(f"* **Concentration**: {mol['concentration']}\n\n")
        
        f.write(f"## Results\n\n")
        if 'simulation_results' in metadata and metadata['simulation_results']['success']:
            f.write(f"* **Status**: Success\n")
            f.write(f"* **Time points**: {metadata['simulation_results']['actual_time_points']}\n")
            f.write(f"* **Timestamp**: {metadata['timestamp']}\n\n")
        else:
            f.write(f"* **Status**: Failed\n")
            if 'simulation_results' in metadata:
                f.write(f"* **Error**: {metadata['simulation_results'].get('error', 'Unknown error')}\n\n")
            
        f.write(f"## Files\n\n")
        f.write(f"* **`spatial_results.png`**: Final state of the simulation\n")
        f.write(f"* **`growth_dashboard.png`**: Overview of strain growth over time\n")
        f.write(f"* **`growth_by_location.png`**: Growth at specific locations\n")
        
        for molecule in metadata['simulation_parameters']['molecules_to_visualize']:
            mol_name = molecule.lower().replace('_', '')
            f.write(f"* **`{mol_name}_heatmap.gif`**: Animation of {molecule}\n")
            
        f.write(f"* **`experiment_log.txt`**: Detailed log of the experiment\n")
        f.write(f"* **`experiment_metadata.json`**: Complete experiment configuration in JSON format\n")

# Example usage
if __name__ == "__main__":
    # Base output directory
    output_base_dir = '/home/ec2-user/multicellularcircuits/diffusionexperiments/'
    
    # Example 3: Complex arrangement with multiple starting molecules
    experiment = 'senseandsecretestrain_IAA'
    colonies = [
        Colony("beta->IAA", (5, 5), "rectangle", width=10, height=10, concentration = 1),
        Colony("IAA->GFP", (5, 25), "rectangle", width=10, height=10, concentration = 1),
        Colony("IAA->IAA_NL", (5, 15), "rectangle", width=10, height=10, concentration = 1)
    ]
    molecule_distributions = [
        MoleculeDistribution(BETA, (5, 5), "rectangle", width=10, height=10, concentration=100),
    ]
    multi_colony_experiment(
        experiment,
        output_base_dir,
        grid_size=(10, 30),
        colonies=colonies,
        molecule_distributions=molecule_distributions,
        molecules_to_visualize=[IAA, BETA, GFP],
        simulation_time=72,
        n_time_points=15,
        notes="Complex arrangement with four colonies and three different starting molecules."
    )