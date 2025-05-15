import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from spatial_yeast_model_v3 import (
    SpatialMultiStrainModel, StrainParameters,
    ALPHA, IAA, BETA, GFP, VENUS, BAR1, GH3,
    ACTIVATION, REPRESSION,
    create_strain_library
)

def simulate_two_strain_relay():
    """
    Simulates a two-strain relay system with a sender and receiver.
    This is similar to Figure 3B in the paper, where a signal from a sender
    strain propagates to a receiver strain.
    """
    strain_library = create_strain_library()
    
    # Create model
    model = SpatialMultiStrainModel(grid_size=(30, 100), dx=0.05)
    
    # Set diffusion coefficients
    model.set_diffusion_coefficient(ALPHA, 0.15)  # Alpha factor diffusion (mm²/hour)
    model.set_diffusion_coefficient(BETA, 0.2)    # Beta-estradiol diffusion
    
    # Set growth parameters
    model.set_growth_parameters(growth_rate=0.15, carrying_capacity=30.0)
    
    # Add strains
    model.add_strain(strain_library['beta->alpha'])  # Sender: beta-estradiol to alpha
    model.add_strain(strain_library['alpha->venus']) # Receiver: alpha-factor to Venus
    
    # Place sender strain on the left
    model.place_strain(0, 15, 10, radius=5, concentration=20.0)
    
    # Place receiver strain on the right
    model.place_strain(1, 15, 90, radius=5, concentration=10.0)
    
    # Place initial beta-estradiol input at the left
    model.place_molecule(BETA, 15, 10, radius=7, concentration=100.0)
    
    # Set simulation time
    model.set_simulation_time(0, 10)
    
    # Run simulation
    print("Running two-strain relay simulation...")
    start_time = time.time()
    results = model.simulate(n_time_points=100)
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Create time series plots at specific locations
    positions = [(15, 10), (15, 25), (15, 50), (15, 75), (15, 90)]
    model.plot_time_series(results, positions, 
                          molecules=[ALPHA, BETA, VENUS],
                          figsize=(12, 10))
    
    # Create animation of alpha-factor diffusion
    alpha_anim = model.create_animation(results, molecule=ALPHA, 
                                       interval=100, cmap='viridis')
    gfp_anim = model.create_animation(results, molecule=VENUS, 
                                       interval=100, cmap='viridis')
    
    # Save animation (if desired)
    # alpha_anim.save('alpha_diffusion.mp4', writer='ffmpeg', fps=20)
    
    return results, {"alpha": alpha_anim, "gfp": gfp_anim}


def simulate_bistable_switch():
    """
    Simulates a spatial bistable switch based on Figure 3 from the paper.
    Creates a system with two stable states (alpha-high/IAA-low and alpha-low/IAA-high)
    using mutual repression.
    """
    strain_library = create_strain_library()
    
    # Create model
    model = SpatialMultiStrainModel(grid_size=(40, 40), dx=0.1)
    
    # Set diffusion coefficients
    model.set_diffusion_coefficient(ALPHA, 0.15)  # Alpha factor diffusion
    model.set_diffusion_coefficient(IAA, 0.67)    # Auxin diffusion
    model.set_diffusion_coefficient(BAR1, 0.05)   # BAR1 diffusion
    model.set_diffusion_coefficient(GH3, 0.05)    # GH3 diffusion
    
    # Set growth parameters
    model.set_growth_parameters(growth_rate=0.1, carrying_capacity=25.0)
    
    # Add strains for bistable switch
    model.add_strain(strain_library['alpha-|IAA'])    # 1: alpha inhibits IAA
    model.add_strain(strain_library['IAA-|alpha'])    # 2: IAA inhibits alpha
    model.add_strain(strain_library['alpha->GH3'])    # 3: alpha activates GH3 (degrades IAA)
    model.add_strain(strain_library['IAA->BAR1'])     # 4: IAA activates BAR1 (degrades alpha)
    model.add_strain(strain_library['IAA-|GFP'])      # 5: Reporter: IAA inhibits GFP
    
    # Place strains in overlapping regions
    # Core mutual inhibition (strains 1 and 2)
    model.place_strain(0, 20, 20, radius=15, concentration=8.0)  # alpha-|IAA
    model.place_strain(1, 20, 20, radius=15, concentration=6.0)  # IAA-|alpha
    
    # Signal degradation (strains 3 and 4)
    model.place_strain(2, 20, 20, radius=15, concentration=4.0)  # alpha->GH3
    model.place_strain(3, 20, 20, radius=15, concentration=3.0)  # IAA->BAR1
    
    # Reporter (strain 5)
    model.place_strain(4, 20, 20, radius=18, concentration=2.5)  # IAA-|GFP
    
    # Set initial conditions - start in alpha-high state
    model.place_molecule(ALPHA, 20, 20, radius=12, concentration=15.0)
    
    # Set simulation time for first phase (alpha-high state)
    model.set_simulation_time(0, 10)
    
    # Run first phase of simulation
    print("Running bistable switch simulation (Phase 1: alpha-high state)...")
    start_time = time.time()
    results_phase1 = model.simulate(n_time_points=50)
    end_time = time.time()
    print(f"Phase 1 completed in {end_time - start_time:.2f} seconds")
    
    # Plot results from phase 1
    model.plot_spatial_results(results_phase1, time_idx=-1, 
                              molecules=[ALPHA, IAA, GFP, BAR1, GH3])
    
    # Create a new model for phase 2
    model2 = SpatialMultiStrainModel(grid_size=(40, 40), dx=0.1)
    
    # Copy settings from phase 1
    model2.diffusion_coefficients = model.diffusion_coefficients.copy()
    model2.growth_rate = model.growth_rate
    model2.carrying_capacity = model.carrying_capacity
    
    # Add the same strains
    for i in range(5):
        model2.add_strain(model.strains[i])
        model2.strain_grids[i] = model.strain_grids[i].copy()
    
    # Set initial conditions for phase 2 - use final state from phase 1, but add IAA pulse
    for molecule in [ALPHA, BAR1, GH3, GFP]:
        model2.initial_molecule_grids[molecule] = results_phase1['molecule_grids'][molecule][-1].copy()
    
    # Add IAA pulse to switch the state
    final_iaa = results_phase1['molecule_grids'][IAA][-1].copy()
    model2.initial_molecule_grids[IAA] = final_iaa
    model2.place_molecule(IAA, 20, 20, radius=12, concentration=50.0)  # Strong IAA pulse
    
    # Set simulation time for phase 2
    model2.set_simulation_time(0, 10)
    
    # Run phase 2 of simulation
    print("Running bistable switch simulation (Phase 2: adding IAA pulse)...")
    start_time = time.time()
    results_phase2 = model2.simulate(n_time_points=50)
    end_time = time.time()
    print(f"Phase 2 completed in {end_time - start_time:.2f} seconds")
    
    # Plot results from phase 2
    model2.plot_spatial_results(results_phase2, time_idx=-1, 
                               molecules=[ALPHA, IAA, GFP, BAR1, GH3])
    
    # Plot time series at center location
    positions = [(20, 20)]
    model.plot_time_series(results_phase1, positions, 
                          molecules=[ALPHA, IAA, GFP],
                          figsize=(10, 6))
    model2.plot_time_series(results_phase2, positions, 
                           molecules=[ALPHA, IAA, GFP],
                           figsize=(10, 6))
    
    # Create animations
    alpha_anim1 = model.create_animation(results_phase1, molecule=ALPHA, 
                                        interval=100, cmap='viridis')
    iaa_anim1 = model.create_animation(results_phase1, molecule=IAA, 
                                      interval=100, cmap='plasma')
    gfp_anim1 = model.create_animation(results_phase1, molecule=GFP, 
                                      interval=100, cmap='Greens')
    
    alpha_anim2 = model2.create_animation(results_phase2, molecule=ALPHA, 
                                         interval=100, cmap='viridis')
    iaa_anim2 = model2.create_animation(results_phase2, molecule=IAA, 
                                       interval=100, cmap='plasma')
    gfp_anim2 = model2.create_animation(results_phase2, molecule=GFP, 
                                       interval=100, cmap='Greens')
    
    return {
        "phase1": {
            "results": results_phase1,
            "animations": {
                "alpha": alpha_anim1,
                "iaa": iaa_anim1,
                "gfp": gfp_anim1
            }
        },
        "phase2": {
            "results": results_phase2,
            "animations": {
                "alpha": alpha_anim2,
                "iaa": iaa_anim2,
                "gfp": gfp_anim2
            }
        }
    }


def simulate_band_pass_filter():
    """
    Simulates a spatial band-pass filter based on Figure 5E from the paper.
    Creates a system that responds only to intermediate concentrations of input.
    """
    strain_library = create_strain_library()
    
    # Create model
    model = SpatialMultiStrainModel(grid_size=(40, 40), dx=0.1)
    
    # Set diffusion coefficients
    model.set_diffusion_coefficient(ALPHA, 0.15)  # Alpha factor diffusion
    model.set_diffusion_coefficient(IAA, 0.67)    # Auxin diffusion
    
    # Set growth parameters
    model.set_growth_parameters(growth_rate=0.2, carrying_capacity=30.0)
    
    # Add strains
    model.add_strain(strain_library['alpha->IAA'])  # alpha activates IAA production
    model.add_strain(strain_library['IAA->IAA'])    # IAA positive feedback
    model.add_strain(strain_library['alpha->GFP'])  # Reporter: alpha activates GFP
    
    # Place strains uniformly
    model.place_strain(0, 20, 20, radius=15, concentration=10.0)  # alpha->IAA
    model.place_strain(1, 20, 20, radius=15, concentration=5.0)   # IAA->IAA
    model.place_strain(2, 20, 20, radius=15, concentration=8.0)   # alpha->GFP
    
    # Test different alpha concentrations
    alpha_concentrations = [0.1, 1.0, 10.0, 100.0]
    results_by_conc = {}
    
    for conc in alpha_concentrations:
        print(f"Running band-pass filter simulation with alpha = {conc}...")
        
        # Reset model with the same strains
        model_conc = SpatialMultiStrainModel(grid_size=(40, 40), dx=0.1)
        model_conc.diffusion_coefficients = model.diffusion_coefficients.copy()
        model_conc.growth_rate = model.growth_rate
        model_conc.carrying_capacity = model.carrying_capacity
        
        # Add the same strains
        for i in range(3):
            model_conc.add_strain(model.strains[i])
            model_conc.strain_grids[i] = model.strain_grids[i].copy()
        
        # Set initial alpha concentration
        model_conc.place_molecule(ALPHA, 20, 20, radius=15, concentration=conc)
        
        # Set simulation time
        model_conc.set_simulation_time(0, 12)
        
        # Run simulation
        start_time = time.time()
        results = model_conc.simulate(n_time_points=60)
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        
        # Store results
        results_by_conc[conc] = results
        
        # Plot final state
        model_conc.plot_spatial_results(results, time_idx=-1, 
                                       molecules=[ALPHA, IAA, GFP])
    
    # Plot comparative results
    center_pos = (20, 20)
    
    # Prepare data for dose-response curve
    dose = alpha_concentrations
    response = []
    
    plt.figure(figsize=(12, 6))
    
    for conc, results in results_by_conc.items():
        # Get GFP concentrations at center
        gfp_time_series = [results['molecule_grids'][GFP][t_idx][center_pos] 
                           for t_idx in range(len(results['t']))]
        
        # Plot time series
        plt.plot(results['t'], gfp_time_series, label=f'Alpha = {conc}')
        
        # Add final value to dose-response
        response.append(gfp_time_series[-1])
    
    plt.xlabel('Time (hours)')
    plt.ylabel('GFP Concentration')
    plt.title('Band-Pass Filter Response Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot dose-response curve
    plt.figure(figsize=(10, 6))
    plt.semilogx(dose, response, 'o-', linewidth=2)
    plt.xlabel('Alpha Factor Concentration (log scale)')
    plt.ylabel('GFP Response at t=12h')
    plt.title('Band-Pass Filter Dose-Response Curve')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results_by_conc


def simulate_spatial_pulse_generator():
    """
    Simulates a spatial pulse generator based on Figure 5C from the paper.
    Creates a system that generates a transient pulse of output.
    """
    strain_library = create_strain_library()
    
    # Create model
    model = SpatialMultiStrainModel(grid_size=(40, 40), dx=0.1)
    
    # Set diffusion coefficients
    model.set_diffusion_coefficient(ALPHA, 0.15)  # Alpha factor diffusion
    model.set_diffusion_coefficient(IAA, 0.67)    # Auxin diffusion
    model.set_diffusion_coefficient(BAR1, 0.05)   # BAR1 diffusion
    
    # Set growth parameters
    model.set_growth_parameters(growth_rate=0.2, carrying_capacity=20.0)
    
    # Add strains for pulse generator (incoherent feed-forward loop)
    model.add_strain(strain_library['alpha->GFP'])   # alpha activates GFP (fast response)
    model.add_strain(strain_library['alpha->IAA'])   # alpha activates IAA (slow response)
    model.add_strain(strain_library['IAA->BAR1'])    # IAA activates BAR1 (degrades alpha)
    
    # Place strains
    model.place_strain(0, 20, 20, radius=15, concentration=12.0)  # alpha->GFP
    model.place_strain(1, 20, 20, radius=15, concentration=8.0)   # alpha->IAA
    model.place_strain(2, 20, 20, radius=15, concentration=5.0)   # IAA->BAR1
    
    # Place initial alpha input
    model.place_molecule(ALPHA, 20, 20, radius=15, concentration=20.0)
    
    # Set simulation time
    model.set_simulation_time(0, 15)
    
    # Run simulation
    print("Running pulse generator simulation...")
    start_time = time.time()
    results = model.simulate(n_time_points=100)
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Plot spatial results at different time points
    time_indices = [0, 25, 50, 75, 99]
    for time_idx in time_indices:
        model.plot_spatial_results(results, time_idx=time_idx, 
                                  molecules=[ALPHA, IAA, GFP, BAR1])
    
    # Create animations
    alpha_anim = model.create_animation(results, molecule=ALPHA)
    iaa_anim = model.create_animation(results, molecule=IAA)
    gfp_anim = model.create_animation(results, molecule=GFP)
    bar1_anim = model.create_animation(results, molecule=BAR1)
    
    # Plot time series at center
    positions = [(20, 20)]
    model.plot_time_series(results, positions, 
                          molecules=[ALPHA, IAA, GFP, BAR1],
                          figsize=(10, 6))
    
    return results, {
        "alpha": alpha_anim,
        "iaa": iaa_anim, 
        "gfp": gfp_anim,
        "bar1": bar1_anim
    }


def main():
    """Run example simulations."""
    print("Spatial Yeast Multicellular Model - Example Simulations")
    print("=" * 60)
    
    # Choose which simulations to run
    run_two_strain = True
    run_three_strain = True
    run_logic_gate = False  # More complex, takes longer
    run_bistable = False    # More complex, takes longer
    run_bandpass = False    # More complex, takes longer
    run_pulse = False       # More complex, takes longer
    
    results = {}
    
    if run_two_strain:
        print("\n1. Two-Strain Relay Simulation")
        print("-" * 40)
        two_strain_results, two_strain_anim = simulate_two_strain_relay()
        results["two_strain"] = two_strain_results
        
    if run_three_strain:
        print("\n2. Three-Strain Relay Simulation")
        print("-" * 40)
        three_strain_results, three_strain_anim = simulate_three_strain_relay()
        results["three_strain"] = three_strain_results
        
    if run_logic_gate:
        print("\n3. Spatial Logic Gate Simulation")
        print("-" * 40)
        logic_gate_results, logic_gate_anims = simulate_spatial_logic_gate()
        results["logic_gate"] = logic_gate_results
        
    if run_bistable:
        print("\n4. Bistable Switch Simulation")
        print("-" * 40)
        bistable_results = simulate_bistable_switch()
        results["bistable"] = bistable_results
        
    if run_bandpass:
        print("\n5. Band-Pass Filter Simulation")
        print("-" * 40)
        bandpass_results = simulate_band_pass_filter()
        results["bandpass"] = bandpass_results
        
    if run_pulse:
        print("\n6. Pulse Generator Simulation")
        print("-" * 40)
        pulse_results, pulse_anims = simulate_spatial_pulse_generator()
        results["pulse"] = pulse_results
    
    print("\nAll simulations completed!")
    return results


def simulate_three_strain_relay():
    """
    Simulates a three-strain relay system with a sender, relay cells, and receiver.
    This demonstrates how relay cells can enhance signal propagation.
    """
    strain_library = create_strain_library()
    
    # Create model
    model = SpatialMultiStrainModel(grid_size=(30, 100), dx=0.05)
    
    # Set diffusion coefficients
    model.set_diffusion_coefficient(ALPHA, 0.15)  # Alpha factor diffusion
    model.set_diffusion_coefficient(BETA, 0.2)    # Beta-estradiol diffusion
    
    # Set growth parameters
    model.set_growth_parameters(growth_rate=0.15, carrying_capacity=30.0)
    
    # Add strains
    model.add_strain(strain_library['beta->alpha'])  # Sender: beta-estradiol to alpha
    model.add_strain(strain_library['alpha->alpha']) # Relay: alpha-factor to alpha-factor
    model.add_strain(strain_library['alpha->venus']) # Receiver: alpha-factor to Venus
    
    # Place sender strain on the left
    model.place_strain(0, 15, 10, radius=5, concentration=20.0)
    
    # Place relay strains in the middle
    for i in range(4):
        x_pos = 30 + i*10
        model.place_strain(1, 15, x_pos, radius=3, concentration=15.0)
    
    # Place receiver strain on the right
    model.place_strain(2, 15, 90, radius=5, concentration=10.0)
    
    # Place initial beta-estradiol input
    model.place_molecule(BETA, 15, 10, radius=7, concentration=100.0)
    
    # Set simulation time
    model.set_simulation_time(0, 10)
    
    # Run simulation
    print("Running three-strain relay simulation...")
    start_time = time.time()
    results = model.simulate(n_time_points=100)
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Create time series plots at specific locations
    positions = [(15, 10), (15, 30), (15, 50), (15, 70), (15, 90)]
    model.plot_time_series(results, positions, 
                          molecules=[ALPHA, BETA, VENUS],
                          figsize=(12, 10))
    
    # Create animation of alpha-factor diffusion
    alpha_anim = model.create_animation(results, molecule=ALPHA, 
                                       interval=100, cmap='viridis')
    
    return results, alpha_anim


def simulate_spatial_logic_gate():
    """
    Simulates a spatial NOR logic gate.
    This is similar to Figure 5B in the paper, where two inputs (alpha and IAA)
    converge onto a single receiver that implements NOR logic.
    """
    strain_library = create_strain_library()
    
    # Create model
    model = SpatialMultiStrainModel(grid_size=(50, 50), dx=0.1)
    
    # Set diffusion coefficients
    model.set_diffusion_coefficient(ALPHA, 0.15)  # Alpha factor diffusion
    model.set_diffusion_coefficient(IAA, 0.67)    # Auxin diffusion
    model.set_diffusion_coefficient(BETA, 0.2)    # Beta-estradiol diffusion
    
    # Set growth parameters
    model.set_growth_parameters(growth_rate=0.1, carrying_capacity=20.0)
    
    # Add strains
    model.add_strain(strain_library['beta->alpha'])  # Input 1: beta-estradiol to alpha
    model.add_strain(strain_library['beta->IAA'])    # Input 2: beta-estradiol to IAA
    model.add_strain(strain_library['IAA-|GFP'])     # Receiver: IAA represses GFP
    
    # Place input 1 strain
    model.place_strain(0, 15, 15, radius=4, concentration=10.0)
    
    # Place input 2 strain
    model.place_strain(1, 35, 15, radius=4, concentration=10.0)
    
    # Place receiver strain
    model.place_strain(2, 25, 35, radius=6, concentration=15.0)
    
    # Place beta-estradiol for input 1 (left)
    model.place_molecule(BETA, 15, 15, radius=5, concentration=100.0)
    
    # Set simulation time
    model.set_simulation_time(0, 12)
    
    # Run simulation
    print("Running spatial logic gate simulation...")
    start_time = time.time()
    results = model.simulate(n_time_points=120)
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Plot results at different time points
    time_indices = [0, 30, 60, 90, 119]
    for time_idx in time_indices:
        model.plot_spatial_results(results, time_idx=time_idx, 
                                  molecules=[ALPHA, IAA, GFP, BETA])
    
    # Create animations
    alpha_anim = model.create_animation(results, molecule=ALPHA)
    iaa_anim = model.create_animation(results, molecule=IAA)
    gfp_anim = model.create_animation(results, molecule=GFP)
    
    # Plot time series at the receiver location
    positions = [(25, 35)]
    model.plot_time_series(results, positions, 
                          molecules=[ALPHA, IAA, GFP],
                          figsize=(10, 6))
    
    return results