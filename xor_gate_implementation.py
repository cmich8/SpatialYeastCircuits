import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass

# Import the functions from the spatial_yeast_model provided
from spatial_yeast_model_carryingcapactiyfixed_v4 import (
    SpatialMultiStrainModel, 
    StrainParameters, 
    ALPHA, IAA, BETA, GFP, 
    ACTIVATION, REPRESSION,
    create_strain_library
)

def implement_xor_gate():
    """
    Implements an XOR gate using yeast strains in a spatial configuration.
    An XOR gate outputs 1 when the inputs are different, and 0 when they are the same.
    """
    # Get pre-defined strain library
    strain_library = create_strain_library()
    
    # Create the model with a 60x60 grid, with 0.1mm spacing
    model = SpatialMultiStrainModel(grid_size=(60, 60), dx=3)
    
    # Set diffusion coefficients (mm²/hour)
    model.set_diffusion_coefficient(ALPHA, 0.15)  # Alpha factor
    model.set_diffusion_coefficient(IAA, 0.67)    # Auxin
    model.set_diffusion_coefficient(BETA, 0.2)    # Beta estradiol
    
    # Set simulation growth parameters
    model.set_growth_parameters(growth_rate=0.2, carrying_capacity=50.0)
    
    # Add strains for the XOR gate implementation
    
    # Input strains
    # 1. Input A: beta-estradiol -> alpha-factor (for input 0/1)
    model.add_strain(strain_library['beta->alpha'])
    
    # 2. Input B: beta-estradiol -> auxin (for input 0/1)
    model.add_strain(strain_library['beta->IAA'])
    
    # Processing strains for XOR logic
    # 3. alpha-factor -> auxin (processes input A)
    model.add_strain(strain_library['alpha->IAA'])
    
    # 4. auxin -> alpha-factor (processes input B)
    model.add_strain(strain_library['IAA->alpha'])
    
    # 5. Output reporter strain: alpha-factor -> venus (fluorescent reporter)
    model.add_strain(strain_library['alpha->venus'])
    
    # 6. Auxin reporter strain: auxin -> GFP 
    model.add_strain(strain_library['IAA->GFP'])
    
    # Place strains in specific locations to create the XOR gate layout
    
    # Input A - left side
    model.place_strain(0, row=30, col=10, radius=3, concentration=10.0)
    
    # Input B - right side
    model.place_strain(1, row=30, col=50, radius=3, concentration=10.0)
    
    # Processing A: alpha -> IAA, place in middle-left
    model.place_strain(2, row=20, col=25, radius=3, concentration=15.0)
    
    # Processing B: IAA -> alpha, place in middle-right
    model.place_strain(3, row=40, col=25, radius=3, concentration=15.0)
    
    # Alpha reporter, place at bottom center
    model.place_strain(4, row=45, col=30, radius=3, concentration=10.0)
    
    # Auxin reporter, place at top center
    model.place_strain(5, row=15, col=30, radius=3, concentration=10.0)
    
    # Define four test scenarios for XOR gate
    scenarios = [
        {"name": "0 XOR 0 = 0", "inputA": 0, "inputB": 0},
        {"name": "1 XOR 0 = 1", "inputA": 10, "inputB": 0},
        {"name": "0 XOR 1 = 1", "inputA": 0, "inputB": 10},
        {"name": "1 XOR 1 = 0", "inputA": 10, "inputB": 10}
    ]
    
    # Run simulations for each scenario
    results = []
    for scenario in scenarios:
        print(f"Running scenario: {scenario['name']}")
        
        # Reset initial molecule distributions
        for molecule in [ALPHA, IAA, BETA, GFP]:
            model.initial_molecule_grids[molecule] = np.zeros(model.grid_size)
        
        # Set up inputs based on scenario
        if scenario['inputA'] > 0:
            model.place_molecule(BETA, row=30, col=10, radius=5, concentration=scenario['inputA'])
        
        if scenario['inputB'] > 0:
            model.place_molecule(BETA, row=30, col=50, radius=5, concentration=scenario['inputB'])
        
        # Set simulation time
        model.set_simulation_time(0, 10)
        
        # Run simulation
        scenario_results = model.simulate(n_time_points=10, competition=True)
        results.append(scenario_results)
        
        # Plot results
        plot_xor_results(model, scenario_results, scenario['name'])
    
    return model, results

def plot_xor_results(model, results, scenario_name):
    """
    Plot the results of the XOR gate simulation for a specific scenario.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle(f"XOR Gate Simulation - Scenario: {scenario_name}", fontsize=16)
    
    # Get the last time point
    time_idx = -1
    time_point = results['t'][time_idx]
    
    # Plot alpha-factor distribution
    alpha_grid = results['molecule_grids'][ALPHA][time_idx]
    im0 = axes[0, 0].imshow(alpha_grid, cmap='Blues', interpolation='nearest')
    axes[0, 0].set_title(f"Alpha-factor at t={time_point:.2f}h")
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Plot auxin distribution
    iaa_grid = results['molecule_grids'][IAA][time_idx]
    im1 = axes[0, 1].imshow(iaa_grid, cmap='Greens', interpolation='nearest')
    axes[0, 1].set_title(f"Auxin at t={time_point:.2f}h")
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot beta-estradiol (input) distribution
    beta_grid = results['molecule_grids'][BETA][time_idx]
    im2 = axes[0, 2].imshow(beta_grid, cmap='Reds', interpolation='nearest')
    axes[0, 2].set_title(f"Beta-estradiol at t={time_point:.2f}h")
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Plot Input Strain A population
    strain_a_grid = results['population_grids'][0][time_idx]
    im3 = axes[1, 0].imshow(strain_a_grid, cmap='YlOrBr', interpolation='nearest')
    axes[1, 0].set_title("Input A Strain Population")
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot Input Strain B population
    strain_b_grid = results['population_grids'][1][time_idx]
    im4 = axes[1, 1].imshow(strain_b_grid, cmap='YlOrBr', interpolation='nearest')
    axes[1, 1].set_title("Input B Strain Population")
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Plot Output Reporter response
    # For XOR, we want to look at both reporters to see the output
    venus_grid = results['molecule_grids']['VENUS'][time_idx]
    gfp_grid = results['molecule_grids']['GFP'][time_idx]
    
    # For XOR, output is the max of the two reporters (since we get output when inputs are different)
    output_grid = np.maximum(venus_grid, gfp_grid)
    im5 = axes[1, 2].imshow(output_grid, cmap='viridis', interpolation='nearest')
    axes[1, 2].set_title(f"Output (VENUS + GFP) at t={time_point:.2f}h")
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def analyze_xor_gate_performance(model, results):
    """
    Analyze and plot the performance of the XOR gate over time.
    """
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Define the region of interest where we expect the XOR gate output
    # This should be the area where the reporter strains are placed
    output_region_alpha = (40, 50, 25, 35)  # (row_start, row_end, col_start, col_end)
    output_region_auxin = (10, 20, 25, 35)  # (row_start, row_end, col_start, col_end)
    
    # Extract time points
    time_points = results[0]['t']
    
    # For each scenario, extract the average fluorescence in the output region
    scenario_names = ["0 XOR 0", "1 XOR 0", "0 XOR 1", "1 XOR 1"]
    expected_outputs = [0, 1, 1, 0]  # Expected output for each scenario
    
    # Plot time series of outputs for each scenario
    for i, scenario_results in enumerate(results):
        venus_data = scenario_results['molecule_grids']['VENUS']
        gfp_data = scenario_results['molecule_grids']['GFP']
        
        # Calculate average Venus (alpha-factor reporter) in output region
        venus_avg = [np.mean(venus_data[t][output_region_alpha[0]:output_region_alpha[1], 
                                        output_region_alpha[2]:output_region_alpha[3]]) 
                    for t in range(len(time_points))]
        
        # Calculate average GFP (auxin reporter) in output region
        gfp_avg = [np.mean(gfp_data[t][output_region_auxin[0]:output_region_auxin[1], 
                                    output_region_auxin[2]:output_region_auxin[3]]) 
                  for t in range(len(time_points))]
        
        # In an XOR gate, the output is a combination of the two reporters
        # high GFP or high Venus means "1", low for both means "0"
        xor_output = [max(venus_avg[t], gfp_avg[t]) for t in range(len(time_points))]
        
        # Plot the XOR output over time
        ax1.plot(time_points, xor_output, label=f"{scenario_names[i]}")
        
        # Plot the final output value
        ax2.bar(i, xor_output[-1], color='blue' if expected_outputs[i] == 1 else 'gray', 
                alpha=0.7, label=scenario_names[i])
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('XOR Output Signal (max of Venus and GFP)')
    ax1.set_title('XOR Gate Output Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Input Scenario')
    ax2.set_ylabel('Final XOR Output Signal')
    ax2.set_title('XOR Gate Final Output for Each Scenario')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(scenario_names)
    
    # Add a threshold line for better visualization
    if len(results) > 0:
        all_outputs = [max(np.mean(results[i]['molecule_grids']['VENUS'][-1][output_region_alpha[0]:output_region_alpha[1], 
                                                              output_region_alpha[2]:output_region_alpha[3]]),
                          np.mean(results[i]['molecule_grids']['GFP'][-1][output_region_auxin[0]:output_region_auxin[1], 
                                                            output_region_auxin[2]:output_region_auxin[3]])) 
                      for i in range(len(results))]
        
        # Calculate a threshold as the average of max and min values
        if len(all_outputs) > 0:
            threshold = (max(all_outputs) + min(all_outputs)) / 2
            ax2.axhline(y=threshold, color='r', linestyle='--', label='Decision Threshold')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('xor_gate_attempt1.png')

if __name__ == "__main__":
    # Run the XOR gate implementation
    model, results = implement_xor_gate()
    
    # Analyze the performance of the XOR gate
    analyze_xor_gate_performance(model, results)