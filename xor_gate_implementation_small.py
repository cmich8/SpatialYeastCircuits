import numpy as np
import matplotlib.pyplot as plt
from simulation_v5_strainspecificgrowthrates import (
    SpatialMultiStrainModel, 
    ALPHA, IAA, BETA, GFP, VENUS,
    create_strain_library
)
outputdir = '/home/ec2-user/multicellularcircuits/xor_testing'
experiment = 'attempt1_simple'
def simple_xor_gate(outputdir,experiment):
    # Get strains
    strain_library = create_strain_library()
    
    # Create smaller model with a 20x20 grid
    model = SpatialMultiStrainModel(grid_size=(20, 20), dx=1)
    
    # Set diffusion coefficients
    model.set_diffusion_coefficient(ALPHA, 0.15)
    model.set_diffusion_coefficient(IAA, 0.67)
    model.set_diffusion_coefficient(BETA, 0.2)
    
    # Add only the essential strains for XOR
    model.add_strain(strain_library['beta->alpha'])  # Input A
    model.add_strain(strain_library['beta->IAA'])    # Input B
    model.add_strain(strain_library['alpha->IAA'])   # Processing
    model.add_strain(strain_library['IAA->GFP'])     # Output reporter
    
    # Place strains in more compact layout
    model.place_strain(0, row=5, col=5, radius=2, concentration=10.0)  # Input A
    model.place_strain(1, row=5, col=15, radius=2, concentration=10.0) # Input B
    model.place_strain(2, row=10, col=10, radius=2, concentration=15.0) # Processing
    model.place_strain(3, row=15, col=10, radius=2, concentration=10.0) # Reporter
    
    # Test one scenario (1 XOR 0)
    for molecule in [ALPHA, IAA, BETA, GFP]:
        model.initial_molecule_grids[molecule] = np.zeros(model.grid_size)
    
    # Add beta-estradiol to Input A only
    model.place_molecule(BETA, row=5, col=5, radius=2, concentration=10.0)
    
    # Set shorter simulation time
    model.set_simulation_time(0, 10)
    
    # Run simulation with fewer time points
    results = model.simulate(n_time_points=10)
    
    figsr = model.plot_spatial_results(results, time_idx=-1, molecules=[ALPHA, IAA, BETA, GFP])
    figsr.savefig(f'{outputdir}/{experiment}_spatialresults.png')
    gfp_anim = model.create_animation(results, molecule=GFP, 
                                       interval=100, cmap='viridis')
    gfp_anim.save(f'{outputdir}/{experiment}_gfp_heatmap.gif', writer='ffmpeg', fps=5)
    alpha_anim = model.create_animation(results, molecule=ALPHA, 
                                       interval=100, cmap='viridis')
    alpha_anim.save('{outputdir}/{experiment}_alpha_heatmap.gif', writer='ffmpeg', fps=5)
    beta_anim = model.create_animation(results, molecule=BETA, 
                                       interval=100, cmap='viridis')
    beta_anim.save('{outputdir}/{experiment}_beta_heatmap.gif', writer='ffmpeg', fps=5)
    beta_anim = model.create_animation(results, molecule=IAA, 
                                       interval=100, cmap='viridis')
    beta_anim.save('{outputdir}/{experiment}_IAA_heatmap.gif', writer='ffmpeg', fps=5)
    
    fig = create_growth_dashboard(results, model)
    fig.savefig(f'{outputdir}/{experiment}_growthdashboard.png')
    fig2 = plot_strain_growth(results, model, specific_locations=[(5,5), (15,10),(5,15), (10,10)])  # With specific locations
    fig2.savefig(f'{outputdir}/{experiment}_growthbylocation.png')

    # Visualize just the final state
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Get the last time point
    time_idx = -1
    time_point = results['t'][time_idx]
    
    # Plot key molecules
    im0 = axes[0, 0].imshow(results['molecule_grids'][ALPHA][time_idx], cmap='Blues')
    axes[0, 0].set_title("Alpha-factor")
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(results['molecule_grids'][IAA][time_idx], cmap='Greens')
    axes[0, 1].set_title("Auxin")
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(results['molecule_grids'][BETA][time_idx], cmap='Reds')
    axes[1, 0].set_title("Beta-estradiol (Input)")
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(results['molecule_grids'][GFP][time_idx], cmap='viridis')
    axes[1, 1].set_title("GFP (Output)")
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{outputdir}/{experiment}_gridoutput.png')
    
    return model, results

def analyze_xor_gate_performance(model, results, outputdir, experiment):
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
    plt.savefig(f'{outputdir}/Poutputdirxor_gate_attempt1.png')


# Run the simplified XOR gate
model, results = simple_xor_gate(outputdir, experiment)

analyze_xor_gate_performance(model, results, outputdir, experiment)