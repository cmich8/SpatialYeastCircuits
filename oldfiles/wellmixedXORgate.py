import numpy as np
import matplotlib.pyplot as plt
import os
from well_mixed_simulation import WellMixedSimulation, create_strain_library, ALPHA, BETA, IAA, GFP, VENUS, BAR1, GH3

"""
This script demonstrates how to use the WellMixedSimulation class to test an XOR gate.
An XOR gate should produce output only when exactly one input is present.

We'll test several different approaches to implementing an XOR gate:

1. Simple approach: Two input strains feeding into one output strain
2. Mutual inhibition approach: Each input inhibits the other's pathway 
3. Signal attenuation approach: When both inputs are present, signal is attenuated
"""

# Create output directory for plots
output_dir = "xor_gate_results"
os.makedirs(output_dir, exist_ok=True)

# Get strain library
strains = create_strain_library()

# =================================================================
# Approach 1: Simple XOR Implementation
# This approach uses beta->IAA and alpha->IAA as input strains
# and IAA->GFP as the output strain
# =================================================================
def test_simple_xor():
    print("\nTesting Simple XOR Implementation")
    
    # Create simulation
    sim = WellMixedSimulation()
    
    # Add strains with appropriate ratios
    sim.add_strain(strains['beta->IAA'], ratio=1.0)    # Senses beta, produces IAA
    sim.add_strain(strains['alpha->IAA'], ratio=1.0)   # Senses alpha, produces IAA
    sim.add_strain(strains['IAA->GFP'], ratio=1.0)     # Senses IAA, produces GFP
    
    # Set simulation parameters
    sim.set_simulation_time(0, 24)      # 24-hour simulation
    sim.set_total_density(1.0)          # Initial total cell density
    
    # Set degradation rates
    sim.set_degradation_rate(ALPHA, 0.1)   # Alpha factor degradation
    sim.set_degradation_rate(IAA, 0.2)     # Auxin degradation
    sim.set_degradation_rate(BETA, 0.05)   # Beta estradiol degradation
    
    # Test XOR gate with different input combinations
    # Testing with [0, 10] for both inputs (2x2 grid of possibilities)
    xor_results = sim.test_xor_gate(
        alpha_concs=[0, 10],
        beta_concs=[0, 10],
        time_points=100,
        plot=True,
        output_dir=os.path.join(output_dir, "simple_xor")
    )
    
    # Check if it behaves like an XOR gate
    print("Simple XOR Gate Results:")
    for reporter in xor_results['reporter_molecules']:
        if np.any([xor_results['final_values'][reporter][combo] > 0 for combo in xor_results['input_combinations']]):
            print(f"  {reporter} final values:")
            for combo in xor_results['input_combinations']:
                alpha, beta = combo
                value = xor_results['final_values'][reporter][combo]
                print(f"    Alpha={alpha}, Beta={beta}: {value:.2f}")


# =================================================================
# Approach 2: XOR with Mutual Inhibition
# This approach uses auxin and alpha factor pathways that inhibit each other
# =================================================================
def test_mutual_inhibition_xor():
    print("\nTesting XOR with Mutual Inhibition")
    
    # Create simulation
    sim = WellMixedSimulation()
    
    # Add strains with appropriate ratios
    sim.add_strain(strains['beta->IAA'], ratio=1.0)      # Senses beta, produces IAA
    sim.add_strain(strains['alpha->IAA'], ratio=1.0)     # Senses alpha, produces IAA
    sim.add_strain(strains['beta->alpha'], ratio=0.5)    # Senses beta, produces alpha
    sim.add_strain(strains['IAA->alpha'], ratio=0.5)     # Senses IAA, produces alpha
    sim.add_strain(strains['IAA->GFP'], ratio=1.0)       # Reporter strain
    
    # Set simulation parameters
    sim.set_simulation_time(0, 24)      # 24-hour simulation
    sim.set_total_density(1.0)          # Initial total cell density
    
    # Set degradation rates
    sim.set_degradation_rate(ALPHA, 0.1)   # Alpha factor degradation
    sim.set_degradation_rate(IAA, 0.15)    # Auxin degradation
    sim.set_degradation_rate(BETA, 0.05)   # Beta estradiol degradation
    
    # Test XOR gate with different input combinations
    # Testing with [0, 10] for both inputs (2x2 grid of possibilities)
    xor_results = sim.test_xor_gate(
        alpha_concs=[0, 10],
        beta_concs=[0, 10],
        time_points=100,
        plot=True,
        output_dir=os.path.join(output_dir, "mutual_inhibition_xor")
    )
    
    # Check if it behaves like an XOR gate
    print("Mutual Inhibition XOR Gate Results:")
    for reporter in xor_results['reporter_molecules']:
        if np.any([xor_results['final_values'][reporter][combo] > 0 for combo in xor_results['input_combinations']]):
            print(f"  {reporter} final values:")
            for combo in xor_results['input_combinations']:
                alpha, beta = combo
                value = xor_results['final_values'][reporter][combo]
                print(f"    Alpha={alpha}, Beta={beta}: {value:.2f}")


# =================================================================
# Approach 3: XOR with Signal Attenuation
# This approach uses strains that activate signal degradation enzymes
# =================================================================
def test_signal_attenuation_xor():
    print("\nTesting XOR with Signal Attenuation")
    
    # Create a custom strain for signal attenuation: alpha -> GH3
    alpha_to_gh3 = strains['alpha->venus']  # Clone from alpha->venus
    alpha_to_gh3.strain_id = 'alpha->GH3'
    alpha_to_gh3.output_molecule = GH3
    
    # Create a custom strain for signal attenuation: beta -> BAR1
    beta_to_bar1 = strains['beta->alpha']  # Clone from beta->alpha
    beta_to_bar1.strain_id = 'beta->BAR1'
    beta_to_bar1.output_molecule = BAR1
    
    # Create simulation
    sim = WellMixedSimulation()
    
    # Add strains with appropriate ratios
    sim.add_strain(strains['beta->IAA'], ratio=1.0)     # Senses beta, produces IAA
    sim.add_strain(strains['alpha->IAA'], ratio=1.0)    # Senses alpha, produces IAA
    sim.add_strain(alpha_to_gh3, ratio=0.5)             # Senses alpha, produces GH3 (degrades IAA)
    sim.add_strain(beta_to_bar1, ratio=0.5)             # Senses beta, produces BAR1 (degrades alpha)
    sim.add_strain(strains['IAA->GFP'], ratio=1.0)      # Senses IAA, produces GFP
    
    # Set simulation parameters
    sim.set_simulation_time(0, 24)      # 24-hour simulation
    sim.set_total_density(1.0)          # Initial total cell density
    
    # Set degradation rates
    sim.set_degradation_rate(ALPHA, 0.1)   # Alpha factor degradation
    sim.set_degradation_rate(IAA, 0.1)     # Auxin degradation
    sim.set_degradation_rate(BETA, 0.05)   # Beta estradiol degradation
    sim.set_degradation_rate(BAR1, 0.05)   # BAR1 degradation
    sim.set_degradation_rate(GH3, 0.05)    # GH3 degradation
    
    # Test XOR gate with different input combinations
    # Testing with [0, 10] for both inputs (2x2 grid of possibilities)
    xor_results = sim.test_xor_gate(
        alpha_concs=[0, 10],
        beta_concs=[0, 10],
        time_points=100,
        plot=True,
        output_dir=os.path.join(output_dir, "signal_attenuation_xor")
    )
    
    # Check if it behaves like an XOR gate
    print("Signal Attenuation XOR Gate Results:")
    for reporter in xor_results['reporter_molecules']:
        if np.any([xor_results['final_values'][reporter][combo] > 0 for combo in xor_results['input_combinations']]):
            print(f"  {reporter} final values:")
            for combo in xor_results['input_combinations']:
                alpha, beta = combo
                value = xor_results['final_values'][reporter][combo]
                print(f"    Alpha={alpha}, Beta={beta}: {value:.2f}")


# =================================================================
# Approach 4: Comprehensive XOR Test with Fine-Tuning
# This approach tests multiple strain ratios and configurations
# =================================================================
def test_comprehensive_xor():
    print("\nPerforming Comprehensive XOR Gate Test")
    
    # Create a custom strain for signal attenuation: alpha -> GH3
    alpha_to_gh3 = strains['alpha->venus']  # Clone from alpha->venus
    alpha_to_gh3.strain_id = 'alpha->GH3'
    alpha_to_gh3.output_molecule = GH3
    
    # Create a custom strain for signal attenuation: beta -> BAR1
    beta_to_bar1 = strains['beta->alpha']  # Clone from beta->alpha
    beta_to_bar1.strain_id = 'beta->BAR1'
    beta_to_bar1.output_molecule = BAR1
    
    # Test various strain ratios
    best_xor_score = -float('inf')
    best_ratios = None
    best_results = None
    
    # Ratios to test
    ratio_sets = [
        {'beta->IAA': 1.0, 'alpha->IAA': 1.0, 'alpha->GH3': 0.5, 'beta->BAR1': 0.5, 'IAA->GFP': 1.0},
        {'beta->IAA': 1.0, 'alpha->IAA': 1.0, 'alpha->GH3': 1.0, 'beta->BAR1': 1.0, 'IAA->GFP': 1.0},
        {'beta->IAA': 2.0, 'alpha->IAA': 2.0, 'alpha->GH3': 1.0, 'beta->BAR1': 1.0, 'IAA->GFP': 1.0},
        {'beta->IAA': 1.0, 'alpha->IAA': 1.0, 'alpha->GH3': 2.0, 'beta->BAR1': 2.0, 'IAA->GFP': 1.0},
        {'beta->IAA': 0.5, 'alpha->IAA': 0.5, 'alpha->GH3': 1.0, 'beta->BAR1': 1.0, 'IAA->GFP': 1.0},
    ]
    
    for i, ratios in enumerate(ratio_sets):
        print(f"Testing ratio set {i+1}/{len(ratio_sets)}: {ratios}")
        
        # Create simulation
        sim = WellMixedSimulation()
        
        # Add strains with specified ratios
        sim.add_strain(strains['beta->IAA'], ratio=ratios['beta->IAA'])
        sim.add_strain(strains['alpha->IAA'], ratio=ratios['alpha->IAA'])
        sim.add_strain(alpha_to_gh3, ratio=ratios['alpha->GH3'])
        sim.add_strain(beta_to_bar1, ratio=ratios['beta->BAR1'])
        sim.add_strain(strains['IAA->GFP'], ratio=ratios['IAA->GFP'])
        
        # Set simulation parameters
        sim.set_simulation_time(0, 24)     # 24-hour simulation
        sim.set_total_density(1.0)         # Initial total cell density
        
        # Set degradation rates
        sim.set_degradation_rate(ALPHA, 0.1)
        sim.set_degradation_rate(IAA, 0.1)
        sim.set_degradation_rate(BETA, 0.05)
        sim.set_degradation_rate(BAR1, 0.05)
        sim.set_degradation_rate(GH3, 0.05)
        
        # Test with finer grid of input concentrations
        xor_results = sim.test_xor_gate(
            alpha_concs=[0, 10],
            beta_concs=[0, 10],
            time_points=100,
            plot=False
        )
        
        # Calculate XOR score based on ideal behavior
        # Ideal XOR: output high when exactly one input is high, low otherwise
        xor_score = 0
        for reporter in xor_results['reporter_molecules']:
            if np.any([xor_results['final_values'][reporter][combo] > 0 for combo in xor_results['input_combinations']]):
                # Get values for each input combination
                values = {}
                for combo in xor_results['input_combinations']:
                    alpha, beta = combo
                    values[combo] = xor_results['final_values'][reporter][combo]
                
                # Check XOR behavior: (0,0)=low, (0,10)=high, (10,0)=high, (10,10)=low
                xor_score = (values[(0, 10)] + values[(10, 0)]) - (values[(0, 0)] + values[(10, 10)])
                print(f"  XOR score for {reporter}: {xor_score:.2f}")
                
                # Print values
                for combo in xor_results['input_combinations']:
                    alpha, beta = combo
                    value = values[combo]
                    print(f"    Alpha={alpha}, Beta={beta}: {value:.2f}")
        
        # Update best ratios if this set is better
        if xor_score > best_xor_score:
            best_xor_score = xor_score
            best_ratios = ratios
            best_results = xor_results
    
    print(f"\nBest XOR performance with ratios: {best_ratios}")
    print(f"Best XOR score: {best_xor_score:.2f}")
    
    # Create simulation with best ratios for final display
    sim = WellMixedSimulation()
    sim.add_strain(strains['beta->IAA'], ratio=best_ratios['beta->IAA'])
    sim.add_strain(strains['alpha->IAA'], ratio=best_ratios['alpha->IAA'])
    sim.add_strain(alpha_to_gh3, ratio=best_ratios['alpha->GH3'])
    sim.add_strain(beta_to_bar1, ratio=best_ratios['beta->BAR1'])
    sim.add_strain(strains['IAA->GFP'], ratio=best_ratios['IAA->GFP'])
    
    # Set simulation parameters
    sim.set_simulation_time(0, 24)
    sim.set_total_density(1.0)
    
    # Set degradation rates
    sim.set_degradation_rate(ALPHA, 0.1)
    sim.set_degradation_rate(IAA, 0.1)
    sim.set_degradation_rate(BETA, 0.05)
    sim.set_degradation_rate(BAR1, 0.05)
    sim.set_degradation_rate(GH3, 0.05)
    
    # Run final simulation with best ratios
    final_results = sim.test_xor_gate(
        alpha_concs=[0, 10],
        beta_concs=[0, 10],
        time_points=100,
        plot=True,
        output_dir=os.path.join(output_dir, "best_xor")
    )


# Run the tests
if __name__ == "__main__":
    test_simple_xor()
    test_mutual_inhibition_xor()
    test_signal_attenuation_xor()
    test_comprehensive_xor()
    
    print("\nAll XOR gate tests completed. Results saved to:", output_dir)