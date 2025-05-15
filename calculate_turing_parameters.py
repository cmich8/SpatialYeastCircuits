import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from simulation_v7 import SpatialMultiStrainModel, create_strain_library
from wellmixed_spatial import WellMixedExperiment

def extract_reaction_parameters_from_model(strain_library, activator='IAA', inhibitor='ALPHA'):
    """
    Extract reaction parameters for Turing analysis from strain models.
    
    Args:
        strain_library: Dictionary of strain parameters
        activator: Activator molecule (default: IAA)
        inhibitor: Inhibitor molecule (default: ALPHA)
        
    Returns:
        Dictionary of reaction parameters
    """
    # Get relevant strains
    activator_producing_strains = [s for s_id, s in strain_library.items() 
                                  if s.output_molecule == activator]
    inhibitor_producing_strains = [s for s_id, s in strain_library.items() 
                                  if s.output_molecule == inhibitor]
    
    # Check if we have the necessary strains
    if not activator_producing_strains or not inhibitor_producing_strains:
        print(f"Warning: Missing strains for {activator} or {inhibitor} production")
        return None
    
    # Map our strains to the activator-inhibitor model:
    # du/dt = a - bu + u²v  (activator equation)
    # dv/dt = c - du - u²v  (inhibitor equation)
    
    # Find strains that match key interactions
    activator_self_promotion = None  # u²v term in activator equation
    inhibitor_production = None      # c term in inhibitor equation
    activator_degradation = None     # -bu term in activator equation
    inhibitor_regulation = None      # -du and -u²v terms in inhibitor equation
    
    # Look for strains that implement these interactions
    for strain in activator_producing_strains:
        if strain.input_molecule == activator:
            # This strain enhances activator production based on activator (potential u² term)
            activator_self_promotion = strain
        
    for strain in inhibitor_producing_strains:
        if strain.input_molecule == activator:
            # This strain produces inhibitor in response to activator
            inhibitor_regulation = strain
    
    # Extract parameters
    # Note: These are approximations based on mapping strain behavior to Turing model
    
    # Parameter a: base production rate of activator
    # Use average of basal production rates from activator-producing strains
    a = np.mean([strain.b for strain in activator_producing_strains])
    
    # Parameter b: degradation rate of activator
    # Use average degradation rate from activator-producing strains
    b = np.mean([strain.d3 for strain in activator_producing_strains])
    
    # Parameter c: base production rate of inhibitor
    # Use average of basal production rates from inhibitor-producing strains
    c = np.mean([strain.b for strain in inhibitor_producing_strains])
    
    # Parameter d: degradation rate of inhibitor 
    # Use average degradation rate from inhibitor-producing strains
    d = np.mean([strain.d3 for strain in inhibitor_producing_strains])
    
    # For completeness, print what strains we used
    print(f"Extracted parameters from:")
    print(f"  Activator producers: {[s.strain_id for s in activator_producing_strains]}")
    print(f"  Inhibitor producers: {[s.strain_id for s in inhibitor_producing_strains]}")
    
    # Create reaction parameters dictionary
    params = {
        'a': float(a),
        'b': float(b),
        'c': float(c), 
        'd': float(d),
        'activator': activator,
        'inhibitor': inhibitor
    }
    
    print(f"Extracted parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
    
    return params

def optimize_parameters_from_simulation(experiment, results, activator='IAA', inhibitor='ALPHA',
                                      init_params=None, time_idx=-1):
    """
    Optimize reaction parameters by fitting to simulation results.
    
    Args:
        experiment: WellMixedExperiment instance
        results: Simulation results
        activator: Activator molecule name
        inhibitor: Inhibitor molecule name
        init_params: Initial parameter guesses (optional)
        time_idx: Time index to use for fitting
        
    Returns:
        Optimized parameter dictionary
    """
    # Check if activator and inhibitor are present in results
    if activator not in results['molecule_grids'] or inhibitor not in results['molecule_grids']:
        print(f"Error: {activator} or {inhibitor} not found in simulation results")
        return None
    
    # Get spatial grid at specified time
    activator_grid = results['molecule_grids'][activator][time_idx]
    inhibitor_grid = results['molecule_grids'][inhibitor][time_idx]
    
    # Flatten grids for easier fitting
    u_data = activator_grid.flatten()
    v_data = inhibitor_grid.flatten()
    
    # Use steady state conditions to estimate parameters
    # At steady state, these equations should be close to zero:
    # du/dt = a - bu + u²v = 0
    # dv/dt = c - du - u²v = 0
    
    if init_params is None:
        # Start with some reasonable values
        init_params = {
            'a': 0.1,
            'b': 0.2,
            'c': 0.1,
            'd': 0.2
        }
    
    # Define objective function for optimization
    def objective(params_array):
        # Unpack parameters
        a, b, c, d = params_array
        
        # Calculate steady state residuals
        du_dt = a - b * u_data + u_data**2 * v_data
        dv_dt = c - d * u_data - u_data**2 * v_data
        
        # Return sum of squared residuals
        return np.sum(du_dt**2) + np.sum(dv_dt**2)
    
    # Initial parameter values
    initial_values = [
        init_params['a'],
        init_params['b'],
        init_params['c'],
        init_params['d']
    ]
    
    # Bounds for parameters (all positive)
    bounds = [(0.001, 10)] * 4
    
    # Run optimization
    result = minimize(
        objective, 
        initial_values,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    # Check if optimization succeeded
    if result.success:
        opt_a, opt_b, opt_c, opt_d = result.x
        
        # Create parameter dictionary
        params = {
            'a': float(opt_a),
            'b': float(opt_b),
            'c': float(opt_c),
            'd': float(opt_d),
            'activator': activator,
            'inhibitor': inhibitor
        }
        
        print(f"Optimized parameters: a={opt_a:.4f}, b={opt_b:.4f}, c={opt_c:.4f}, d={opt_d:.4f}")
        
        # Calculate model fitness
        final_residual = result.fun
        print(f"Fit quality: residual={final_residual:.6e}")
        
        return params
    else:
        print(f"Optimization failed: {result.message}")
        return None

def verify_turing_conditions(params, D1, D2):
    """
    Verify if parameters satisfy Turing instability conditions.
    
    Args:
        params: Dictionary of reaction parameters
        D1: Diffusion coefficient of activator
        D2: Diffusion coefficient of inhibitor
        
    Returns:
        Dictionary with Turing conditions check results
    """
    # Extract parameters
    a = params['a'] 
    b = params['b']
    c = params['c']
    d = params['d']
    
    # Create steady state calculation function
    def steady_state_eqs(x):
        u, v = x
        eq1 = a - b*u + u**2*v
        eq2 = c - d*u - u**2*v
        return [eq1, eq2]
    
    # Numerically find steady state
    from scipy.optimize import root
    sol = root(steady_state_eqs, [1.0, 1.0])
    
    if not sol.success:
        print("Failed to find steady state")
        return {'is_turing_possible': False}
    
    # Get steady state
    u0, v0 = sol.x
    
    # Check steady state stability
    # Compute Jacobian matrix at steady state
    du_du = -b + 2*u0*v0
    du_dv = u0**2
    dv_du = -d - 2*u0*v0
    dv_dv = -u0**2
    
    J = np.array([[du_du, du_dv], 
                   [dv_du, dv_dv]])
    
    # Get eigenvalues
    eigenvalues = np.linalg.eigvals(J)
    
    # Check if stable without diffusion (both eigenvalues must have negative real parts)
    is_stable = all(eigenvalue.real < 0 for eigenvalue in eigenvalues)
    
    # Calculate trace and determinant
    trace = np.trace(J)
    det = np.linalg.det(J)
    
    # Turing instability requires:
    # 1. Stable system without diffusion
    # 2. trace(J) < 0 and det(J) > 0
    # 3. Additional diffusion-related condition
    
    # Check diffusion-driven instability condition
    diff_cond1 = du_du * D2 + dv_dv * D1
    diff_cond2 = D1 * D2
    turing_cond = diff_cond1**2 - 4 * diff_cond2 * det
    
    # Full Turing conditions check
    is_turing_possible = is_stable and det > 0 and trace < 0 and turing_cond > 0
    
    # Compute diffusion ratio
    diffusion_ratio = D2 / D1
    
    results = {
        'is_stable_without_diffusion': is_stable,
        'trace': trace,
        'determinant': det,
        'eigenvalues': eigenvalues,
        'diffusion_ratio': diffusion_ratio,
        'turing_instability_condition': turing_cond > 0,
        'is_turing_possible': is_turing_possible,
        'steady_state': (u0, v0)
    }
    
    # Print results
    print(f"\nTuring conditions check:")
    print(f"  Steady state: u0={u0:.4f}, v0={v0:.4f}")
    print(f"  Stable without diffusion: {is_stable}")
    print(f"  Trace(J) = {trace:.4f} {'< 0 ✓' if trace < 0 else '≥ 0 ✗'}")
    print(f"  Det(J) = {det:.4f} {'> 0 ✓' if det > 0 else '≤ 0 ✗'}")
    print(f"  Diffusion ratio (D2/D1) = {diffusion_ratio:.4f}")
    print(f"  Diffusion instability condition: {turing_cond > 0}")
    print(f"  Turing pattern possible: {is_turing_possible}")
    
    return results

def run_parameter_extraction_example():
    """Run an example of parameter extraction and verification."""
    # Get strain library
    strain_library = create_strain_library()
    
    # 1. Extract parameters from strain models
    print("\n=== Extracting parameters from strain models ===")
    model_params = extract_reaction_parameters_from_model(strain_library)
    
    # 2. Create a simulation to fit parameters to
    print("\n=== Creating simulation for parameter fitting ===")
    # Create experiment with activator-inhibitor dynamics
    exp = WellMixedExperiment(grid_size=(50, 50), dx=0.1, simulation_time=(0, 48))
    
    # Add activator and inhibitor producing strains
    exp.add_strain('alpha->IAA', concentration=0.01)  # Inhibitor producing activator
    exp.add_strain('IAA->alpha', concentration=0.01)  # Activator producing inhibitor
    exp.add_strain('IAA->IAA', concentration=0.015)   # Activator autocatalysis
    
    # Add initial molecule concentrations
    from simulation_v7 import ALPHA, IAA
    exp.add_molecule(ALPHA, 5.0)
    exp.add_molecule(IAA, 0.5)
    
    # Run simulation
    print("Running simulation...")
    results = exp.run_simulation()
    
    # 3. Fit parameters to simulation results
    print("\n=== Fitting parameters to simulation results ===")
    fitted_params = optimize_parameters_from_simulation(
        exp, results, 
        init_params=model_params,
        time_idx=-1  # Use final time point
    )
    
    # 4. Verify Turing conditions
    print("\n=== Verifying Turing conditions ===")
    # Get diffusion coefficients
    D1 = exp.model.diffusion_coefficients['IAA']  # Activator diffusion
    D2 = exp.model.diffusion_coefficients['ALPHA']  # Inhibitor diffusion
    
    # Check extracted parameters
    if model_params:
        print("\nChecking model-extracted parameters:")
        model_check = verify_turing_conditions(model_params, D1, D2)
    
    # Check fitted parameters
    if fitted_params:
        print("\nChecking simulation-fitted parameters:")
        fitted_check = verify_turing_conditions(fitted_params, D1, D2)
    
    # 5. Suggest parameter adjustments if needed
    print("\n=== Parameter adjustment suggestions ===")
    if fitted_params:
        if not fitted_check['is_turing_possible']:
            print("Adjusting parameters to enable Turing patterns...")
            
            # Make a copy of fitted parameters
            adjusted_params = fitted_params.copy()
            
            # Common adjustments:
            # 1. Increase diffusion ratio (D2/D1)
            # 2. Adjust parameters to maintain stability without diffusion
            #    but allow instability with diffusion
            
            # Adjust parameters
            # For example: increase activator self-activation
            adjusted_params['a'] *= 1.2  # Increase activator production
            
            # Verify adjusted parameters
            print("\nChecking adjusted parameters:")
            adjusted_check = verify_turing_conditions(adjusted_params, D1, D2)
            
            if adjusted_check['is_turing_possible']:
                print("\nSuccessfully adjusted parameters to enable Turing patterns!")
                print(f"Adjusted parameters: a={adjusted_params['a']:.4f}, b={adjusted_params['b']:.4f}, "
                      f"c={adjusted_params['c']:.4f}, d={adjusted_params['d']:.4f}")
            else:
                # Try more aggressive adjustments
                print("\nTrying more aggressive parameter adjustments...")
                
                # More substantial adjustments:
                # 1. Increase activator production (a) and decay (b)
                # 2. Decrease inhibitor production (c) and increase removal (d)
                
                adjusted_params['a'] = fitted_params['a'] * 1.5
                adjusted_params['b'] = fitted_params['b'] * 1.2
                adjusted_params['c'] = fitted_params['c'] * 0.8
                adjusted_params['d'] = fitted_params['d'] * 1.2
                
                print("\nChecking more aggressively adjusted parameters:")
                adjusted_check2 = verify_turing_conditions(adjusted_params, D1, D2)
                
                if adjusted_check2['is_turing_possible']:
                    print("\nSuccessfully adjusted parameters to enable Turing patterns!")
                    print(f"Adjusted parameters: a={adjusted_params['a']:.4f}, b={adjusted_params['b']:.4f}, "
                          f"c={adjusted_params['c']:.4f}, d={adjusted_params['d']:.4f}")
                else:
                    print("\nCould not find suitable parameter adjustments.")
                    print("Consider increasing diffusion ratio or redesigning strains.")
        else:
            print("No parameter adjustments needed - Turing patterns are already possible.")
    
    return {
        'model_params': model_params,
        'fitted_params': fitted_params,
        'diffusion_coefficients': {'D1': D1, 'D2': D2}
    }

if __name__ == "__main__":
    # Run the example
    results = run_parameter_extraction_example()
    print("\nParameter extraction complete!")
