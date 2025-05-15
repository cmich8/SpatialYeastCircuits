import numpy as np
from scipy.optimize import minimize

class TuringGradientOptimizer:
    def __init__(self, model, strain_library, analyzer, base_params):
        """
        Initialize the optimizer with your model and analysis tools.
        
        Args:
            model: The SpatialMultiStrainModel class
            strain_library: Dictionary of strain parameters
            analyzer: Instance of TuringPatternAnalyzer
            base_params: Dictionary of starting parameters
        """
        self.model = model
        self.strain_library = strain_library
        self.analyzer = analyzer
        self.base_params = base_params
        
    def objective_function(self, params_vector):
        """
        Objective function to minimize - returns negative quality
        since we want to maximize pattern quality.
        
        Args:
            params_vector: Array of parameters to optimize
            
        Returns:
            Negative quality metric (lower is better for optimization)
        """
        # Convert parameter vector back to dictionary
        params = self._vector_to_params(params_vector)
        
        # Run simulation with these parameters
        sim_results = self._run_simulation(params)
        
        # Analyze final pattern
        if sim_results is None:
            return 10000  # Large penalty for failed simulations
            
        final_pattern = sim_results['molecule_grids']['GFP'][-1]
        
        # Calculate effective diffusion ratio
        D1 = params.get('wire_density', 0) * 5 + 0.67  # Simple model of wire effect
        D2 = 0.15  # Alpha factor diffusion
        
        # Verify if pattern is a Turing pattern
        verification = self.analyzer.verify_turing_pattern(
            sim_results, 
            self.base_params,
            D1, D2,
            time_idx=-1,
            molecule='GFP'
        )
        
        # Calculate pattern metrics
        regularity = self.analyzer.pattern_regularity_metrics(final_pattern)
        
        # Create quality metric (higher is better)
        quality = 0
        
        # Reward Turing patterns
        if verification['is_turing_pattern']:
            quality += 5
            
        # Reward pattern regularity (low CV)
        cv = regularity['coefficient_of_variation'] 
        if cv < float('inf'):
            quality += max(0, 1 - 4*cv)
            
        # Reward matching theoretical wavelength
        if verification['wavelength_ratio'] > 0:
            wavelength_match = max(0, 1 - abs(verification['wavelength_ratio'] - 1))
            quality += 3 * wavelength_match
            
        # Reward isotropy
        if verification['is_isotropic']:
            quality += 2
            
        # Return negative quality for minimization
        return -quality
    
    def _vector_to_params(self, params_vector):
        """
        Convert parameter vector to parameter dictionary.
        
        Args:
            params_vector: Array of parameters
            
        Returns:
            Dictionary with parameter names and values
        """
        # Example mapping - customize for your parameters
        return {
            'wire_density': params_vector[0],
            'activator_concentration': params_vector[1],
            'inhibitor_concentration': params_vector[2]
        }
    
    def _run_simulation(self, params):
        """
        Run simulation with given parameters.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            Simulation results or None if simulation failed
        """
        try:
            # Setup model with parameters
            sim_model = self.model(grid_size=(100, 100), dx=0.1)
            
            # Create wire strain grid
            np.random.seed(42)
            wire_grid = np.random.random((100, 100)) < params['wire_density']
            
            # Add activator-producing strain
            sim_model.add_strain(self.strain_library['alpha->IAA'])
            sim_model.place_strain(0, 50, 50, shape="circle", radius=5, 
                                 concentration=params['activator_concentration'])
            
            # Add inhibitor-producing strain
            sim_model.add_strain(self.strain_library['IAA->alpha'])
            sim_model.place_strain(1, 50, 50, shape="circle", radius=5, 
                                 concentration=params['inhibitor_concentration'])
            
            # Add wire strain
            sim_model.add_strain(self.strain_library['IAA->IAA'], initial_grid=wire_grid)
            
            # Add reporter
            sim_model.add_strain(self.strain_library['IAA->GFP'])
            sim_model.place_strain(3, 0, 0, shape="rectangle", 
                                 width=100, height=100, concentration=0.5)
            
            # Run simulation
            sim_model.set_simulation_time(0, 24)
            results = sim_model.simulate(n_time_points=50)
            
            return results
            
        except Exception as e:
            print(f"Simulation failed: {str(e)}")
            return None
    
    def optimize(self, initial_params=None, bounds=None):
        """
        Run optimization to find best parameters.
        
        Args:
            initial_params: Initial parameter vector (optional)
            bounds: Parameter bounds as list of tuples (optional)
            
        Returns:
            Optimization results
        """
        if initial_params is None:
            # Default initial parameters
            initial_params = np.array([0.2, 1.0, 1.0])
            
        if bounds is None:
            # Default bounds
            bounds = [
                (0.05, 0.5),   # wire_density
                (0.1, 2.0),    # activator_concentration
                (0.1, 2.0)     # inhibitor_concentration
            ]
        
        # Run optimization
        result = minimize(
            self.objective_function,
            initial_params,
            method='L-BFGS-B',  # Works well with bounds
            bounds=bounds,
            options={'maxiter': 20}  # Limit iterations due to computational cost
        )
        
        # Convert result to parameter dictionary
        optimal_params = self._vector_to_params(result.x)
        
        return {
            'optimal_params': optimal_params,
            'optimization_result': result,
            'final_quality': -result.fun
        }

def optimize_with_numerical_gradient(self, initial_params, learning_rate=0.05, 
                                    n_iterations=10, step_size=0.01):
    """
    Optimize parameters using numerical gradient descent.
    
    Args:
        initial_params: Dictionary of initial parameters
        learning_rate: Learning rate for gradient descent
        n_iterations: Number of iterations
        step_size: Step size for numerical gradient calculation
        
    Returns:
        Optimization history and final parameters
    """
    # Convert initial parameters to vector
    params_keys = list(initial_params.keys())
    current_params = np.array([initial_params[k] for k in params_keys])
    
    # Optimization history
    history = []
    
    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}")
        
        # Current parameter values as dictionary
        current_dict = {k: v for k, v in zip(params_keys, current_params)}
        
        # Evaluate current quality
        current_quality = -self.objective_function(current_params)
        history.append({
            'iteration': i,
            'params': current_dict.copy(),
            'quality': current_quality
        })
        
        print(f"Current parameters: {current_dict}")
        print(f"Current quality: {current_quality}")
        
        # Calculate numerical gradient
        gradient = np.zeros_like(current_params)
        
        for j in range(len(current_params)):
            # Add small step to parameter j
            params_plus = current_params.copy()
            params_plus[j] += step_size
            
            # Calculate quality with parameter j increased
            quality_plus = -self.objective_function(params_plus)
            
            # Numerical gradient
            gradient[j] = (quality_plus - current_quality) / step_size
        
        # Update parameters using gradient
        current_params = current_params + learning_rate * gradient
        
        # Enforce bounds
        current_params = np.clip(current_params, [0.05, 0.1, 0.1], [0.5, 2.0, 2.0])
        
        print(f"Gradient: {gradient}")
        print(f"Updated parameters: {current_params}")
        print("-" * 50)
    
    # Final evaluation
    final_dict = {k: v for k, v in zip(params_keys, current_params)}
    final_quality = -self.objective_function(current_params)
    
    return {
        'history': history,
        'final_params': final_dict,
        'final_quality': final_quality
    }

# Initialize your components
analyzer = TuringPatternAnalyzer(grid_size=(100, 100))
strain_library = create_strain_library()

# Base parameters for reaction system
base_params = {
    'a': 0.1,
    'b': 0.9,
    'c': 0.1,
    'd': 0.9,
    'activator': 'IAA',
    'inhibitor': 'ALPHA'
}

# Initial parameters to optimize
initial_params = {
    'wire_density': 0.2,
    'activator_concentration': 1.0,
    'inhibitor_concentration': 1.0
}

# Create optimizer
optimizer = TuringGradientOptimizer(
    model=SpatialMultiStrainModel,
    strain_library=strain_library,
    analyzer=analyzer,
    base_params=base_params
)

# Run optimization
result = optimizer.optimize_with_numerical_gradient(
    initial_params=initial_params,
    learning_rate=0.05,
    n_iterations=10
)

# Plot optimization history
iterations = [h['iteration'] for h in result['history']]
qualities = [h['quality'] for h in result['history']]

plt.figure(figsize=(10, 6))
plt.plot(iterations, qualities, 'bo-')
plt.xlabel('Iteration')
plt.ylabel('Pattern Quality')
plt.title('Turing Pattern Optimization Progress')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Optimal parameters: {result['final_params']}")
print(f"Final quality: {result['final_quality']}")