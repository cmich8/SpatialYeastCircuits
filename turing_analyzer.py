import numpy as np
from scipy.optimize import root
from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

class TuringAnalyzer:
    """
    A class to analyze and score reaction-diffusion systems for Turing pattern formation.
    Takes reaction terms and diffusion coefficients, evaluates if the system can form Turing patterns,
    and provides a quantitative score of how well the system satisfies the requirements.
    """
    
    def __init__(self, reaction_terms, diffusion_coefficients, params=None):
        """
        Initialize the analyzer with reaction terms and diffusion coefficients.
        
        Args:
            reaction_terms: Tuple of functions (f, g) where f(u, v, params) and g(u, v, params)
                           are the reaction terms for activator and inhibitor.
            diffusion_coefficients: Tuple (Du, Dv) of diffusion coefficients for activator and inhibitor.
            params: Dictionary of parameters for the reaction terms (optional).
        """
        self.f, self.g = reaction_terms
        self.Du, self.Dv = diffusion_coefficients
        self.params = params or {}
        
        # Default scoring weights
        self.weights = {
            'steady_state': 1.0,
            'stable_without_diffusion': 3.0,
            'trace_condition': 2.0,
            'determinant_condition': 2.0,
            'interaction_structure': 3.0,
            'diffusion_ratio': 2.5,
            'instability_with_diffusion': 3.5,
            'has_critical_wavenumber': 3.0
        }
    
    def set_scoring_weights(self, weights):
        """
        Set custom weights for the scoring system.
        
        Args:
            weights: Dictionary with weight values for each condition.
        """
        self.weights.update(weights)
    
    def find_steady_state(self, initial_guess=None):
        """
        Find the homogeneous steady state where f(u,v) = 0 and g(u,v) = 0.
        
        Args:
            initial_guess: Initial guess for (u, v) values.
            
        Returns:
            Tuple (u0, v0) of steady state values.
        """
        if initial_guess is None:
            initial_guess = [1.0, 1.0]  # Default initial guess
        
        def equations(vars):
            u, v = vars
            return [
                self.f(u, v, self.params),
                self.g(u, v, self.params)
            ]
        
        try:
            sol = root(equations, initial_guess)
            if sol.success:
                return sol.x
            else:
                raise ValueError("Failed to find steady state. Try different initial guess.")
        except Exception as e:
            print(f"Error finding steady state: {str(e)}")
            return None
    
    def compute_jacobian(self, steady_state):
        """
        Compute the Jacobian matrix at the given steady state.
        
        Args:
            steady_state: Tuple (u0, v0) of steady state values.
            
        Returns:
            2x2 Jacobian matrix.
        """
        u0, v0 = steady_state
        h = 1e-8  # Step size for numerical differentiation
        
        # Partial derivatives with respect to u
        df_du = (self.f(u0 + h, v0, self.params) - self.f(u0, v0, self.params)) / h
        dg_du = (self.g(u0 + h, v0, self.params) - self.g(u0, v0, self.params)) / h
        
        # Partial derivatives with respect to v
        df_dv = (self.f(u0, v0 + h, self.params) - self.f(u0, v0, self.params)) / h
        dg_dv = (self.g(u0, v0 + h, self.params) - self.g(u0, v0, self.params)) / h
        
        return np.array([[df_du, df_dv], [dg_du, dg_dv]])
    
    def analyze_stability(self, jacobian):
        """
        Analyze the stability of the system without diffusion.
        
        Args:
            jacobian: 2x2 Jacobian matrix at steady state.
            
        Returns:
            Dictionary with stability information.
        """
        trace = np.trace(jacobian)
        det = np.linalg.det(jacobian)
        eigenvalues = np.linalg.eigvals(jacobian)
        
        return {
            'trace': trace,
            'determinant': det,
            'eigenvalues': eigenvalues,
            'is_stable': trace < 0 and det > 0
        }
    
    def check_interaction_structure(self, jacobian):
        """
        Check if the interaction structure follows activator-inhibitor pattern.
        
        Args:
            jacobian: 2x2 Jacobian matrix at steady state.
            
        Returns:
            Dictionary with interaction structure information.
        """
        fu, fv = jacobian[0, 0], jacobian[0, 1]
        gu, gv = jacobian[1, 0], jacobian[1, 1]
        
        # Ideal activator-inhibitor system has:
        # fu > 0 (self-activation)
        # fv < 0 (cross-inhibition)
        # gu > 0 (cross-activation)
        # gv < 0 (self-inhibition)
        
        correct_structure = (fu > 0) and (fv < 0) and (gu > 0) and (gv < 0)
        
        # Calculate how well the structure matches the ideal
        structure_score = 0.0
        if fu > 0:  # Self-activation
            structure_score += 1.0
        if fv < 0:  # Cross-inhibition
            structure_score += 1.0
        if gu > 0:  # Cross-activation
            structure_score += 1.0
        if gv < 0:  # Self-inhibition
            structure_score += 1.0
        
        return {
            'correct_structure': correct_structure,
            'structure_score': structure_score / 4.0,
            'self_activation': fu > 0,
            'cross_inhibition': fv < 0,
            'cross_activation': gu > 0,
            'self_inhibition': gv < 0
        }
    
    def check_diffusion_instability(self, jacobian):
        """
        Check if diffusion causes instability.
        
        Args:
            jacobian: 2x2 Jacobian matrix at steady state.
            
        Returns:
            Dictionary with diffusion instability information.
        """
        fu, fv = jacobian[0, 0], jacobian[0, 1]
        gu, gv = jacobian[1, 0], jacobian[1, 1]
        Du, Dv = self.Du, self.Dv
        
        det_J = fu * gv - fv * gu
        
        # Condition 1: Dv*fu + Du*gv > 0
        condition1 = Dv * fu + Du * gv > 0
        
        # Condition 2: (Dv*fu + Du*gv)² > 4*Du*Dv*det_J
        condition2_left = (Dv * fu + Du * gv) ** 2
        condition2_right = 4 * Du * Dv * det_J
        condition2 = condition2_left > condition2_right
        
        # Compute diffusion ratio
        diffusion_ratio = Dv / Du
        
        # Critical ratio for instability based on Jacobian
        if fu > 0:
            critical_ratio = (fu ** 2) / det_J if det_J > 0 else float('inf')
        else:
            critical_ratio = float('inf')
        
        # How well the diffusion ratio exceeds the critical ratio
        ratio_score = min(1.0, diffusion_ratio / (10.0)) if critical_ratio > 0 else 0.0
        
        return {
            'condition1': condition1,
            'condition2': condition2,
            'diffusion_instability': condition1 and condition2,
            'diffusion_ratio': diffusion_ratio,
            'critical_ratio': critical_ratio,
            'ratio_sufficient': diffusion_ratio > 10,
            'ratio_score': ratio_score
        }
    
    def compute_dispersion_relation(self, jacobian, k_range=None):
        """
        Compute the dispersion relation for wavenumbers k.
        
        Args:
            jacobian: 2x2 Jacobian matrix at steady state.
            k_range: Range of wavenumbers to scan (default: 0.01 to 5)
            
        Returns:
            Dictionary with dispersion relation data.
        """
        if k_range is None:
            k_range = np.linspace(0.01, 5, 1000)
            
        fu, fv = jacobian[0, 0], jacobian[0, 1]
        gu, gv = jacobian[1, 0], jacobian[1, 1]
        Du, Dv = self.Du, self.Dv
        
        growth_rates = []
        
        for k in k_range:
            k_squared = k ** 2
            
            # Calculate terms in the characteristic equation
            trace_k = fu + gv - (Du + Dv) * k_squared
            det_k = (fu - Du * k_squared) * (gv - Dv * k_squared) - fv * gu
            
            # Calculate eigenvalues (growth rates) for this wavenumber
            discriminant = trace_k ** 2 - 4 * det_k
            
            if discriminant >= 0:
                lambda1 = 0.5 * (trace_k + np.sqrt(discriminant))
                lambda2 = 0.5 * (trace_k - np.sqrt(discriminant))
                growth_rate = max(lambda1, lambda2)
            else:
                # Complex eigenvalues, take real part
                real_part = 0.5 * trace_k
                growth_rate = real_part
            
            growth_rates.append(growth_rate)
        
        # Find maximum growth rate and corresponding wavenumber
        growth_rates = np.array(growth_rates)
        max_idx = np.argmax(growth_rates)
        k_max = k_range[max_idx]
        max_growth_rate = growth_rates[max_idx]
        
        # Calculate predicted wavelength
        if k_max > 0:
            predicted_wavelength = 2 * np.pi / k_max
        else:
            predicted_wavelength = float('inf')
        
        return {
            'k_range': k_range,
            'growth_rates': growth_rates,
            'k_max': k_max,
            'max_growth_rate': max_growth_rate,
            'predicted_wavelength': predicted_wavelength,
            'has_positive_growth': max_growth_rate > 0
        }
    
    def analyze_system(self, initial_guess=None):
        """
        Perform comprehensive analysis of the reaction-diffusion system.
        
        Args:
            initial_guess: Initial guess for steady state (optional).
            
        Returns:
            Dictionary with complete analysis results.
        """
        # Initialize results with default values
        results = {
            'has_steady_state': False,
            'stable_without_diffusion': False,
            'trace_negative': False,
            'determinant_positive': False,
            'correct_interaction_structure': False,
            'diffusion_ratio_sufficient': False,
            'diffusion_instability': False,
            'has_positive_growth': False,
            'is_turing_capable': False,
            'score': 0.0,
            'k_range': np.linspace(0.01, 5, 100),  # Default k range
            'growth_rates': np.zeros(100)           # Default growth rates (all zero)
        }
        
        try:
            # Step 1: Find steady state
            steady_state = self.find_steady_state(initial_guess)
            if steady_state is None:
                print("Warning: Could not find steady state")
                return results
            
            results['steady_state'] = steady_state
            results['has_steady_state'] = True
            
            # Step 2: Compute Jacobian at steady state
            jacobian = self.compute_jacobian(steady_state)
            results['jacobian'] = jacobian
            
            # Step 3: Analyze stability without diffusion
            stability = self.analyze_stability(jacobian)
            results.update(stability)
            results['trace_negative'] = stability['trace'] < 0
            results['determinant_positive'] = stability['determinant'] > 0
            results['stable_without_diffusion'] = stability['is_stable']
            
            # Step 4: Check interaction structure
            interaction = self.check_interaction_structure(jacobian)
            results.update(interaction)
            results['correct_interaction_structure'] = interaction['correct_structure']
            
            # Step 5: Check diffusion-driven instability
            diffusion = self.check_diffusion_instability(jacobian)
            results.update(diffusion)
            results['diffusion_ratio_sufficient'] = diffusion['ratio_sufficient']
            results['diffusion_instability'] = diffusion['diffusion_instability']
            
            # Step 6: Compute dispersion relation
            try:
                dispersion = self.compute_dispersion_relation(jacobian)
                results.update(dispersion)
                results['has_positive_growth'] = dispersion['has_positive_growth']
            except Exception as e:
                print(f"Warning: Error computing dispersion relation: {str(e)}")
                # Keep default k_range and growth_rates
            
            # Step 7: Determine if system is Turing capable
            results['is_turing_capable'] = (
                results['has_steady_state'] and
                results['stable_without_diffusion'] and
                results['diffusion_instability'] and
                results['has_positive_growth']
            )
            
            # Step 8: Calculate overall score
            score = 0.0
            max_score = 0.0
            
            # Score for having a steady state
            if results['has_steady_state']:
                score += self.weights['steady_state']
            max_score += self.weights['steady_state']
            
            # Score for stability without diffusion
            if results['stable_without_diffusion']:
                score += self.weights['stable_without_diffusion']
            max_score += self.weights['stable_without_diffusion']
            
            # Score for trace condition
            if results['trace_negative']:
                score += self.weights['trace_condition']
            max_score += self.weights['trace_condition']
            
            # Score for determinant condition
            if results['determinant_positive']:
                score += self.weights['determinant_condition']
            max_score += self.weights['determinant_condition']
            
            # Score for interaction structure (proportional)
            score += self.weights['interaction_structure'] * interaction['structure_score']
            max_score += self.weights['interaction_structure']
            
            # Score for diffusion ratio (proportional)
            score += self.weights['diffusion_ratio'] * diffusion['ratio_score']
            max_score += self.weights['diffusion_ratio']
            
            # Score for instability with diffusion
            if results['diffusion_instability']:
                score += self.weights['instability_with_diffusion']
            max_score += self.weights['instability_with_diffusion']
            
            # Score for having a critical wavenumber with positive growth
            if results['has_positive_growth']:
                score += self.weights['has_critical_wavenumber']
            max_score += self.weights['has_critical_wavenumber']
            
            # Normalize score to 0-100 scale
            if max_score > 0:
                results['score'] = 100.0 * (score / max_score)
            else:
                results['score'] = 0.0
                
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def plot_dispersion_relation(self, results, figsize=(10, 6)):
        """
        Plot the dispersion relation from analysis results.
        
        Args:
            results: Results dictionary from analyze_system().
            figsize: Figure size as (width, height) tuple.
            
        Returns:
            Matplotlib figure or None if plotting not possible.
        """
        # Check if required data is present
        if 'k_range' not in results or 'growth_rates' not in results:
            print("Warning: Results do not contain dispersion relation data. Cannot plot.")
            
            # Create a figure with an error message
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 
                   "Dispersion relation data not available.\nAnalysis may not have completed successfully.", 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            return fig
        
        # Create the plot with the available data
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(results['k_range'], results['growth_rates'])
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Check if maximum growth data is available
        if 'has_positive_growth' in results and results['has_positive_growth']:
            if 'k_max' in results and 'max_growth_rate' in results:
                ax.axvline(x=results['k_max'], color='r', linestyle='--')
                
                # Safe text placement
                y_pos = results['max_growth_rate'] / 2
                if y_pos == 0:
                    y_pos = 0.5 * max(results['growth_rates']) if len(results['growth_rates']) > 0 else 0.1
                
                wavelength_text = f"λ = {results['predicted_wavelength']:.2f}" if 'predicted_wavelength' in results else ""
                ax.text(results['k_max'], y_pos, 
                        f"k_max = {results['k_max']:.3f}\n{wavelength_text}", 
                        va='center', ha='right')
        
        ax.set_title("Dispersion Relation")
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("Growth Rate")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_report(self, results):
        """
        Create a text report summarizing the analysis results.
        
        Args:
            results: Results dictionary from analyze_system().
            
        Returns:
            Formatted text report.
        """
        report = "Turing Pattern Analysis Report\n"
        report += "===========================\n\n"
        
        report += f"Overall Score: {results['score']:.1f}/100\n"
        report += f"Turing-capable: {'YES' if results['is_turing_capable'] else 'NO'}\n\n"
        
        report += "Summary of Requirements:\n"
        report += f"1. Has steady state: {'✓' if results['has_steady_state'] else '✗'}\n"
        report += f"2. Stable without diffusion: {'✓' if results['stable_without_diffusion'] else '✗'}\n"
        report += f"3. Trace negative: {'✓' if results['trace_negative'] else '✗'}\n"
        report += f"4. Determinant positive: {'✓' if results['determinant_positive'] else '✗'}\n"
        report += f"5. Correct interaction structure: {'✓' if results['correct_interaction_structure'] else '✗'}\n"
        report += f"6. Diffusion ratio sufficient: {'✓' if results['diffusion_ratio_sufficient'] else '✗'}\n"
        report += f"7. Diffusion causes instability: {'✓' if results['diffusion_instability'] else '✗'}\n"
        report += f"8. Positive growth rate exists: {'✓' if results['has_positive_growth'] else '✗'}\n\n"
        
        report += "Detailed Analysis:\n"
        
        if results['has_steady_state']:
            u0, v0 = results['steady_state']
            report += f"Steady State: u₀ = {u0:.4f}, v₀ = {v0:.4f}\n\n"
        
        report += "Jacobian Matrix at Steady State:\n"
        if 'jacobian' in results:
            J = results['jacobian']
            report += f"[ {J[0,0]:.4f}  {J[0,1]:.4f} ]\n"
            report += f"[ {J[1,0]:.4f}  {J[1,1]:.4f} ]\n\n"
        
        # Fix for the error: check if trace is numeric before formatting
        if 'trace' in results:
            trace_value = results['trace']
            if isinstance(trace_value, (int, float)):
                report += f"Trace: {trace_value:.4f}\n"
            else:
                report += f"Trace: {trace_value}\n"
        else:
            report += "Trace: N/A\n"
        
        # Fix for the error: check if determinant is numeric before formatting
        if 'determinant' in results:
            det_value = results['determinant']
            if isinstance(det_value, (int, float)):
                report += f"Determinant: {det_value:.4f}\n"
            else:
                report += f"Determinant: {det_value}\n"
        else:
            report += "Determinant: N/A\n"
        
        if 'eigenvalues' in results:
            eigs = results['eigenvalues']
            # Check if eigenvalues are complex
            if np.iscomplexobj(eigs):
                report += f"Eigenvalues: {eigs[0]:.4f}, {eigs[1]:.4f}\n\n"
            else:
                report += f"Eigenvalues: {eigs[0]:.4f}, {eigs[1]:.4f}\n\n"
        
        report += "Interaction Structure:\n"
        report += f"- Self-activation: {'✓' if results.get('self_activation', False) else '✗'}\n"
        report += f"- Cross-inhibition: {'✓' if results.get('cross_inhibition', False) else '✗'}\n"
        report += f"- Cross-activation: {'✓' if results.get('cross_activation', False) else '✗'}\n"
        report += f"- Self-inhibition: {'✓' if results.get('self_inhibition', False) else '✗'}\n\n"
        
        # Fix for the diffusion ratio formatting
        diff_ratio = results.get('diffusion_ratio')
        if diff_ratio is not None and isinstance(diff_ratio, (int, float)):
            report += f"Diffusion Ratio (Dv/Du): {diff_ratio:.4f}\n"
        else:
            report += f"Diffusion Ratio (Dv/Du): {diff_ratio}\n"
            
        # Fix for the critical ratio formatting
        crit_ratio = results.get('critical_ratio')
        if crit_ratio is not None and isinstance(crit_ratio, (int, float)) and not np.isinf(crit_ratio):
            report += f"Critical Ratio Required: {crit_ratio:.4f}\n\n"
        else:
            report += f"Critical Ratio Required: {crit_ratio}\n\n"
        
        if results['has_positive_growth']:
            report += "Dispersion Relation Analysis:\n"
            max_growth = results.get('max_growth_rate')
            if max_growth is not None and isinstance(max_growth, (int, float)):
                report += f"- Maximum Growth Rate: {max_growth:.4f}\n"
            else:
                report += f"- Maximum Growth Rate: {max_growth}\n"
                
            k_max = results.get('k_max')
            if k_max is not None and isinstance(k_max, (int, float)):
                report += f"- Critical Wavenumber (k_max): {k_max:.4f}\n"
            else:
                report += f"- Critical Wavenumber (k_max): {k_max}\n"
                
            wavelength = results.get('predicted_wavelength')
            if wavelength is not None and isinstance(wavelength, (int, float)) and not np.isinf(wavelength):
                report += f"- Predicted Pattern Wavelength: {wavelength:.4f}\n"
            else:
                report += f"- Predicted Pattern Wavelength: {wavelength}\n"
        
        return report

# Define the classic Gierer-Meinhardt activator-inhibitor system
def f(u, v, p):
    """
    Activator reaction term for the Gierer-Meinhardt model.
    u: activator concentration
    v: inhibitor concentration
    """
    a = p.get('a', 0.1)      # Base production
    b = p.get('b', 1.0)      # Linear degradation
    rho = p.get('rho', 0.0)  # Basal production
    
    # Key term: u²/v is the autocatalytic production with inhibition
    return a - b*u + (u*u)/(v*(1.0 + 0.01*u*u)) + rho

def g(u, v, p):
    """
    Inhibitor reaction term for the Gierer-Meinhardt model.
    u: activator concentration
    v: inhibitor concentration
    """
    c = p.get('c', 0.9)      # Production rate by activator
    d = p.get('d', 1.0)      # Linear degradation
    
    # Inhibitor produced by activator
    return c*u*u - d*v

# Set parameters known to work well
params = {
    'a': 0.1,    # Base activator production
    'b': 1.0,    # Activator degradation 
    'c': 0.9,    # Inhibitor production rate
    'd': 0.9,    # Inhibitor degradation
    'rho': 0.01  # Small basal activator production
}

# Set diffusion coefficients with very high ratio
Du = 0.05       # Activator diffusion (slow)
Dv = 1.0        # Inhibitor diffusion (fast)
# Diffusion ratio of 20:1, well above the minimum needed

# Create analyzer
analyzer = TuringAnalyzer((f, g), (Du, Dv), params)

# Try multiple initial guesses if needed
initial_guesses = [
    [1.0, 1.0],   # Default guess
    [0.5, 0.5],   # Lower guess
    [2.0, 2.0],   # Higher guess
    [0.2, 0.8],   # Asymmetric guess
    [0.8, 0.2]    # Another asymmetric guess
]

# Try each initial guess until we get a valid result
for guess in initial_guesses:
    print(f"Trying with initial guess: {guess}")
    results = analyzer.analyze_system(initial_guess=guess)
    
    if results['has_steady_state']:
        print(f"Found steady state: {results['steady_state']}")
        break

# Print report
print("\n" + "="*50)
print(analyzer.create_report(results))

# Plot dispersion relation
fig = analyzer.plot_dispersion_relation(results)
plt.tight_layout()
plt.savefig("dispersion_relation.png")
plt.show()

# Plot score breakdown
weights = analyzer.weights
categories = list(weights.keys())
scores = []

for category in categories:
    if category == 'steady_state' and results['has_steady_state']:
        scores.append(weights[category])
    elif category == 'stable_without_diffusion' and results['stable_without_diffusion']:
        scores.append(weights[category])
    elif category == 'trace_condition' and results['trace_negative']:
        scores.append(weights[category])
    elif category == 'determinant_condition' and results['determinant_positive']:
        scores.append(weights[category])
    elif category == 'interaction_structure':
        scores.append(weights[category] * results['structure_score'])
    elif category == 'diffusion_ratio':
        scores.append(weights[category] * results['ratio_score'])
    elif category == 'instability_with_diffusion' and results['diffusion_instability']:
        scores.append(weights[category])
    elif category == 'has_critical_wavenumber' and results['has_positive_growth']:
        scores.append(weights[category])
    else:
        scores.append(0)

# Create score breakdown plot
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, scores, color='skyblue')

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.1f}', ha='center', va='bottom')

plt.xlabel('Criteria')
plt.ylabel('Score Contribution')
plt.title('Turing Pattern Score Breakdown')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("score_breakdown.png")
plt.show()

# Print key parameters
print("\nKey Parameter Values:")
print(f"Steady State: u₀ = {results['steady_state'][0]:.4f}, v₀ = {results['steady_state'][1]:.4f}")
print(f"Diffusion Ratio: {results['diffusion_ratio']:.2f} (>10 required)")
print(f"Predicted Pattern Wavelength: {results['predicted_wavelength']:.4f}")
print(f"Overall Score: {results['score']:.1f}/100")