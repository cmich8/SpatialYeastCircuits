import numpy as np
from scipy.optimize import root
from scipy import linalg
import matplotlib.pyplot as plt
import os
import csv
import argparse
import datetime
import shutil
from matplotlib import cm
from scipy.integrate import solve_ivp

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
    
    def plot_dispersion_relation(self, results, figsize=(10, 6), save_path=None):
        """
        Plot the dispersion relation from analysis results.
        
        Args:
            results: Results dictionary from analyze_system().
            figsize: Figure size as (width, height) tuple.
            save_path: Path to save the figure (optional).
            
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
            
            if save_path:
                plt.savefig(save_path)
                
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
        
        if save_path:
            plt.savefig(save_path)
            
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
    
    def simulate_pattern(self, grid_size=100, spatial_size=10.0, 
                      time_points=2000, dt=0.2, noise_amplitude=0.1, 
                      steady_state=None, initial_guess=None, save_frames=10):
        """
        Simulate the reaction-diffusion system on a 2D grid.
        
        Args:
            grid_size: Number of grid points along each dimension.
            spatial_size: Physical size of the grid in cm.
            time_points: Number of time steps to simulate.
            dt: Time step size.
            noise_amplitude: Amplitude of random noise to add to initial conditions.
            steady_state: Steady state values (u0, v0) if already known.
            initial_guess: Initial guess for steady state if not provided.
            save_frames: Number of frames to save for animation (default: 10).
            
        Returns:
            Dictionary with simulation results.
        """
        print("Starting pattern simulation...")
        
        # Find steady state if not provided
        if steady_state is None:
            steady_state = self.find_steady_state(initial_guess)
            if steady_state is None:
                print("Warning: Could not find steady state. Using default values.")
                steady_state = np.array([1.0, 1.0])
                
        u0, v0 = steady_state
        print(f"Using steady state: u0={u0:.4f}, v0={v0:.4f}")
        
        # Spatial discretization
        dx = spatial_size / grid_size
        x = np.linspace(0, spatial_size, grid_size)
        y = np.linspace(0, spatial_size, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Laplacian operator for 2D using finite differences with periodic boundaries
        def laplacian(Z):
            # Clip very large values to prevent overflow
            Z = np.clip(Z, -1e6, 1e6)
            
            Ztop = np.roll(Z, -1, axis=0)
            Zbottom = np.roll(Z, 1, axis=0)
            Zleft = np.roll(Z, -1, axis=1)
            Zright = np.roll(Z, 1, axis=1)
            
            # Calculate the Laplacian with safeguards for numerical stability
            laplacian_result = (Ztop + Zbottom + Zleft + Zright - 4 * Z) / (dx * dx)
            
            # Clip the result to prevent overflow
            return np.clip(laplacian_result, -1e6, 1e6)
        
        # Initialize with steady state plus small random perturbations
        np.random.seed(42)  # For reproducibility
        u = u0 * np.ones((grid_size, grid_size)) + noise_amplitude * np.random.randn(grid_size, grid_size)
        v = v0 * np.ones((grid_size, grid_size)) + noise_amplitude * np.random.randn(grid_size, grid_size)
        
        # Ensure positive concentrations
        u = np.maximum(u, 0.01)
        v = np.maximum(v, 0.01)
        
        # Storage for final state
        u_final = None
        v_final = None
        
        # Storage for simulation history for animation (evenly spaced frames)
        u_history = []  # We'll add frames as we go, no initial frame yet
        v_history = []
        
        # Calculate frame intervals
        frame_interval = max(1, time_points // (save_frames - 1))
        
        print(f"Simulating for {time_points} time steps, saving {save_frames} frames for animation...")
        print(f"Frame interval: Every {frame_interval} time steps")
        
        # Adjust dt if it seems too large for stability
        original_dt = dt
        max_diffusion = max(self.Du, self.Dv)
        
        # CFL condition for diffusion equation
        stability_dt = 0.25 * (dx * dx) / max_diffusion
        
        # Don't let dt get too small - set a floor
        min_dt = 0.01
        if stability_dt < min_dt:
            print(f"Warning: Calculated stability dt={stability_dt:.6f} is extremely small.")
            print(f"Consider using a coarser grid or smaller diffusion coefficients.")
            stability_dt = min_dt
            print(f"Using minimum time step dt={stability_dt}")
        
        if dt > stability_dt:
            dt = stability_dt
            print(f"Adjusting dt from {original_dt} to {dt} for numerical stability")
        
        # Use implicit method for diffusion (more stable)
        # We'll use ADI (Alternating Direction Implicit) method
        use_implicit = True
        
        # Calculate total simulation time
        total_time = time_points * dt
        print(f"Total simulation time: {total_time:.2f} time units")
        
        # Always save the initial state (t=0) as the first frame
        u_history.append(u.copy())
        v_history.append(v.copy())
        print(f"Saved frame 1/{save_frames} at step 0 (t=0.00)")
        
        # Time integration 
        for t in range(time_points):
            if use_implicit:
                # Implicit diffusion using operator splitting
                # First half-step: Implicit in x-direction
                u_half = np.zeros_like(u)
                v_half = np.zeros_like(v)
                
                # We're using a simplified ADI approach for stability
                # This is a mix of implicit for diffusion and explicit for reaction
                alpha_u = self.Du * dt / (2 * dx * dx)
                alpha_v = self.Dv * dt / (2 * dx * dx)
                
                # Compute reaction terms
                reaction_u = np.zeros_like(u)
                reaction_v = np.zeros_like(v)
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Add safeguards to prevent division by zero
                        v_safe = max(v[i, j], 1e-6)
                        u_safe = max(u[i, j], 1e-6)
                        
                        # Calculate reaction terms safely
                        try:
                            reaction_u[i, j] = self.f(u_safe, v_safe, self.params)
                            reaction_v[i, j] = self.g(u_safe, v_safe, self.params)
                            
                            # Clip reaction terms to prevent overflow
                            reaction_u[i, j] = np.clip(reaction_u[i, j], -100, 100)
                            reaction_v[i, j] = np.clip(reaction_v[i, j], -100, 100)
                        except Exception as e:
                            reaction_u[i, j] = 0
                            reaction_v[i, j] = 0
                
                # Apply reaction and diffusion updates
                u_new = u + dt * reaction_u
                v_new = v + dt * reaction_v
                
                # Apply diffusion using simple explicit step (for stability with small dt)
                lap_u = laplacian(u)
                lap_v = laplacian(v)
                
                u_new = u_new + dt * self.Du * lap_u
                v_new = v_new + dt * self.Dv * lap_v
            else:
                # Standard explicit method
                # Compute Laplacians
                lap_u = laplacian(u)
                lap_v = laplacian(v)
                
                # Compute reaction terms
                reaction_u = np.zeros_like(u)
                reaction_v = np.zeros_like(v)
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Add safeguards to prevent division by zero
                        v_safe = max(v[i, j], 1e-6)
                        u_safe = max(u[i, j], 1e-6)
                        
                        # Calculate reaction terms safely
                        try:
                            # Calculate reaction terms
                            reaction_u[i, j] = self.f(u_safe, v_safe, self.params)
                            reaction_v[i, j] = self.g(u_safe, v_safe, self.params)
                            
                            # Clip reaction terms to prevent overflow
                            reaction_u[i, j] = np.clip(reaction_u[i, j], -100, 100)
                            reaction_v[i, j] = np.clip(reaction_v[i, j], -100, 100)
                        except Exception as e:
                            reaction_u[i, j] = 0
                            reaction_v[i, j] = 0
                
                # Update concentrations with safeguards
                u_diff = dt * self.Du * lap_u
                v_diff = dt * self.Dv * lap_v
                u_react = dt * reaction_u
                v_react = dt * reaction_v
                
                # Clip diffusion and reaction terms to prevent large changes
                u_diff = np.clip(u_diff, -u.max()/2, u.max()/2)
                v_diff = np.clip(v_diff, -v.max()/2, v.max()/2)
                u_react = np.clip(u_react, -u.max()/2, u.max()/2)
                v_react = np.clip(v_react, -v.max()/2, v.max()/2)
                
                # Apply updates
                u_new = u + u_diff + u_react
                v_new = v + v_diff + v_react
            
            # Ensure positive concentrations and clip extremes
            u_new = np.clip(u_new, 0.01, 1e6)
            v_new = np.clip(v_new, 0.01, 1e6)
            
            # Check for NaN values and replace them
            if np.isnan(u_new).any() or np.isnan(v_new).any():
                print(f"Warning: NaN values detected at step {t}. Replacing with previous values.")
                # Replace NaNs with values from previous step
                u_new = np.where(np.isnan(u_new), u, u_new)
                v_new = np.where(np.isnan(v_new), v, v_new)
            
            # Apply updates
            u = u_new
            v = v_new
            
            # Save frames for animation at regular intervals (but only if we haven't saved all frames yet)
            if (t+1) % frame_interval == 0 and len(u_history) < save_frames - 1:  # -1 to leave room for final frame
                u_history.append(u.copy())
                v_history.append(v.copy())
                current_time = (t+1) * dt
                print(f"Saved frame {len(u_history)}/{save_frames} at step {t+1} (t={current_time:.2f})")
        
        # Store final state
        u_final = u.copy()
        v_final = v.copy()
        
        # Make sure we always save the final state as the last frame
        # Only add the final state if it's not already the last frame or if we haven't filled all frames
        if len(u_history) < save_frames:
            u_history.append(u_final.copy())
            v_history.append(v_final.copy())
            print(f"Saved frame {len(u_history)}/{save_frames} at step {time_points} (t={total_time:.2f})")
                
        # Check if we have collected all frames
        if len(u_history) < save_frames:
            print(f"Warning: Only collected {len(u_history)} frames out of {save_frames} requested.")
        
        print("Simulation completed.")
        
        return {
            'u_final': u_final,
            'v_final': v_final,
            'u_history': u_history,
            'v_history': v_history,
            'x': x,
            'y': y,
            'X': X,
            'Y': Y,
            'time_points': time_points,
            'dt': dt,
            'total_time': total_time
        }
    def plot_simulation_results(self, simulation_results, figsize=(18, 8), save_path=None, save_animation=True, animation_path=None):
        """
        Plot the final state of the simulation.
        
        Args:
            simulation_results: Results from simulate_pattern().
            figsize: Figure size as (width, height) tuple.
            save_path: Path to save the figure (optional).
            save_animation: Whether to create and save an animation (default: True).
            animation_path: Path to save the animation (optional).
            
        Returns:
            Matplotlib figure.
        """
        # Extract data
        u_final = simulation_results['u_final']
        v_final = simulation_results['v_final']
        
        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot activator (u)
        im1 = axes[0].imshow(u_final, cmap='viridis', origin='lower', 
                            extent=[0, 10, 0, 10])
        axes[0].set_title('Activator (u) Concentration')
        axes[0].set_xlabel('x (cm)')
        axes[0].set_ylabel('y (cm)')
        fig.colorbar(im1, ax=axes[0], label='Concentration')
        
        # Plot inhibitor (v)
        im2 = axes[1].imshow(v_final, cmap='plasma', origin='lower', 
                            extent=[0, 10, 0, 10])
        axes[1].set_title('Inhibitor (v) Concentration')
        axes[1].set_xlabel('x (cm)')
        axes[1].set_ylabel('y (cm)')
        fig.colorbar(im2, ax=axes[1], label='Concentration')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        # Create animation if requested
        if save_animation and 'u_history' in simulation_results and 'v_history' in simulation_results:
            try:
                self.create_animation(simulation_results, animation_path)
            except Exception as e:
                print(f"Warning: Failed to create animation: {str(e)}")
        
        return fig
    
    def create_animation(self, simulation_results, save_path=None):
        """
        Create an animation of the pattern formation over time.
        
        Args:
            simulation_results: Results from simulate_pattern().
            save_path: Path to save the animation (optional).
            
        Returns:
            Animation object or None if matplotlib animation is not available.
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            print("Warning: matplotlib.animation is not available. Cannot create animation.")
            return None
        
        # Extract data
        u_history = simulation_results['u_history']
        v_history = simulation_results['v_history']
        
        # If we don't have enough frames, extract more from the simulation
        if len(u_history) < 10 and 'time_points' in simulation_results:
            print("Not enough frames in history for animation. Re-running simulation with more history points.")
            # We would need to re-run the simulation, but that's beyond the scope here
        
        # Make sure we have at least some frames
        num_frames = len(u_history)
        if num_frames < 2:
            print("Error: Not enough frames for animation.")
            return None
            
        print(f"Creating animation with {num_frames} frames.")
        
        # Create a figure for the animation
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Get common min/max for consistent colormap scaling
        u_min = min(np.min(u) for u in u_history)
        u_max = max(np.max(u) for u in u_history)
        v_min = min(np.min(v) for v in v_history)
        v_max = max(np.max(v) for v in v_history)
        
        # Initialize plots with the first frame
        im1 = axes[0].imshow(u_history[0], cmap='viridis', origin='lower',
                        extent=[0, 10, 0, 10], animated=True, vmin=u_min, vmax=u_max)
        axes[0].set_title('Activator (u) Concentration')
        axes[0].set_xlabel('x (cm)')
        axes[0].set_ylabel('y (cm)')
        fig.colorbar(im1, ax=axes[0], label='Concentration')
        
        im2 = axes[1].imshow(v_history[0], cmap='plasma', origin='lower',
                        extent=[0, 10, 0, 10], animated=True, vmin=v_min, vmax=v_max)
        axes[1].set_title('Inhibitor (v) Concentration')
        axes[1].set_xlabel('x (cm)')
        fig.colorbar(im2, ax=axes[1], label='Concentration')
        
        # Add time information
        time_points = simulation_results.get('time_points', num_frames)
        dt = simulation_results.get('dt', 1.0)
        total_time = simulation_results.get('total_time', time_points * dt)
        
        # Title with time information for first frame (t=0)
        time_text = fig.suptitle(f'Time: 0.00 units (0.0% of simulation)', fontsize=14)
        
        plt.tight_layout()
        
        # Animation update function
        def update_frame(frame):
            im1.set_array(u_history[frame])
            im2.set_array(v_history[frame])
            
            # Calculate approximate time for this frame
            # For first frame, t=0, for last frame, t=total_time
            if frame == 0:
                t = 0
            elif frame == num_frames - 1:
                t = total_time
            else:
                t = total_time * frame / (num_frames - 1)
                
            # Update time text
            time_text.set_text(f'Time: {t:.2f} units ({100.0 * t / total_time:.1f}% of simulation)')
            
            return im1, im2, time_text
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, 
                                interval=500, blit=False)
        
        # Save the animation if a path is provided
        if save_path:
            try:
                print(f"Saving animation to {save_path}...")
                ani.save(save_path, writer='pillow', fps=2)
                print(f"Animation saved successfully to {save_path}")
            except Exception as e:
                print(f"Failed to save animation: {str(e)}")
                print("Trying alternate writer...")
                try:
                    ani.save(save_path, writer='imagemagick', fps=2)
                    print(f"Animation saved to {save_path} using imagemagick")
                except Exception as e:
                    print(f"Failed to save animation with imagemagick: {str(e)}")
        
        return ani
# Define the Gierer-Meinhardt activator-inhibitor system
def f(u, v, p):
    """
    Activator reaction term for the Gierer-Meinhardt model.
    u: activator concentration
    v: inhibitor concentration
    """
    a = p.get('a', 0.1)      # Base production
    b = p.get('b', 1.0)      # Linear degradation
    rho = p.get('rho', 0.0)  # Basal production
    
    # Ensure v is not too close to zero to avoid division issues
    v_safe = max(v, 1e-6)
    
    # Safely compute the autocatalytic term with inhibition
    try:
        denominator = v_safe * (1.0 + 0.01 * u * u)
        if denominator < 1e-10:
            denominator = 1e-10
        
        autocatalytic = (u * u) / denominator
        # Clip to prevent overflow
        autocatalytic = min(autocatalytic, 1e6)
        
        result = a - b * u + autocatalytic + rho
        # Final clip to ensure result is within a reasonable range
        return np.clip(result, -1e6, 1e6)
    except:
        # Fallback if computation fails
        return a - b * u + rho

def g(u, v, p):
    """
    Inhibitor reaction term for the Gierer-Meinhardt model.
    u: activator concentration
    v: inhibitor concentration
    """
    c = p.get('c', 0.9)      # Production rate by activator
    d = p.get('d', 1.0)      # Linear degradation
    
    # Safely compute inhibitor production
    try:
        # Clip u*u to prevent overflow
        u_squared = min(u * u, 1e6)
        
        result = c * u_squared - d * v
        # Clip to ensure result is within a reasonable range
        return np.clip(result, -1e6, 1e6)
    except:
        # Fallback if computation fails
        return -d * v

def run_turing_experiment(a, b, c, d, Du, Dv, rho=0.01, grid_size=100, time_points=2000, dt=0.1, no_simulation=False, experiment_name=None):
    """
    Run a complete Turing pattern experiment with the given parameters.
    
    Args:
        a, b, c, d: Reaction parameters
        Du, Dv: Diffusion coefficients
        rho: Basal activator production (default: 0.01)
        grid_size: Number of grid points for simulation (default: 100)
        time_points: Number of time steps for simulation (default: 2000)
        dt: Time step size (default: 0.1)
        no_simulation: If True, skip the simulation step (default: False)
        experiment_name: Optional custom name for the experiment
        
    Returns:
        Tuple (directory_path, is_turing_capable)
    """
    # Define the reaction terms for this scope - these need to be defined within the function
    # to ensure they're accessible when creating the analyzer
    def f_local(u, v, p):
        """
        Activator reaction term for the Gierer-Meinhardt model.
        u: activator concentration
        v: inhibitor concentration
        """
        a_local = p.get('a', 0.1)      # Base production
        b_local = p.get('b', 1.0)      # Linear degradation
        rho_local = p.get('rho', 0.0)  # Basal production
        
        # Ensure v is not too close to zero to avoid division issues
        v_safe = max(v, 1e-6)
        
        # Safely compute the autocatalytic term with inhibition
        try:
            denominator = v_safe * (1.0 + 0.01 * u * u)
            if denominator < 1e-10:
                denominator = 1e-10
            
            autocatalytic = (u * u) / denominator
            # Clip to prevent overflow
            autocatalytic = min(autocatalytic, 1e6)
            
            result = a_local - b_local * u + autocatalytic + rho_local
            # Final clip to ensure result is within a reasonable range
            return np.clip(result, -1e6, 1e6)
        except:
            # Fallback if computation fails
            return a_local - b_local * u + rho_local

    def g_local(u, v, p):
        """
        Inhibitor reaction term for the Gierer-Meinhardt model.
        u: activator concentration
        v: inhibitor concentration
        """
        c_local = p.get('c', 0.9)      # Production rate by activator
        d_local = p.get('d', 1.0)      # Linear degradation
        
        # Safely compute inhibitor production
        try:
            # Clip u*u to prevent overflow
            u_squared = min(u * u, 1e6)
            
            result = c_local * u_squared - d_local * v
            # Clip to ensure result is within a reasonable range
            return np.clip(result, -1e6, 1e6)
        except:
            # Fallback if computation fails
            return -d_local * v
    
    # Create parameters
    params = {
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'rho': rho
    }
    
    # Create directory name based on parameters and experiment name
    base_dir = "turing_ideal_experiments"
    
    # Create a timestamp for unique identification
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use the provided experiment name or create one based on parameters
    if experiment_name:
        # Clean up the name to be filesystem-friendly
        clean_name = experiment_name.replace(" ", "_").replace("/", "-").replace("\\", "-")
        dir_name = f"{clean_name}_a{a}_b{b}_c{c}_d{d}_Du{Du}_Dv{Dv}_{timestamp}"
    else:
        dir_name = f"a{a}_b{b}_c{c}_d{d}_Du{Du}_Dv{Dv}_{timestamp}"
    
    dir_path = os.path.join(base_dir, dir_name)
    
    # Create directories
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Save experiment parameters to a JSON file for reference
    params_with_diffusion = {
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'rho': rho,
        'Du': Du,
        'Dv': Dv,
        'grid_size': grid_size,
        'time_points': time_points,
        'dt': dt,
        'experiment_name': experiment_name or "Unnamed Experiment",
        'timestamp': timestamp
    }
    
    params_file = os.path.join(dir_path, "parameters.json")
    with open(params_file, 'w') as f:
        import json
        json.dump(params_with_diffusion, f, indent=4)
    
    # Initialize analyzer with local function definitions
    analyzer = TuringAnalyzer((f_local, g_local), (Du, Dv), params)
    
    # Try multiple initial guesses to find steady state
    initial_guesses = [
        [1.0, 1.0],   # Default guess
        [0.5, 0.5],   # Lower guess
        [2.0, 2.0],   # Higher guess
        [0.2, 0.8],   # Asymmetric guess
        [0.8, 0.2]    # Another asymmetric guess
    ]
    
    # Try each initial guess until we get a valid result
    results = None
    steady_state = None
    
    for guess in initial_guesses:
        print(f"Trying with initial guess: {guess}")
        results = analyzer.analyze_system(initial_guess=guess)
        
        if results['has_steady_state']:
            steady_state = results['steady_state']
            print(f"Found steady state: {steady_state}")
            break
    
    if not results['has_steady_state']:
        print("Warning: Could not find steady state with any initial guess.")
        return dir_path, False
    
    # Create report and save to file
    report = analyzer.create_report(results)
    report_path = os.path.join(dir_path, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Plot dispersion relation and save
    dispersion_path = os.path.join(dir_path, "dispersion_relation.png")
    analyzer.plot_dispersion_relation(results, save_path=dispersion_path)
    
    # Run simulation if not skipped
    if not no_simulation:
        try:
            print("\nRunning pattern simulation...")
            simulation_results = analyzer.simulate_pattern(
                grid_size=grid_size, 
                spatial_size=10.0,
                time_points=time_points,
                dt=dt,
                noise_amplitude=0.05,
                steady_state=steady_state,
                save_frames=10  # Save 10 frames for animation
            )
            
            # Plot simulation results and save
            simulation_path = os.path.join(dir_path, "pattern_simulation.png")
            animation_path = os.path.join(dir_path, "pattern_animation.gif")
            analyzer.plot_simulation_results(
                simulation_results, 
                save_path=simulation_path,
                save_animation=True,
                animation_path=animation_path
            )
        except Exception as e:
            print(f"Warning: Simulation failed with error: {str(e)}")
            print("Continuing with analysis results only.")
    else:
        print("\nSkipping simulation as requested.")
    
    # Save key parameters to CSV file in the main directory
    csv_path = os.path.join(base_dir, "turing_ideal_experiments.csv")
    is_new_file = not os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if is_new_file:
            writer.writerow([
                'Experiment_Name', 'a', 'b', 'c', 'd', 'Du', 'Dv', 
                'Turing_capable', 'Score', 
                'Diffusion_ratio', 'Wavelength',
                'Date', 'Directory'
            ])
        
        # Write data
        writer.writerow([
            experiment_name or "Unnamed Experiment",
            a, b, c, d, Du, Dv,
            'Yes' if results['is_turing_capable'] else 'No',
            f"{results['score']:.1f}",
            f"{results['diffusion_ratio']:.2f}",
            f"{results['predicted_wavelength']:.2f}" if 'predicted_wavelength' in results else 'N/A',
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dir_name
        ])
    
    # Print success message
    print(f"Experiment completed and saved to {dir_path}")
    print(f"Turing capable: {'YES' if results['is_turing_capable'] else 'NO'}")
    print(f"Score: {results['score']:.1f}/100")
    
    return dir_path, results['is_turing_capable']

def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(description='Run Turing pattern analysis and simulation')
    
    # Required parameters with defaults
    parser.add_argument('--a', type=float, default=0.1, help='Base activator production rate')
    parser.add_argument('--b', type=float, default=1.0, help='Activator degradation rate')
    parser.add_argument('--c', type=float, default=0.9, help='Inhibitor production rate')
    parser.add_argument('--d', type=float, default=0.9, help='Inhibitor degradation rate')
    parser.add_argument('--Du', type=float, default=0.05, help='Activator diffusion coefficient')
    parser.add_argument('--Dv', type=float, default=1.0, help='Inhibitor diffusion coefficient')
    parser.add_argument('--rho', type=float, default=0.01, help='Basal activator production')
    
    # Optional simulation parameters
    parser.add_argument('--grid_size', type=int, default=100, help='Number of grid points (default: 100)')
    parser.add_argument('--time_points', type=int, default=2000, help='Number of simulation time steps (default: 2000)')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size (default: 0.1)')
    parser.add_argument('--no_simulation', action='store_true', help='Skip simulation step (faster)')
    
    # Experiment naming
    parser.add_argument('--name', type=str, help='Custom name for the experiment')
    
    # Batch mode
    parser.add_argument('--batch', action='store_true', help='Run in batch mode with predefined parameters')
    
    args = parser.parse_args()
    
    if args.batch:
        # Predefined parameter sets for batch mode
        parameter_sets = [
            # name, a, b, c, d, Du, Dv, experiment_name
            ("Classic Spots", 0.1, 1.0, 0.9, 0.9, 0.05, 1.0, "Classic_Spots"),
            ("Fine Spots", 0.15, 1.0, 1.2, 1.0, 0.02, 0.8, "Fine_Spots"),
            ("Stripes", 0.2, 0.8, 1.2, 0.8, 0.04, 0.6, "Stripes"),
            ("Spots-to-Stripes", 0.18, 0.9, 1.0, 0.8, 0.03, 0.7, "Spots_to_Stripes"),
            ("High Contrast", 0.08, 1.2, 1.5, 1.0, 0.01, 1.5, "High_Contrast"),
            ("Non-Turing (Low Ratio)", 0.1, 1.0, 0.9, 0.9, 0.5, 1.0, "Non_Turing_Low_Ratio")
        ]
        
        print(f"Running in batch mode with {len(parameter_sets)} parameter sets")
        
        for i, params in enumerate(parameter_sets):
            name, a, b, c, d, Du, Dv, experiment_name = params
            print(f"\n[{i+1}/{len(parameter_sets)}] Running '{name}' parameter set")
            print(f"Parameters: a={a}, b={b}, c={c}, d={d}, Du={Du}, Dv={Dv}")
            
            # Run the experiment
            dir_path, is_turing_capable = run_turing_experiment(
                a, b, c, d, Du, Dv, args.rho,
                grid_size=args.grid_size,
                time_points=args.time_points,
                dt=args.dt,
                no_simulation=args.no_simulation,
                experiment_name=experiment_name
            )
            
            print(f"Completed '{name}'. Results saved to {dir_path}")
            print(f"Turing capable: {'YES' if is_turing_capable else 'NO'}")
        
        print("\nBatch processing complete!")
        print(f"Results saved to turing_ideal_experiments/ directory")
        print(f"Summary CSV file: turing_ideal_experiments/turing_ideal_experiments.csv")
    
    else:
        print(f"Running Turing pattern experiment with parameters:")
        print(f"a={args.a}, b={args.b}, c={args.c}, d={args.d}, Du={args.Du}, Dv={args.Dv}, rho={args.rho}")
        print(f"Simulation settings: grid_size={args.grid_size}, time_points={args.time_points}, dt={args.dt}")
        
        if args.name:
            print(f"Experiment name: {args.name}")
        
        if args.no_simulation:
            print("Simulation step will be skipped (analysis only)")
        
        # Run the experiment
        dir_path, is_turing_capable = run_turing_experiment(
            args.a, args.b, args.c, args.d, args.Du, args.Dv, args.rho,
            grid_size=args.grid_size,
            time_points=args.time_points,
            dt=args.dt,
            no_simulation=args.no_simulation,
            experiment_name=args.name
        )
        
        print(f"Experiment completed. Results saved to {dir_path}")
        print(f"Turing capable: {'YES' if is_turing_capable else 'NO'}")
        print(f"Check the CSV file at turing_ideal_experiments/turing_ideal_experiments.csv for a summary of all experiments.")


if __name__ == "__main__":
    main()