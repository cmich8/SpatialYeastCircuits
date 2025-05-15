import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, root
from scipy.spatial import cKDTree
import numpy.linalg as la

class TuringPatternAnalyzer:
    """
    A comprehensive toolkit for analyzing and verifying Turing patterns
    in reaction-diffusion systems, particularly for synthetic biology applications
    using yeast strains with artificially enhanced diffusion via wire strains.
    """
    
    def __init__(self, grid_size=(100, 100)):
        """Initialize the analyzer with the grid size."""
        self.grid_size = grid_size
    
    def compute_jacobian(self, params, steady_state):
        """
        Compute the Jacobian matrix of the reaction terms at the homogeneous steady state.
        
        Args:
            params: Dictionary of reaction parameters
            steady_state: Homogeneous steady state values (u0, v0)
            
        Returns:
            2x2 Jacobian matrix of partial derivatives
        """
        # Extract reaction parameters
        a = params.get('a', 0.1)
        b = params.get('b', 0.9)
        c = params.get('c', 0.1)
        d = params.get('d', 0.9)
        
        # Unpack steady state
        u0, v0 = steady_state
        
        # Compute Jacobian matrix
        # For example, for a simple activator-inhibitor system:
        # du/dt = a - bu + u^2*v
        # dv/dt = c - du - u^2*v
        
        # Partial derivatives
        du_du = -b + 2*u0*v0
        du_dv = u0**2
        dv_du = -d - 2*u0*v0
        dv_dv = -u0**2
        
        # Assemble Jacobian
        J = np.array([[du_du, du_dv], 
                       [dv_du, dv_dv]])
        
        return J
    
    def find_homogeneous_steady_state(self, params, initial_guess=None):
        """
        Find the homogeneous steady state of the reaction system.
        
        Args:
            params: Dictionary of reaction parameters
            initial_guess: Initial guess for steady state (u0, v0)
            
        Returns:
            Steady state values (u0, v0)
        """
        # Extract reaction parameters
        a = params.get('a', 0.1)
        b = params.get('b', 0.9)
        c = params.get('c', 0.1)
        d = params.get('d', 0.9)
        
        if initial_guess is None:
            initial_guess = (1.0, 1.0)
        
        # Define the steady state equations
        def steady_state_eqs(x):
            u, v = x
            eq1 = a - b*u + u**2*v
            eq2 = c - d*u - u**2*v
            return [eq1, eq2]
        
        # Solve using root finding
        sol = root(steady_state_eqs, initial_guess)
        
        if sol.success:
            return sol.x
        else:
            raise ValueError("Failed to find homogeneous steady state")
    
    def check_homogeneous_stability(self, params, steady_state=None):
        """
        Check if the homogeneous steady state is stable without diffusion.
        
        Args:
            params: Dictionary of reaction parameters
            steady_state: Optional steady state values (u0, v0)
            
        Returns:
            Boolean indicating stability and eigenvalues
        """
        # Find steady state if not provided
        if steady_state is None:
            steady_state = self.find_homogeneous_steady_state(params)
        
        # Compute Jacobian at steady state
        J = self.compute_jacobian(params, steady_state)
        
        # Compute eigenvalues
        eigenvalues = la.eigvals(J)
        
        # Check if all real parts are negative (stable)
        is_stable = all(eigenvalue.real < 0 for eigenvalue in eigenvalues)
        
        return {
            'is_stable': is_stable,
            'eigenvalues': eigenvalues,
            'trace': np.trace(J),
            'determinant': la.det(J)
        }
    
    def dispersion_relation(self, k_squared, params, D1, D2, steady_state=None):
        """
        Compute the dispersion relation for spatial perturbations.
        
        Args:
            k_squared: Square of wavenumber k
            params: Dictionary of reaction parameters
            D1, D2: Diffusion coefficients
            steady_state: Optional steady state values
            
        Returns:
            Maximum growth rate for this wavenumber
        """
        # Find steady state if not provided
        if steady_state is None:
            steady_state = self.find_homogeneous_steady_state(params)
        
        # Compute Jacobian at steady state
        J = self.compute_jacobian(params, steady_state)
        
        # Add diffusion terms
        diffusion_matrix = np.array([[D1, 0], [0, D2]])
        J_k = J - k_squared * diffusion_matrix
        
        # Calculate eigenvalues
        eigenvalues = la.eigvals(J_k)
        
        # Return maximum real part (growth rate)
        return max(eigenvalues.real)
    
    def calculate_dispersion_relation(self, params, D1, D2, k_range=None, steady_state=None):
        """
        Calculate the dispersion relation over a range of wavenumbers.
        
        Args:
            params: Dictionary of reaction parameters
            D1, D2: Diffusion coefficients
            k_range: Range of wavenumbers to scan (default: 0 to 5)
            steady_state: Optional steady state values
            
        Returns:
            Dictionary with wavenumbers, growth rates, and predicted wavelength
        """
        # Find steady state if not provided
        if steady_state is None:
            steady_state = self.find_homogeneous_steady_state(params)
        
        # Set default k range if not provided
        if k_range is None:
            k_range = np.linspace(0.01, 5, 1000)
        
        # Calculate growth rate for each wavenumber
        growth_rates = [self.dispersion_relation(k**2, params, D1, D2, steady_state) for k in k_range]
        
        # Find k_max (wavenumber with maximum growth rate)
        max_idx = np.argmax(growth_rates)
        k_max = k_range[max_idx]
        max_growth_rate = growth_rates[max_idx]
        
        # Calculate predicted wavelength
        if k_max > 0:
            predicted_wavelength = 2 * np.pi / k_max
        else:
            predicted_wavelength = float('inf')
        
        return {
            'wavenumbers': k_range,
            'growth_rates': growth_rates,
            'k_max': k_max,
            'max_growth_rate': max_growth_rate,
            'predicted_wavelength': predicted_wavelength,
            'has_turing_instability': max_growth_rate > 0
        }
    
    def fourier_analysis(self, pattern):
        """
        Perform Fourier analysis on a spatial pattern.
        
        Args:
            pattern: 2D array of pattern values
            
        Returns:
            Dictionary with FFT results and power spectrum
        """
        # Compute 2D FFT
        fft = np.fft.fft2(pattern)
        fft_shifted = np.fft.fftshift(fft)
        power_spectrum = np.abs(fft_shifted)**2
        
        # Get frequency coordinates
        ny, nx = pattern.shape
        kx = np.fft.fftfreq(nx)
        ky = np.fft.fftfreq(ny)
        kx_shifted = np.fft.fftshift(kx)
        ky_shifted = np.fft.fftshift(ky)
        
        # Create meshgrid of wavenumbers
        kx_mesh, ky_mesh = np.meshgrid(kx_shifted, ky_shifted)
        k_magnitude = np.sqrt(kx_mesh**2 + ky_mesh**2)
        
        return {
            'fft': fft,
            'fft_shifted': fft_shifted,
            'power_spectrum': power_spectrum,
            'kx': kx_shifted,
            'ky': ky_shifted,
            'k_magnitude': k_magnitude
        }
    
    def analyze_pattern_wavelength(self, pattern):
        """
        Analyze the characteristic wavelength of a pattern using FFT.
        
        Args:
            pattern: 2D array of pattern values
            
        Returns:
            Dictionary with characteristic wavelength and radial profile
        """
        # Perform Fourier analysis
        fft_results = self.fourier_analysis(pattern)
        power_spectrum = fft_results['power_spectrum']
        k_magnitude = fft_results['k_magnitude']
        
        # Compute radial average of power spectrum
        k_bins = np.linspace(0, np.max(k_magnitude), 100)
        radial_profile = np.zeros_like(k_bins)
        
        for i in range(len(k_bins)-1):
            mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i+1])
            if np.any(mask):
                radial_profile[i] = np.mean(power_spectrum[mask])
        
        # Find peaks in radial profile (skip DC component)
        peaks, properties = find_peaks(radial_profile[1:], height=0)
        peaks = peaks + 1  # Adjust indices to account for skipped DC
        
        # Get most prominent peak (other than DC)
        if len(peaks) > 0:
            # Find peak with highest power
            peak_heights = properties['peak_heights']
            strongest_peak_idx = np.argmax(peak_heights)
            k_peak = k_bins[peaks[strongest_peak_idx]]
            
            # Convert to wavelength
            if k_peak > 0:
                wavelength = 1/k_peak
            else:
                wavelength = float('inf')
        else:
            k_peak = 0
            wavelength = float('inf')
        
        # Check for isotropy by examining 2D power spectrum
        # A ring in k-space indicates isotropy
        kx_center_idx = len(fft_results['kx']) // 2
        ky_center_idx = len(fft_results['ky']) // 2
        
        # Get power along x and y axes
        power_x = power_spectrum[ky_center_idx, :]
        power_y = power_spectrum[:, kx_center_idx]
        
        # Calculate correlation between x and y profiles
        # High correlation suggests isotropy
        correlation = np.corrcoef(power_x, power_y)[0, 1]
        is_isotropic = correlation > 0.7  # Threshold for isotropy
        
        return {
            'wavelength': wavelength,
            'k_peak': k_peak,
            'radial_profile': radial_profile,
            'k_bins': k_bins,
            'isotropy_correlation': correlation,
            'is_isotropic': is_isotropic
        }
    
    def spatial_autocorrelation(self, pattern):
        """
        Compute the 2D spatial autocorrelation of a pattern.
        
        Args:
            pattern: 2D array of pattern values
            
        Returns:
            2D autocorrelation
        """
        # Normalize pattern
        pattern_norm = pattern - np.mean(pattern)
        
        # Compute autocorrelation using FFT
        fft = np.fft.fft2(pattern_norm)
        power_spectrum = np.abs(fft)**2
        autocorr = np.fft.ifft2(power_spectrum).real
        
        # Normalize and center
        autocorr = np.fft.fftshift(autocorr)
        autocorr = autocorr / np.max(autocorr)
        
        return autocorr
    
    def extract_pattern_wavelength_from_autocorr(self, pattern):
        """
        Extract pattern wavelength from spatial autocorrelation.
        
        Args:
            pattern: 2D array of pattern values
            
        Returns:
            Dictionary with wavelength from autocorrelation
        """
        # Compute autocorrelation
        autocorr = self.spatial_autocorrelation(pattern)
        
        # Get pattern center
        center_y, center_x = np.array(autocorr.shape) // 2
        
        # Extract horizontal and vertical profiles through center
        h_profile = autocorr[center_y, :]
        v_profile = autocorr[:, center_x]
        
        # Find peaks in horizontal profile (excluding center peak)
        h_peaks, _ = find_peaks(h_profile[center_x+1:])
        h_peaks = h_peaks + center_x + 1  # Adjust indices
        
        # Find peaks in vertical profile (excluding center peak)
        v_peaks, _ = find_peaks(v_profile[center_y+1:])
        v_peaks = v_peaks + center_y + 1  # Adjust indices
        
        # Calculate wavelengths
        if len(h_peaks) > 0:
            h_wavelength = 2 * (h_peaks[0] - center_x)
        else:
            h_wavelength = float('inf')
            
        if len(v_peaks) > 0:
            v_wavelength = 2 * (v_peaks[0] - center_y)
        else:
            v_wavelength = float('inf')
        
        # Average wavelength
        if h_wavelength < float('inf') and v_wavelength < float('inf'):
            avg_wavelength = (h_wavelength + v_wavelength) / 2
        elif h_wavelength < float('inf'):
            avg_wavelength = h_wavelength
        elif v_wavelength < float('inf'):
            avg_wavelength = v_wavelength
        else:
            avg_wavelength = float('inf')
        
        return {
            'autocorrelation': autocorr,
            'h_wavelength': h_wavelength,
            'v_wavelength': v_wavelength,
            'avg_wavelength': avg_wavelength
        }
    
    def identify_local_maxima(self, pattern, min_distance=3):
        """
        Identify local maxima in a pattern.
        
        Args:
            pattern: 2D array of pattern values
            min_distance: Minimum distance between peaks
            
        Returns:
            Array of (y, x) coordinates of maxima
        """
        # Smooth pattern to reduce noise
        smoothed = ndimage.gaussian_filter(pattern, sigma=1.0)
        
        # Find local maxima
        data_max = ndimage.maximum_filter(smoothed, size=min_distance)
        maxima = (smoothed == data_max)
        
        # Remove maxima at boundaries
        maxima[0, :] = False
        maxima[-1, :] = False
        maxima[:, 0] = False
        maxima[:, -1] = False
        
        # Label and get positions of maxima
        labeled_maxima, num_maxima = ndimage.label(maxima)
        maxima_positions = ndimage.center_of_mass(maxima, labeled_maxima, range(1, num_maxima+1))
        
        return np.array(maxima_positions)
    
    def pattern_regularity_metrics(self, pattern, min_distance=3):
        """
        Calculate metrics for pattern regularity.
        
        Args:
            pattern: 2D array of pattern values
            min_distance: Minimum distance between peaks
            
        Returns:
            Dictionary with regularity metrics
        """
        # Identify local maxima
        maxima_positions = self.identify_local_maxima(pattern, min_distance)
        
        # Calculate nearest neighbor distances
        if len(maxima_positions) >= 5:  # Need enough peaks for statistics
            tree = cKDTree(maxima_positions)
            distances, _ = tree.query(maxima_positions, k=2)  # Get distance to nearest neighbor
            distances = distances[:, 1]  # Skip self (k=0)
            
            # Calculate statistics
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            cv = std_dist / mean_dist  # Coefficient of variation
            
            # For a true Turing pattern, expect low CV (regular spacing)
            is_turing_pattern = cv < 0.25  # Threshold for regularity
        else:
            mean_dist = 0
            std_dist = 0
            cv = float('inf')
            is_turing_pattern = False
        
        # Calculate hexagonal order parameter if enough peaks
        if len(maxima_positions) >= 7:
            # For each peak, find 6 nearest neighbors
            _, indices = tree.query(maxima_positions, k=7)  # k=7 includes self
            
            # Calculate hexagonal bond order parameter
            psi6_values = []
            for i, center_idx in enumerate(range(len(maxima_positions))):
                neighbors = indices[i, 1:]  # Skip self
                vectors = maxima_positions[neighbors] - maxima_positions[i]
                
                # Calculate angles to neighbors
                angles = np.arctan2(vectors[:, 0], vectors[:, 1])
                
                # Calculate ψ6 = 1/6 ∑ exp(6iθ)
                psi6 = np.mean(np.exp(6j * angles))
                psi6_values.append(abs(psi6))  # Take magnitude
            
            hex_order = np.mean(psi6_values)
        else:
            hex_order = 0
        
        return {
            'num_peaks': len(maxima_positions),
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'coefficient_of_variation': cv,
            'hexagonal_order': hex_order,
            'is_turing_pattern': is_turing_pattern,
            'peak_positions': maxima_positions
        }
    
    def track_wave_front(self, results, molecule='GFP', threshold=0.5):
        """
        Track the position of a wave front over time.
        
        Args:
            results: Simulation results dictionary
            molecule: Name of molecule to track
            threshold: Concentration threshold defining the wave front
            
        Returns:
            Array of wave front positions over time
        """
        # Get molecule data
        if molecule in results['molecule_grids']:
            molecule_data = results['molecule_grids'][molecule]
        else:
            raise ValueError(f"Molecule {molecule} not found in results")
        
        # Calculate wave front position at each time point
        wave_positions = []
        
        # Get dimensions
        grid_height, grid_width = self.grid_size
        center_y, center_x = grid_height // 2, grid_width // 2
        
        for time_idx, grid in enumerate(molecule_data):
            # Normalize grid
            max_val = np.max(grid)
            if max_val > 0:
                norm_grid = grid / max_val
            else:
                norm_grid = grid
            
            # Calculate distance from center to threshold contour
            # Average over multiple angles for robustness
            angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
            distances = []
            
            for angle in angles:
                # Create a line from center outward
                rho = min(grid_height, grid_width) // 2
                y = np.clip(
                    np.round(center_y + np.arange(rho) * np.sin(angle)).astype(int),
                    0, grid_height-1
                )
                x = np.clip(
                    np.round(center_x + np.arange(rho) * np.cos(angle)).astype(int),
                    0, grid_width-1
                )
                
                # Get values along this line
                values = norm_grid[y, x]
                
                # Find first point below threshold
                threshold_idx = np.argmax(values < threshold)
                if threshold_idx > 0:
                    distances.append(threshold_idx)
            
            # Average distance to wave front
            if distances:
                wave_positions.append(np.mean(distances))
            else:
                wave_positions.append(0)
        
        return np.array(wave_positions)
    
    def measure_effective_diffusion(self, wave_positions, times):
        """
        Measure effective diffusion coefficient from wave front propagation.
        
        Args:
            wave_positions: Array of wave front positions
            times: Array of time points
            
        Returns:
            Effective diffusion coefficient
        """
        # Skip initial points for fitting (t=0)
        positions = wave_positions[1:]
        fit_times = times[1:]
        
        # Define wave propagation model: x ≈ √(k*D*t)
        def wave_model(t, D_eff):
            return np.sqrt(4 * D_eff * t)
        
        # Fit model to data
        try:
            popt, pcov = curve_fit(wave_model, fit_times, positions)
            D_effective = popt[0]
            D_error = np.sqrt(pcov[0, 0])
            
            # Calculate R² goodness of fit
            residuals = positions - wave_model(fit_times, D_effective)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((positions - np.mean(positions))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            fit_quality = r_squared > 0.9  # Good fit if R² > 0.9
        except:
            D_effective = 0
            D_error = 0
            r_squared = 0
            fit_quality = False
        
        return {
            'effective_diffusion': D_effective,
            'error': D_error,
            'r_squared': r_squared,
            'good_fit': fit_quality
        }
    
    def verify_turing_pattern(self, simulation_results, params, D1, D2, time_idx=-1, molecule='GFP'):
        """
        Comprehensive verification that a pattern is a true Turing pattern.
        
        Args:
            simulation_results: Results from simulation
            params: Dictionary of reaction parameters
            D1, D2: Diffusion coefficients
            time_idx: Time index to analyze (default: last time point)
            molecule: Molecule to analyze
            
        Returns:
            Dictionary with verification results
        """
        # Get pattern at specified time
        pattern = simulation_results['molecule_grids'][molecule][time_idx]
        
        # 1. Check homogeneous steady state stability
        try:
            steady_state = self.find_homogeneous_steady_state(params)
            stability_result = self.check_homogeneous_stability(params, steady_state)
            is_stable_without_diffusion = stability_result['is_stable']
        except:
            is_stable_without_diffusion = False
            stability_result = {'is_stable': False}
        
        # 2. Check dispersion relation
        try:
            dispersion_data = self.calculate_dispersion_relation(params, D1, D2, steady_state=steady_state)
            has_positive_growth_rate = dispersion_data['max_growth_rate'] > 0
            theoretical_wavelength = dispersion_data['predicted_wavelength']
        except:
            has_positive_growth_rate = False
            theoretical_wavelength = 0
            dispersion_data = {'has_turing_instability': False}
        
        # 3. Analyze pattern wavelength using Fourier analysis
        wavelength_data = self.analyze_pattern_wavelength(pattern)
        measured_wavelength = wavelength_data['wavelength']
        
        # 4. Calculate pattern regularity metrics
        metrics = self.pattern_regularity_metrics(pattern)
        
        # 5. Check autocorrelation
        autocorr_data = self.extract_pattern_wavelength_from_autocorr(pattern)
        autocorr_wavelength = autocorr_data['avg_wavelength']
        
        # 6. Check if pattern wavelength matches prediction
        if theoretical_wavelength > 0 and measured_wavelength < float('inf'):
            wavelength_ratio = measured_wavelength / theoretical_wavelength
            wavelength_match = 0.7 < wavelength_ratio < 1.3  # Within 30% of prediction
        else:
            wavelength_ratio = 0
            wavelength_match = False
        
        # Check multiple conditions for a true Turing pattern
        is_turing_pattern = (
            is_stable_without_diffusion and 
            has_positive_growth_rate and 
            metrics['is_turing_pattern'] and
            wavelength_match and
            wavelength_data['is_isotropic']
        )
        
        # For wire strain modifications, check if the effective diffusion rates
        # satisfy Turing conditions (typically D2/D1 > 10)
        if 'effective_D1' in params and 'effective_D2' in params:
            D1_eff = params['effective_D1']
            D2_eff = params['effective_D2']
            diffusion_ratio = D2_eff / D1_eff if D1_eff > 0 else 0
            diffusion_ratio_sufficient = diffusion_ratio > 10
        else:
            diffusion_ratio = D2 / D1 if D1 > 0 else 0
            diffusion_ratio_sufficient = diffusion_ratio > 10
        
        return {
            'is_turing_pattern': is_turing_pattern,
            'stability_without_diffusion': is_stable_without_diffusion,
            'has_positive_growth_modes': has_positive_growth_rate,
            'has_turing_instability': dispersion_data.get('has_turing_instability', False),
            'pattern_regularity': metrics['coefficient_of_variation'],
            'is_regular': metrics['is_turing_pattern'],
            'measured_wavelength': measured_wavelength,
            'autocorr_wavelength': autocorr_wavelength,
            'theoretical_wavelength': theoretical_wavelength,
            'wavelength_ratio': wavelength_ratio,
            'wavelength_match': wavelength_match,
            'is_isotropic': wavelength_data['is_isotropic'],
            'diffusion_ratio': diffusion_ratio,
            'diffusion_ratio_sufficient': diffusion_ratio_sufficient,
            'hexagonal_order': metrics['hexagonal_order'],
            'num_peaks': metrics['num_peaks']
        }
    
    def plot_pattern_analysis(self, pattern, params=None, D1=None, D2=None, figsize=(16, 10)):
        """
        Create a comprehensive plot of pattern analysis.
        
        Args:
            pattern: 2D array of pattern values
            params: Optional dictionary of reaction parameters
            D1, D2: Optional diffusion coefficients
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure with comprehensive pattern analysis
        """
        fig = plt.figure(figsize=figsize)
        
        # Create grid for plots
        if params is not None and D1 is not None and D2 is not None:
            grid = plt.GridSpec(2, 4, figure=fig)
        else:
            grid = plt.GridSpec(2, 3, figure=fig)
        
        # 1. Original pattern
        ax1 = fig.add_subplot(grid[0, 0])
        im1 = ax1.imshow(pattern, cmap='viridis')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("Original Pattern")
        
        # 2. FFT Power Spectrum (log scale)
        fft_results = self.fourier_analysis(pattern)
        power_spectrum = fft_results['power_spectrum']
        
        ax2 = fig.add_subplot(grid[0, 1])
        # Add small constant and use log scale for better visualization
        im2 = ax2.imshow(np.log(power_spectrum + 1e-10), cmap='inferno')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title("FFT Power Spectrum (log scale)")
        
        # 3. Radial Power Profile
        wavelength_data = self.analyze_pattern_wavelength(pattern)
        radial_profile = wavelength_data['radial_profile']
        k_bins = wavelength_data['k_bins']
        
        ax3 = fig.add_subplot(grid[0, 2])
        ax3.plot(k_bins, radial_profile)
        if wavelength_data['k_peak'] > 0:
            ax3.axvline(x=wavelength_data['k_peak'], color='r', linestyle='--')
            ax3.text(wavelength_data['k_peak'], max(radial_profile)/2, 
                     f"k_peak = {wavelength_data['k_peak']:.3f}\nλ = {wavelength_data['wavelength']:.2f}", 
                     va='center', ha='right')
        ax3.set_title("Radial Power Spectrum")
        ax3.set_xlabel("Wavenumber k")
        ax3.set_ylabel("Power")
        
        # 4. Autocorrelation
        autocorr = self.spatial_autocorrelation(pattern)
        
        ax4 = fig.add_subplot(grid[1, 0])
        im4 = ax4.imshow(autocorr, cmap='coolwarm', vmin=-0.5, vmax=1.0)
        plt.colorbar(im4, ax=ax4)
        ax4.set_title("Spatial Autocorrelation")
        
        # 5. Local Maxima
        metrics = self.pattern_regularity_metrics(pattern)
        peak_positions = metrics['peak_positions']
        
        ax5 = fig.add_subplot(grid[1, 1])
        im5 = ax5.imshow(pattern, cmap='viridis')
        if len(peak_positions) > 0:
            ax5.plot(peak_positions[:, 1], peak_positions[:, 0], 'r.', markersize=8)
        ax5.set_title(f"Local Maxima (CV = {metrics['coefficient_of_variation']:.3f})")
        
        # 6. Nearest Neighbor Distances Histogram
        ax6 = fig.add_subplot(grid[1, 2])
        
        if len(peak_positions) >= 5:
            tree = cKDTree(peak_positions)
            distances, _ = tree.query(peak_positions, k=min(2, len(peak_positions)))
            distances = distances[:, 1] if distances.shape[1] > 1 else []  # Skip self
            
            if len(distances) > 0:
                ax6.hist(distances, bins=15, density=True, alpha=0.7)
                ax6.axvline(x=metrics['mean_distance'], color='r', linestyle='--')
                ax6.text(metrics['mean_distance'], 0.9*ax6.get_ylim()[1], 
                        f"Mean = {metrics['mean_distance']:.2f}", 
                        va='top', ha='right')
            else:
                ax6.text(0.5, 0.5, "Not enough data for histogram", 
                        va='center', ha='center', transform=ax6.transAxes)
        else:
            ax6.text(0.5, 0.5, "Not enough peaks for statistics", 
                    va='center', ha='center', transform=ax6.transAxes)
        
        ax6.set_title("Nearest Neighbor Distances")
        ax6.set_xlabel("Distance")
        
        # 7. Dispersion Relation (if parameters provided)
        if params is not None and D1 is not None and D2 is not None:
            try:
                # Find steady state
                steady_state = self.find_homogeneous_steady_state(params)
                
                # Calculate dispersion relation
                dispersion_data = self.calculate_dispersion_relation(params, D1, D2, steady_state=steady_state)
                
                ax7 = fig.add_subplot(grid[0, 3])
                ax7.plot(dispersion_data['wavenumbers'], dispersion_data['growth_rates'])
                ax7.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                if dispersion_data['max_growth_rate'] > 0:
                    ax7.axvline(x=dispersion_data['k_max'], color='r', linestyle='--')
                    ax7.text(dispersion_data['k_max'], max(dispersion_data['growth_rates'])/2, 
                            f"k_max = {dispersion_data['k_max']:.3f}\nλ = {dispersion_data['predicted_wavelength']:.2f}", 
                            va='center', ha='right')
                
                ax7.set_title("Dispersion Relation")
                ax7.set_xlabel("Wavenumber k")
                ax7.set_ylabel("Growth Rate")
                
                # 8. Wavelength comparison
                ax8 = fig.add_subplot(grid[1, 3])
                
                # Compare predicted and observed wavelengths
                wavelengths = [
                    dispersion_data['predicted_wavelength'],
                    wavelength_data['wavelength'],
                    metrics['mean_distance'],
                    self.extract_pattern_wavelength_from_autocorr(pattern)['avg_wavelength']
                ]
                labels = ['Predicted', 'FFT', 'Peak Dist', 'Autocorr']
                
                ax8.bar(labels, wavelengths, alpha=0.7)
                ax8.set_title("Wavelength Comparison")
                ax8.set_ylabel("Wavelength")
                
                # Add verification result as text
                verify_result = self.verify_turing_pattern(
                    {'molecule_grids': {'GFP': [pattern]}}, 
                    params, D1, D2, 
                    time_idx=0, 
                    molecule='GFP'
                )
                
                is_turing = verify_result['is_turing_pattern']
                text_color = 'green' if is_turing else 'red'
                ax8.text(0.5, -0.15, f"Turing Pattern: {is_turing}", 
                        color=text_color, fontweight='bold',
                        va='center', ha='center', transform=ax8.transAxes)
            except Exception as e:
                ax7 = fig.add_subplot(grid[0, 3])
                ax7.text(0.5, 0.5, f"Error calculating dispersion relation:\n{str(e)}", 
                        va='center', ha='center', transform=ax7.transAxes)
                
                ax8 = fig.add_subplot(grid[1, 3])
                ax8.text(0.5, 0.5, "Wavelength comparison unavailable", 
                        va='center', ha='center', transform=ax8.transAxes)
        
        plt.tight_layout()
        return fig