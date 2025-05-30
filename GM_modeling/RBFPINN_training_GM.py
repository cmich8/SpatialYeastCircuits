import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from PIL import Image
import json
import datetime
from pathlib import Path

class GiererMeinhardtRBFPINN:
    """
    RBF-PINN implementation for Gierer-Meinhardt parameter inference
    Based on the methodology from Matas-Gil & Endres (2024) but adapted for G-M equations
    """
    
    def __init__(self, num_rbf_units=80, model_type='classic', spatial_domain=(0, 10), 
                 grid_size=50, learning_rate=0.001):
        """
        Initialize the RBF-PINN for Gierer-Meinhardt equations
        
        Args:
            num_rbf_units: Number of RBF kernels in the hidden layer
            model_type: 'classic' or 'saturated' G-M model
            spatial_domain: Tuple (x_min, x_max) for spatial domain
            grid_size: Number of grid points per dimension
            learning_rate: Learning rate for optimization
        """
        self.num_rbf_units = num_rbf_units
        self.model_type = model_type
        self.spatial_domain = spatial_domain
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        
        # Parameter names for different G-M models
        if model_type == 'classic':
            self.param_names = ['a', 'b', 'c', 'd', 'Du', 'Dv']
        elif model_type == 'saturated':
            self.param_names = ['a', 'b', 'c', 'd', 'saturation', 'Du', 'Dv']
        else:
            raise ValueError("model_type must be 'classic' or 'saturated'")
        
        self.num_params = len(self.param_names)
        
        # Create spatial coordinate grids
        self.x = np.linspace(spatial_domain[0], spatial_domain[1], grid_size)
        self.y = np.linspace(spatial_domain[0], spatial_domain[1], grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.coords = np.stack([self.X.flatten(), self.Y.flatten()], axis=1)
        
        # Build networks
        self.u_network = self._build_rbf_network(name='u_network')
        self.v_network = self._build_rbf_network(name='v_network')
        
        # Initialize model parameters as trainable variables
        self._initialize_parameters()
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Training history
        self.history = {
            'total_loss': [],
            'data_loss_u': [],
            'data_loss_v': [],
            'pde_loss': [],
            'parameters': {name: [] for name in self.param_names}
        }
        
    def _build_rbf_network(self, name):
        """Build a single RBF network for u or v approximation"""
        
        # Input layer: spatial coordinates (x, y)
        inputs = keras.layers.Input(shape=(2,), name=f'{name}_input')
        
        # RBF layer - we'll implement this as a custom layer
        rbf_layer = RBFLayer(self.num_rbf_units, 
                           spatial_bounds=self.spatial_domain,
                           name=f'{name}_rbf')
        rbf_output = rbf_layer(inputs)
        
        # Output layer - single concentration value
        outputs = keras.layers.Dense(1, activation='linear', 
                                   name=f'{name}_output')(rbf_output)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def _initialize_parameters(self):
        """Initialize G-M model parameters as trainable variables"""
        
        # Default parameter ranges based on typical G-M values
        param_ranges = {
            'classic': {
                'a': (0.05, 0.2),
                'b': (0.8, 1.2), 
                'c': (0.5, 1.5),
                'd': (0.5, 1.5),
                'Du': (0.01, 0.1),
                'Dv': (0.5, 2.0)
            },
            'saturated': {
                'a': (0.05, 0.2),
                'b': (0.8, 1.2),
                'c': (0.5, 1.5), 
                'd': (0.5, 1.5),
                'saturation': (0.001, 0.1),
                'Du': (0.01, 0.1),
                'Dv': (0.5, 2.0)
            }
        }
        
        ranges = param_ranges[self.model_type]
        
        # Initialize parameters randomly within reasonable ranges
        self.model_params = {}
        for name in self.param_names:
            low, high = ranges[name]
            initial_value = np.random.uniform(low, high)
            self.model_params[name] = tf.Variable(
                initial_value, 
                trainable=True, 
                name=f'param_{name}',
                dtype=tf.float32
            )
    
    def set_initial_parameters(self, param_dict):
        """Set initial parameter values from a dictionary"""
        for name, value in param_dict.items():
            if name in self.model_params:
                self.model_params[name].assign(float(value))
    
    @tf.function
    def compute_laplacian(self, network, coords):
        """Compute the Laplacian using persistent gradient tapes"""
        
        # Ensure coords are the right type and shape
        coords = tf.cast(coords, tf.float32)
        
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                # Explicitly watch the coordinates
                tape1.watch(coords)
                tape2.watch(coords)
                
                # Get network output
                u = network(coords)
                
            # First derivatives - ensure we have valid gradients
            grad_u = tape1.gradient(u, coords)
            
            if grad_u is None:
                # Fallback: return zeros if gradients can't be computed
                return tf.zeros_like(u)
            
            u_x = grad_u[:, 0:1]  # ∂u/∂x
            u_y = grad_u[:, 1:2]  # ∂u/∂y
            
        # Second derivatives
        u_xx = tape2.gradient(u_x, coords)
        u_yy = tape2.gradient(u_y, coords)
        
        # Clean up tapes
        del tape1, tape2
        
        # Handle None gradients
        if u_xx is None or u_yy is None:
            print("Warning: Second derivatives could not be computed, returning zeros")
            return tf.zeros_like(u)
        
        # Extract the relevant components
        u_xx_component = u_xx[:, 0:1]  # ∂²u/∂x²
        u_yy_component = u_yy[:, 1:2]  # ∂²u/∂y²
        
        # Laplacian = ∂²u/∂x² + ∂²u/∂y²
        laplacian = u_xx_component + u_yy_component
        
        return laplacian
    
    def prepare_training_data(self, u_pattern, v_pattern):
        """Prepare training data with proper tensor handling"""
        
        # Ensure patterns are float32
        u_pattern = u_pattern.astype(np.float32)
        v_pattern = v_pattern.astype(np.float32)
        
        # Reshape patterns to match coordinate grid
        u_flat = u_pattern.flatten().reshape(-1, 1)
        v_flat = v_pattern.flatten().reshape(-1, 1)
        
        # Create interior coordinates (exclude boundary)
        boundary_width = 3
        interior_mask = np.ones_like(self.X, dtype=bool)
        interior_mask[:boundary_width, :] = False
        interior_mask[-boundary_width:, :] = False  
        interior_mask[:, :boundary_width] = False
        interior_mask[:, -boundary_width:] = False
        
        interior_coords = np.stack([
            self.X[interior_mask], 
            self.Y[interior_mask]
        ], axis=1).astype(np.float32)
        
        # Convert to TensorFlow tensors with explicit dtype
        coords_tensor = tf.constant(self.coords, dtype=tf.float32)
        u_data_tensor = tf.constant(u_flat, dtype=tf.float32)
        v_data_tensor = tf.constant(v_flat, dtype=tf.float32)
        interior_coords_tensor = tf.constant(interior_coords, dtype=tf.float32)
        
        return coords_tensor, u_data_tensor, v_data_tensor, interior_coords_tensor


class RBFLayer(tf.keras.layers.Layer):
    """Fixed RBF layer with proper gradient handling"""
    
    def __init__(self, num_units, spatial_bounds=(0, 10), **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.num_units = num_units
        self.spatial_bounds = spatial_bounds
        
    def build(self, input_shape):
        # RBF centers (learnable)
        self.centers = self.add_weight(
            name='centers',
            shape=(self.num_units, 2),
            initializer='uniform',
            trainable=True,
            dtype=tf.float32
        )
        
        # RBF widths/variances (learnable, ensure positive)
        self.log_widths = self.add_weight(
            name='log_widths',
            shape=(self.num_units,),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )
        
        # RBF weights (learnable)  
        self.weights_rbf = self.add_weight(
            name='weights_rbf',
            shape=(self.num_units,),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        
        # Initialize centers randomly within spatial bounds
        x_min, x_max = self.spatial_bounds
        center_init = tf.random.uniform(
            (self.num_units, 2), 
            minval=x_min, 
            maxval=x_max,
            dtype=tf.float32
        )
        self.centers.assign(center_init)
        
        # Initialize log_widths (widths will be exp(log_widths))
        initial_width = (x_max - x_min) / np.sqrt(self.num_units)
        self.log_widths.assign(tf.ones(self.num_units, dtype=tf.float32) * tf.math.log(initial_width))
        
        super(RBFLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Ensure inputs are float32
        inputs = tf.cast(inputs, tf.float32)
        
        # Compute distances to all centers
        expanded_inputs = tf.expand_dims(inputs, 1)  # (batch, 1, 2)
        expanded_centers = tf.expand_dims(self.centers, 0)  # (1, num_units, 2)
        
        # Euclidean distance squared
        distances_sq = tf.reduce_sum(
            tf.square(expanded_inputs - expanded_centers), 
            axis=2
        )  # (batch, num_units)
        
        # Apply RBF kernel (Gaussian) with positive widths
        widths = tf.exp(self.log_widths)  # Ensure positive widths
        widths_sq = tf.square(widths)
        rbf_outputs = tf.exp(-distances_sq / (2 * widths_sq + 1e-8))  # Add small epsilon
        
        # Apply weights
        weighted_outputs = rbf_outputs * self.weights_rbf
        
        return weighted_outputs    


def load_gm_training_data(csv_file, base_folder=None, max_samples=None, target_grid_size=100):
    """
    Load G-M training data with proper image handling
    """
    print(f"Loading G-M training data from {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} total experiments in CSV")
    
    # Filter for successful classic G-M experiments
    if 'Model_Type' in df.columns:
        classic_mask = df['Model_Type'] == 'Classic'
        df_classic = df[classic_mask].copy()
        print(f"Found {len(df_classic)} Classic G-M experiments")
    else:
        df_classic = df.copy()
    
    if 'Status' in df.columns:
        success_mask = df_classic['Status'] == 'success'
        df_success = df_classic[success_mask].copy()
        print(f"Found {len(df_success)} successful experiments")
    else:
        df_success = df_classic.copy()
    
    if len(df_success) == 0:
        raise ValueError("No successful classic G-M experiments found!")
    
    # Limit samples if requested
    if max_samples and len(df_success) > max_samples:
        df_success = df_success.sample(max_samples, random_state=42)
        print(f"Randomly selected {max_samples} samples for training")
    
    experiments = []
    
    for idx, row in df_success.iterrows():
        try:
            # Get experiment directory
            if 'Output_Directory' in row:
                exp_dir = row['Output_Directory']
            elif 'Directory' in row:
                exp_dir = row['Directory']
            else:
                print(f"Warning: No directory column found for row {idx}")
                continue
            
            # Build full path
            if base_folder and not os.path.isabs(exp_dir):
                exp_dir = os.path.join(base_folder, exp_dir)
            
            # Look for pattern image
            image_path = os.path.join(exp_dir, 'pattern_simulation.png')
            if not os.path.exists(image_path):
                print(f"Warning: Pattern image not found: {image_path}")
                continue
            
            # Load and process image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            print(f"Debug - Original image shape: {img_array.shape}")
            
            # Handle color vs grayscale
            if len(img_array.shape) == 3:
                img_gray = img_array[:, :, 0]  # Use red channel
            else:
                img_gray = img_array
            
            print(f"Debug - Grayscale image shape: {img_gray.shape}")
            
            # Check if side-by-side (width >> height)
            height, width = img_gray.shape
            
            if width > height * 1.8:  # Definitely side-by-side
                print(f"Debug - Detected side-by-side layout: {width} x {height}")
                
                # Split exactly in half
                mid_point = width // 2
                u_pattern = img_gray[:, :mid_point]
                v_pattern = img_gray[:, mid_point:]
                
                print(f"Debug - After split: u={u_pattern.shape}, v={v_pattern.shape}")
                
                # Make square by cropping to the smaller dimension
                min_dim = min(u_pattern.shape[0], u_pattern.shape[1])
                
                # Crop from center to make square
                if u_pattern.shape[0] > u_pattern.shape[1]:
                    # Height > width, crop height
                    start_row = (u_pattern.shape[0] - min_dim) // 2
                    u_pattern = u_pattern[start_row:start_row + min_dim, :]
                    v_pattern = v_pattern[start_row:start_row + min_dim, :]
                elif u_pattern.shape[1] > u_pattern.shape[0]:
                    # Width > height, crop width  
                    start_col = (u_pattern.shape[1] - min_dim) // 2
                    u_pattern = u_pattern[:, start_col:start_col + min_dim]
                    v_pattern = v_pattern[:, start_col:start_col + min_dim]
                
                print(f"Debug - After making square: u={u_pattern.shape}, v={v_pattern.shape}")
                
            else:
                # Single pattern or already square
                print(f"Debug - Single pattern or square: {width} x {height}")
                u_pattern = img_gray
                v_pattern = img_gray  # Use same pattern for both (not ideal but works)
            
            # Resize to target size for efficiency
            if target_grid_size and target_grid_size != u_pattern.shape[0]:
                print(f"Debug - Resizing from {u_pattern.shape[0]} to {target_grid_size}")
                
                # Resize using PIL for better quality
                u_img = Image.fromarray((u_pattern * 255).astype(np.uint8))
                v_img = Image.fromarray((v_pattern * 255).astype(np.uint8))
    
                # Replace the resize lines with:
                u_img_resized = u_img.resize((target_grid_size, target_grid_size), Image.Resampling.LANCZOS)
                v_img_resized = v_img.resize((target_grid_size, target_grid_size), Image.Resampling.LANCZOS)
                
                u_pattern = np.array(u_img_resized).astype(np.float32) / 255.0
                v_pattern = np.array(v_img_resized).astype(np.float32) / 255.0
            
            # Normalize to [0, 1]
            u_pattern = (u_pattern - u_pattern.min()) / (u_pattern.max() - u_pattern.min() + 1e-8)
            v_pattern = (v_pattern - v_pattern.min()) / (v_pattern.max() - v_pattern.min() + 1e-8)
            
            print(f"Debug - Final patterns: u={u_pattern.shape}, v={v_pattern.shape}")
            
            # Extract parameters
            true_params = {}
            param_cols = ['a', 'b', 'c', 'd', 'Du', 'Dv']
            
            for col in param_cols:
                if col in row and pd.notna(row[col]):
                    true_params[col] = float(row[col])
                else:
                    print(f"Warning: Parameter {col} not found for experiment {idx}")
                    true_params[col] = 0.0
            
            experiments.append({
                'u_pattern': u_pattern,
                'v_pattern': v_pattern,
                'true_parameters': true_params,
                'exp_dir': exp_dir,
                'row_idx': idx,
                'grid_size': u_pattern.shape[0]
            })
            
            print(f"✅ Successfully loaded experiment {idx}")
            break  # Remove this to load all experiments
            
        except Exception as e:
            print(f"❌ Error loading experiment {idx}: {e}")
            continue
    
    print(f"Successfully loaded {len(experiments)} experiments")
    
    if len(experiments) == 0:
        raise ValueError("No experiments could be loaded!")
    
    return experiments


def train_rbf_pinn_on_gm_data(csv_file, base_folder=None, model_type='classic',
                              experiment_idx=0, num_rbf_units=80, epochs=50000,
                              save_results=True):
    """
    Train RBF-PINN on a specific G-M experiment
    
    Args:
        csv_file: Path to CSV with experiment metadata
        base_folder: Base folder containing experiment directories
        model_type: 'classic' or 'saturated'
        experiment_idx: Which experiment to use for training (0-based index)
        num_rbf_units: Number of RBF kernels
        epochs: Training epochs
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary with training results and evaluation metrics
    """
    
    print("="*60)
    print(f"TRAINING RBF-PINN FOR GIERER-MEINHARDT PARAMETER INFERENCE")
    print("="*60)
    
    # Load training data
    experiments = load_gm_training_data(csv_file, base_folder, max_samples=None)
    
    if experiment_idx >= len(experiments):
        raise ValueError(f"Experiment index {experiment_idx} out of range. "
                        f"Available: 0-{len(experiments)-1}")
    
    # Select experiment
    exp_data = experiments[experiment_idx]
    u_pattern = exp_data['u_pattern']
    v_pattern = exp_data['v_pattern']
    true_params = exp_data['true_parameters']
    
    print(f"Selected experiment {experiment_idx}:")
    print(f"  Pattern shape: {u_pattern.shape}")
    print(f"  True parameters: {true_params}")
    print(f"  Model type: {model_type}")
    
    # Create RBF-PINN model
    model = GiererMeinhardtRBFPINN(
        num_rbf_units=num_rbf_units,
        model_type=model_type,
        spatial_domain=(0, 10),
        grid_size=u_pattern.shape[0]
    )
    
    # Set reasonable initial parameter guesses
    initial_params = {
        'a': 0.1,
        'b': 1.0, 
        'c': 0.9,
        'd': 0.9,
        'Du': 0.05,
        'Dv': 1.0
    }
    
    if model_type == 'saturated':
        initial_params['saturation'] = 0.01
    
    model.set_initial_parameters(initial_params)
    print(f"Initial parameter guess: {initial_params}")
    
    # Define training schedule
    def data_weight_schedule(epoch):
        return 1.0  # Constant data weight
    
    def pde_weight_schedule(epoch):
        phase1_epochs = epochs // 5  # 20% for data-only training
        if epoch < phase1_epochs:
            return 0.0
        else:
            # Gradually increase PDE weight
            progress = (epoch - phase1_epochs) / (epochs - phase1_epochs)
            return min(2.0, progress * 3.0)  # Max PDE weight of 2.0
    
    # Train the model
    print(f"\nStarting training for {epochs} epochs...")
    start_time = datetime.datetime.now()
    
    model.train(
        u_pattern=u_pattern,
        v_pattern=v_pattern,
        epochs=epochs,
        data_weight_schedule=data_weight_schedule,
        pde_weight_schedule=pde_weight_schedule,
        print_every=epochs//20,  # Print 20 times during training
        phase1_epochs=epochs//5
    )
    
    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    
    # Get final results
    predicted_params = model.get_current_parameters()
    u_pred, v_pred = model.get_predictions()
    
    print(f"\nTraining completed in {training_time}")
    print("\nFINAL RESULTS:")
    print("-" * 50)
    
    # Calculate parameter errors
    param_errors = {}
    for name in model.param_names:
        true_val = true_params[name]
        pred_val = predicted_params[name]
        
        if abs(true_val) > 1e-8:
            rel_error = abs(pred_val - true_val) / abs(true_val) * 100
        else:
            rel_error = abs(pred_val) * 100
        
        param_errors[name] = rel_error
        
        print(f"{name:12s}: True={true_val:8.4f}, "
              f"Predicted={pred_val:8.4f}, "
              f"Error={rel_error:6.2f}%")
    
    overall_error = np.mean(list(param_errors.values()))
    print(f"\nOverall average error: {overall_error:.2f}%")
    
    # Evaluate pattern similarity using RAPS (Radially Averaged Power Spectrum)
    def compute_raps(pattern):
        """Compute radially averaged power spectrum"""
        fft = np.fft.fft2(pattern)
        power_spectrum = np.abs(fft)**2
        
        # Get center coordinates
        cy, cx = np.array(pattern.shape) // 2
        
        # Create radius array
        y, x = np.ogrid[:pattern.shape[0], :pattern.shape[1]]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Compute radial average
        r_int = r.astype(int)
        radial_prof = np.bincount(r_int.ravel(), power_spectrum.ravel())
        nr = np.bincount(r_int.ravel())
        radial_prof = radial_prof / (nr + 1e-8)  # Avoid division by zero
        
        return radial_prof
    
    # Compute pattern similarity
    try:
        raps_true_u = compute_raps(u_pattern)
        raps_pred_u = compute_raps(u_pred)
        raps_true_v = compute_raps(v_pattern)
        raps_pred_v = compute_raps(v_pred)
        
        # RAPS difference (MSE between RAPS profiles)
        min_len = min(len(raps_true_u), len(raps_pred_u))
        raps_diff_u = np.mean((raps_true_u[:min_len] - raps_pred_u[:min_len])**2)
        
        min_len = min(len(raps_true_v), len(raps_pred_v))
        raps_diff_v = np.mean((raps_true_v[:min_len] - raps_pred_v[:min_len])**2)
        
        print(f"\nPattern similarity (RAPS difference):")
        print(f"  U pattern: {raps_diff_u:.6f}")
        print(f"  V pattern: {raps_diff_v:.6f}")
        
    except Exception as e:
        print(f"Could not compute RAPS similarity: {e}")
        raps_diff_u = raps_diff_v = None
    
    # Create results summary
    results = {
        'experiment_idx': experiment_idx,
        'model_type': model_type,
        'num_rbf_units': num_rbf_units,
        'epochs': epochs,
        'training_time_seconds': training_time.total_seconds(),
        'true_parameters': true_params,
        'predicted_parameters': predicted_params,
        'parameter_errors': param_errors,
        'overall_error_percent': overall_error,
        'raps_difference_u': raps_diff_u,
        'raps_difference_v': raps_diff_v,
        'training_history': model.history,
        'pattern_shapes': {
            'u_true': u_pattern.shape,
            'v_true': v_pattern.shape,
            'u_pred': u_pred.shape,
            'v_pred': v_pred.shape
        }
    }
    
    # Save results if requested
    if save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"rbf_pinn_gm_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "trained_model")
        model.save_model(model_path)
        
        # Save results summary
        results_path = os.path.join(save_dir, "results_summary.json")
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, np.integer):
                    json_results[key] = int(value)
                elif isinstance(value, np.floating):
                    json_results[key] = float(value)
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=4, default=str)
        
        # Create comprehensive plots
        plot_comprehensive_results(
            model, u_pattern, v_pattern, u_pred, v_pred, 
            true_params, predicted_params, save_dir
        )
        
        print(f"\nResults saved to directory: {save_dir}")
        print(f"  Model: {model_path}")
        print(f"  Summary: {results_path}")
        print(f"  Plots: {save_dir}/comprehensive_results.png")
    
    return results


def plot_comprehensive_results(model, u_true, v_true, u_pred, v_pred, 
                             true_params, pred_params, save_dir):
    """Create comprehensive result visualization"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: True vs Predicted patterns
    im1 = axes[0,0].imshow(u_true, cmap='viridis', origin='lower')
    axes[0,0].set_title('True u (Activator)')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(u_pred, cmap='viridis', origin='lower')
    axes[0,1].set_title('Predicted u (Activator)')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[0,2].imshow(v_true, cmap='plasma', origin='lower')
    axes[0,2].set_title('True v (Inhibitor)')
    plt.colorbar(im3, ax=axes[0,2])
    
    im4 = axes[0,3].imshow(v_pred, cmap='plasma', origin='lower')
    axes[0,3].set_title('Predicted v (Inhibitor)')
    plt.colorbar(im4, ax=axes[0,3])
    
    # Row 2: Training curves
    epochs = range(len(model.history['total_loss']))
    
    axes[1,0].semilogy(epochs, model.history['total_loss'], 'b-', linewidth=2)
    axes[1,0].set_title('Total Loss')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Loss (log scale)')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].semilogy(epochs, model.history['data_loss_u'], 'r-', label='u', linewidth=2)
    axes[1,1].semilogy(epochs, model.history['data_loss_v'], 'g-', label='v', linewidth=2)
    axes[1,1].set_title('Data Fitting Losses')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Loss (log scale)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].semilogy(epochs, model.history['pde_loss'], 'm-', linewidth=2)
    axes[1,2].set_title('PDE Residual Loss')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('Loss (log scale)')
    axes[1,2].grid(True, alpha=0.3)
    
    # Parameter evolution
    colors = plt.cm.tab10(np.linspace(0, 1, len(model.param_names)))
    for name, color in zip(model.param_names, colors):
        axes[1,3].plot(epochs, model.history['parameters'][name], 
                      color=color, label=name, linewidth=2)
    axes[1,3].set_title('Parameter Evolution')
    axes[1,3].set_xlabel('Epoch')
    axes[1,3].set_ylabel('Parameter Value')
    axes[1,3].legend()
    axes[1,3].grid(True, alpha=0.3)
    
    # Row 3: Parameter comparison and error analysis
    param_names = list(true_params.keys())
    true_vals = [true_params[name] for name in param_names]
    pred_vals = [pred_params[name] for name in param_names]
    
    x_pos = np.arange(len(param_names))
    width = 0.35
    
    axes[2,0].bar(x_pos - width/2, true_vals, width, label='True', alpha=0.8)
    axes[2,0].bar(x_pos + width/2, pred_vals, width, label='Predicted', alpha=0.8)
    axes[2,0].set_xlabel('Parameters')
    axes[2,0].set_ylabel('Value')
    axes[2,0].set_title('Parameter Comparison')
    axes[2,0].set_xticks(x_pos)
    axes[2,0].set_xticklabels(param_names)
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    
    # Error percentages
    errors = []
    for name in param_names:
        true_val = true_params[name]
        pred_val = pred_params[name]
        if abs(true_val) > 1e-8:
            error = abs(pred_val - true_val) / abs(true_val) * 100
        else:
            error = abs(pred_val) * 100
        errors.append(error)
    
    axes[2,1].bar(param_names, errors, color='orange', alpha=0.7)
    axes[2,1].set_xlabel('Parameters')
    axes[2,1].set_ylabel('Relative Error (%)')
    axes[2,1].set_title('Parameter Errors')
    axes[2,1].tick_params(axis='x', rotation=45)
    axes[2,1].grid(True, alpha=0.3)
    
    # Pattern difference maps
    u_diff = np.abs(u_true - u_pred)
    v_diff = np.abs(v_true - v_pred)
    
    im5 = axes[2,2].imshow(u_diff, cmap='Reds', origin='lower')
    axes[2,2].set_title('|u_true - u_pred|')
    plt.colorbar(im5, ax=axes[2,2])
    
    im6 = axes[2,3].imshow(v_diff, cmap='Reds', origin='lower')
    axes[2,3].set_title('|v_true - v_pred|')
    plt.colorbar(im6, ax=axes[2,3])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, "comprehensive_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save training progress plot
    progress_path = os.path.join(save_dir, "training_progress.png")
    model.plot_training_progress(save_path=progress_path)
    plt.close()


def evaluate_multiple_experiments(csv_file, base_folder=None, model_type='classic',
                                num_experiments=5, num_rbf_units=80, epochs=30000):
    """
    Evaluate RBF-PINN performance on multiple experiments
    
    Args:
        csv_file: Path to experiment CSV
        base_folder: Base folder for experiments
        model_type: 'classic' or 'saturated'
        num_experiments: Number of experiments to test
        num_rbf_units: Number of RBF units
        epochs: Training epochs per experiment
        
    Returns:
        Dictionary with aggregated results
    """
    
    print("="*60)
    print(f"EVALUATING RBF-PINN ON MULTIPLE G-M EXPERIMENTS")
    print("="*60)
    
    # Load available experiments
    experiments = load_gm_training_data(csv_file, base_folder)
    
    if num_experiments > len(experiments):
        num_experiments = len(experiments)
        print(f"Limiting to {num_experiments} available experiments")
    
    all_results = []
    
    for i in range(num_experiments):
        print(f"\n{'='*40}")
        print(f"EXPERIMENT {i+1}/{num_experiments}")
        print(f"{'='*40}")
        
        try:
            results = train_rbf_pinn_on_gm_data(
                csv_file=csv_file,
                base_folder=base_folder,
                model_type=model_type,
                experiment_idx=i,
                num_rbf_units=num_rbf_units,
                epochs=epochs,
                save_results=False  # Don't save individual results
            )
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error in experiment {i+1}: {e}")
            continue
    
    if not all_results:
        raise ValueError("No experiments completed successfully!")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS ({len(all_results)} experiments)")
    print(f"{'='*60}")
    
    param_names = all_results[0]['true_parameters'].keys()
    
    # Calculate statistics for each parameter
    param_stats = {}
    for param in param_names:
        errors = [r['parameter_errors'][param] for r in all_results if param in r['parameter_errors']]
        param_stats[param] = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'median_error': np.median(errors)
        }
        
        print(f"{param:12s}: {param_stats[param]['mean_error']:6.2f}% ± {param_stats[param]['std_error']:5.2f}% "
              f"(range: {param_stats[param]['min_error']:5.1f}% - {param_stats[param]['max_error']:5.1f}%)")
    
    overall_errors = [r['overall_error_percent'] for r in all_results]
    overall_stats = {
        'mean_error': np.mean(overall_errors),
        'std_error': np.std(overall_errors),
        'min_error': np.min(overall_errors),
        'max_error': np.max(overall_errors),
        'median_error': np.median(overall_errors)
    }
    
    print(f"\nOverall performance:")
    print(f"  Mean error: {overall_stats['mean_error']:.2f}% ± {overall_stats['std_error']:.2f}%")
    print(f"  Range: {overall_stats['min_error']:.1f}% - {overall_stats['max_error']:.1f}%")
    print(f"  Median: {overall_stats['median_error']:.2f}%")
    
    # Training time statistics
    training_times = [r['training_time_seconds'] for r in all_results]
    print(f"\nTraining time:")
    print(f"  Mean: {np.mean(training_times):.1f} seconds")
    print(f"  Range: {np.min(training_times):.1f} - {np.max(training_times):.1f} seconds")
    
    # Save aggregated results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"rbf_pinn_evaluation_summary_{timestamp}.json"
    
    summary = {
        'model_type': model_type,
        'num_experiments': len(all_results),
        'num_rbf_units': num_rbf_units,
        'epochs': epochs,
        'parameter_statistics': param_stats,
        'overall_statistics': overall_stats,
        'individual_results': all_results,
        'timestamp': timestamp
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    
    print(f"\nEvaluation summary saved to: {summary_file}")
    
    return summary


# Example usage and main execution
if __name__ == "__main__":
    print("RBF-PINN for Gierer-Meinhardt Parameter Inference")
    print("=" * 60)
    
    # Configuration - update these paths to match your setup
    CSV_FILE = "GM_training_data_targeted_20250528_211446/batch_experiments_summary.csv"
    csv_file = CSV_FILE
    BASE_FOLDER = "./"  # Adjust as needed
    base_folder = BASE_FOLDER
    # Check if files exist
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found: {CSV_FILE}")
        print("Please update CSV_FILE path to point to your experiment summary CSV")
        exit(1)
    # Quick diagnostic - add this to your script:
    experiments = load_gm_training_data(csv_file, base_folder, max_samples=None)
    if experiments:
        sample_u = experiments[0]['u_pattern']
        sample_v = experiments[0]['v_pattern']
        print(f"Pattern shapes: u={sample_u.shape}, v={sample_v.shape}")
    
    # Use the actual pattern size
    grid_size = sample_u.shape[0]
    print(f"Setting RBF-PINN grid_size to: {grid_size}")
    # Example 1: Train on a single experiment
    print("\nExample 1: Training RBF-PINN on single experiment")
    try:
        results = train_rbf_pinn_on_gm_data(
            csv_file=CSV_FILE,
            base_folder=BASE_FOLDER,
            model_type='classic',
            experiment_idx=0,
            num_rbf_units=80,
            epochs=20000,
            save_results=True
        )
        
        print(f"Single experiment completed with {results['overall_error_percent']:.2f}% average error")
        
    except Exception as e:
        print(f"Error in single experiment: {e}")
    
    # Example 2: Evaluate on multiple experiments (uncomment to run)
    """
    print("\nExample 2: Evaluating on multiple experiments")
    try:
        summary = evaluate_multiple_experiments(
            csv_file=CSV_FILE,
            base_folder=BASE_FOLDER,
            model_type='classic',
            num_experiments=3,
            num_rbf_units=80,
            epochs=15000
        )
        
        print(f"Multi-experiment evaluation completed")
        print(f"Overall performance: {summary['overall_statistics']['mean_error']:.2f}% ± {summary['overall_statistics']['std_error']:.2f}%")
        
    except Exception as e:
        print(f"Error in multi-experiment evaluation: {e}")
    """
    
    print("\nRBF-PINN evaluation completed!")