import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

class SimpleGMRBFPINN:
    """Simple RBF-PINN to predict G-M parameters from pattern images"""
    
    def __init__(self, num_rbf_units=60, model_type='classic'):
        self.num_rbf_units = num_rbf_units
        self.model_type = model_type
        
        # Parameter names for different G-M models
        if model_type == 'classic':
            self.param_names = ['a', 'b', 'c', 'd', 'Du', 'Dv']
        elif model_type == 'saturated':
            self.param_names = ['a', 'b', 'c', 'd', 'p', 'Du', 'Dv']
        
        self.num_params = len(self.param_names)
        
        # Build the network
        self.model = self._build_network()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
    def _build_network(self):
        """Build RBF network that takes image and outputs parameters"""
        
        # Input: flattened image
        image_input = keras.layers.Input(shape=(None,), name='image_input')  # Flattened image
        
        # Reshape to get spatial coordinates implicitly
        # For now, we'll use a simple dense approach
        
        # Dense layers to process the image
        x = keras.layers.Dense(256, activation='relu')(image_input)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        
        # RBF-like layer (approximated with dense + gaussian activation)
        rbf_layer = keras.layers.Dense(self.num_rbf_units, activation='relu')(x)
        
        # Output layer - predict parameters
        param_output = keras.layers.Dense(self.num_params, activation='linear', name='parameters')(rbf_layer)
        
        model = keras.Model(inputs=image_input, outputs=param_output)
        return model
    
    def gm_physics_loss(self, y_true, y_pred):
        """Physics-informed loss based on G-M equations"""
        # For now, just use MSE - we'll add physics constraints later
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def compile_model(self):
        """Compile the model"""
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.gm_physics_loss,
            metrics=['mae']
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the model"""
        
        self.compile_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_parameters(self, X):
        """Predict parameters from images"""
        predictions = self.model.predict(X)
        
        # Convert back to named parameters
        result = []
        for pred in predictions:
            param_dict = {name: val for name, val in zip(self.param_names, pred)}
            result.append(param_dict)
        
        return result

def load_data_from_folder(base_folder, csv_file):
    """
    Load images and parameters from your folder structure
    
    Args:
        base_folder: base path where experiment folders are located
        csv_file: path to CSV file with parameters
    
    Returns:
        X: array of flattened images
        y: array of parameter values
        metadata: dataframe with all info
    """
    
    print(f"Loading data from CSV: {csv_file}")
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")
    print(f"CSV columns: {list(df.columns)}")
    
    # Filter for classic GM model only (ignore saturated for now)
    classic_mask = df['Model_Type'] == 'Classic'
    df_classic = df[classic_mask].copy()
    print(f"Found {len(df_classic)} Classic G-M experiments")
    
    # Filter for Turing-capable patterns only
    turing_mask = df_classic['Turing_Capable'] == 'YES'
    df_turing = df_classic[turing_mask].copy()
    print(f"Found {len(df_turing)} Turing-capable Classic G-M patterns")
    
    if len(df_turing) == 0:
        print("Warning: No Turing-capable classic patterns found!")
        print("Available Model_Type values:", df['Model_Type'].unique())
        print("Available Turing_Capable values:", df['Turing_Capable'].unique())
        # Use all classic patterns if no Turing-capable ones found
        df_turing = df_classic.copy()
        print(f"Using all {len(df_turing)} classic patterns instead")
    
    # Load images and match with parameters
    images = []
    parameters = []
    valid_rows = []
    
    for idx, row in df_turing.iterrows():
        # Get the output directory for this experiment
        output_dir = row['Output_Directory']
        
        # Construct full path to the image
        if os.path.isabs(output_dir):
            # Absolute path
            image_path = os.path.join(output_dir, 'pattern_simulation.png')
        else:
            # Relative path - combine with base folder
            image_path = os.path.join(base_folder, output_dir, 'pattern_simulation.png')
        
        print(f"Looking for image: {image_path}")
        
        if os.path.exists(image_path):
            try:
                # Load image
                img = Image.open(image_path)
                if img.mode != 'L':  # Convert to grayscale if needed
                    img = img.convert('L')
                
                # Convert to numpy array and flatten
                img_array = np.array(img)
                img_flat = img_array.flatten()
                
                images.append(img_flat)
                
                # Extract parameters for classic G-M model
                param_row = []
                param_columns = ['a', 'b', 'c', 'd', 'Du', 'Dv']
                
                for col in param_columns:
                    if col in row and pd.notna(row[col]):
                        param_row.append(float(row[col]))
                    else:
                        print(f"Warning: Column {col} not found or is NaN in row {idx}")
                        param_row.append(0.0)  # Default value
                
                parameters.append(param_row)
                valid_rows.append(row)
                print(f"Successfully loaded experiment {row['Experiment_Number']}")
                
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        else:
            print(f"Image not found: {image_path}")
    
    if len(images) == 0:
        raise ValueError("No images were successfully loaded! Check your paths and file structure.")
    
    X = np.array(images, dtype=np.float32)
    y = np.array(parameters, dtype=np.float32)
    
    # Normalize images to [0, 1]
    X = X / 255.0
    
    print(f"\nSuccessfully loaded {len(X)} image-parameter pairs")
    print(f"Image shape: {X.shape}")
    print(f"Parameter shape: {y.shape}")
    print(f"Parameter ranges:")
    param_names = ['a', 'b', 'c', 'd', 'Du', 'Dv']
    for i, name in enumerate(param_names):
        print(f"  {name}: {y[:, i].min():.4f} to {y[:, i].max():.4f}")
    
    return X, y, pd.DataFrame(valid_rows)

def train_gm_predictor(base_folder, csv_file, model_type='classic', test_split=0.2):
    """
    Complete training pipeline
    
    Args:
        base_folder: base path where experiment output directories are located
        csv_file: path to CSV with parameters
        model_type: 'classic' or 'saturated' (using 'classic' for now)
        test_split: fraction of data for testing
    """
    
    print("="*60)
    print("TRAINING CLASSIC G-M PARAMETER PREDICTOR")
    print("="*60)
    
    # Load data
    X, y, metadata = load_data_from_folder(base_folder, csv_file)
    
    if len(X) < 10:
        print(f"Warning: Only {len(X)} samples found. Need more data for reliable training.")
        print("Consider including more experiments or relaxing filtering criteria.")
    
    # Split data
    n_samples = len(X)
    n_test = max(1, int(n_samples * test_split))  # At least 1 test sample
    
    # Random shuffle
    np.random.seed(42)  # For reproducible results
    indices = np.random.permutation(n_samples)
    
    X_train = X[indices[n_test:]]
    y_train = y[indices[n_test:]]
    X_test = X[indices[:n_test]]
    y_test = y[indices[:n_test]]
    
    print(f"\nData split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and train model
    model = SimpleGMRBFPINN(num_rbf_units=60, model_type=model_type)
    
    print(f"\nModel architecture:")
    print(f"Input size: {X.shape[1]} (flattened image)")
    print(f"Output size: {model.num_params} parameters: {model.param_names}")
    
    print("\nStarting training...")
    history = model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=200,
        batch_size=min(16, len(X_train))  # Adjust batch size for small datasets
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    predictions = model.predict_parameters(X_test)
    
    # Calculate errors
    param_names = model.param_names
    errors = {name: [] for name in param_names}
    
    print(f"\nDetailed results for {len(predictions)} test samples:")
    print("-" * 80)
    print(f"{'Sample':<8} {'Parameter':<12} {'True':<10} {'Predicted':<10} {'Error %':<10}")
    print("-" * 80)
    
    for i, pred_dict in enumerate(predictions):
        for j, param_name in enumerate(param_names):
            true_val = y_test[i, j]
            pred_val = pred_dict[param_name]
            
            if abs(true_val) > 1e-8:  # Avoid division by zero
                rel_error = abs(pred_val - true_val) / abs(true_val) * 100
                errors[param_name].append(rel_error)
                
                print(f"{i+1:<8} {param_name:<12} {true_val:<10.4f} {pred_val:<10.4f} {rel_error:<10.1f}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY RESULTS:")
    print("="*60)
    
    for param_name in param_names:
        if len(errors[param_name]) > 0:
            mean_error = np.mean(errors[param_name])
            std_error = np.std(errors[param_name])
            min_error = np.min(errors[param_name])
            max_error = np.max(errors[param_name])
            print(f"{param_name:4s}: {mean_error:6.2f}% ± {std_error:5.2f}% (min: {min_error:5.1f}%, max: {max_error:5.1f}%)")
    
    overall_error = np.mean([np.mean(errors[name]) for name in param_names if len(errors[name]) > 0])
    print(f"\nOverall average error: {overall_error:6.2f}%")
    
    # Determine performance level
    if overall_error < 1.0:
        performance = "EXCELLENT"
    elif overall_error < 5.0:
        performance = "GOOD"
    elif overall_error < 10.0:
        performance = "FAIR"
    else:
        performance = "NEEDS IMPROVEMENT"
    
    print(f"Performance level: {performance}")
    
    # Plot results
    plot_training_results(history, predictions, y_test, param_names)
    
    return model, history, predictions, y_test, param_names

def plot_training_results(history, predictions, y_test, param_names):
    """Plot training history and prediction results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history
    axes[0,0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training History')
    axes[0,0].legend()
    axes[0,0].set_yscale('log')
    
    # Parameter predictions vs true values
    n_params = len(param_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))
    
    for i, param_name in enumerate(param_names):
        true_vals = y_test[:, i]
        pred_vals = [pred[param_name] for pred in predictions]
        
        axes[0,1].scatter(true_vals, pred_vals, alpha=0.6, 
                         color=colors[i], label=param_name, s=20)
    
    # Perfect prediction line
    all_true = y_test.flatten()
    all_pred = np.array([[pred[name] for name in param_names] for pred in predictions]).flatten()
    min_val, max_val = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    axes[0,1].set_xlabel('True Parameter Value')
    axes[0,1].set_ylabel('Predicted Parameter Value')
    axes[0,1].set_title('Parameter Predictions')
    axes[0,1].legend()
    
    # Error distribution
    all_errors = []
    for i, param_name in enumerate(param_names):
        true_vals = y_test[:, i]
        pred_vals = [pred[param_name] for pred in predictions]
        
        for j in range(len(true_vals)):
            if abs(true_vals[j]) > 1e-8:
                rel_error = abs(pred_vals[j] - true_vals[j]) / abs(true_vals[j]) * 100
                all_errors.append(rel_error)
    
    axes[0,2].hist(all_errors, bins=20, alpha=0.7, edgecolor='black')
    axes[0,2].set_xlabel('Relative Error (%)')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].set_title('Error Distribution')
    axes[0,2].axvline(np.mean(all_errors), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(all_errors):.1f}%')
    axes[0,2].legend()
    
    # Individual parameter errors
    param_errors = {name: [] for name in param_names}
    for i, pred_dict in enumerate(predictions):
        for j, param_name in enumerate(param_names):
            true_val = y_test[i, j]
            pred_val = pred_dict[param_name]
            
            if abs(true_val) > 1e-8:
                rel_error = abs(pred_val - true_val) / abs(true_val) * 100
                param_errors[param_name].append(rel_error)
    
    # Box plot of errors by parameter
    error_data = [param_errors[name] for name in param_names if len(param_errors[name]) > 0]
    valid_names = [name for name in param_names if len(param_errors[name]) > 0]
    
    if error_data:
        axes[1,0].boxplot(error_data, labels=valid_names)
        axes[1,0].set_xlabel('Parameter')
        axes[1,0].set_ylabel('Relative Error (%)')
        axes[1,0].set_title('Error by Parameter')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Sample predictions table
    axes[1,1].axis('off')
    n_show = min(5, len(predictions))
    table_data = []
    
    for i in range(n_show):
        row = [f"Sample {i+1}"]
        for param_name in param_names:
            true_val = y_test[i, param_names.index(param_name)]
            pred_val = predictions[i][param_name]
            row.append(f"{true_val:.3f}")
            row.append(f"{pred_val:.3f}")
        table_data.append(row)
    
    # Create column headers
    headers = ["Sample"]
    for param_name in param_names:
        headers.extend([f"{param_name}_true", f"{param_name}_pred"])
    
    table = axes[1,1].table(cellText=table_data, colLabels=headers, 
                           cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    axes[1,1].set_title('Sample Predictions')
    
    # Summary statistics
    axes[1,2].axis('off')
    summary_text = "Summary Statistics:\n\n"
    
    for param_name in param_names:
        if len(param_errors[param_name]) > 0:
            mean_err = np.mean(param_errors[param_name])
            std_err = np.std(param_errors[param_name])
            summary_text += f"{param_name}: {mean_err:.2f}% ± {std_err:.2f}%\n"
    
    overall_error = np.mean(all_errors)
    summary_text += f"\nOverall Error: {overall_error:.2f}%"
    summary_text += f"\nTotal Samples: {len(predictions)}"
    summary_text += f"\nParameters: {len(param_names)}"
    
    axes[1,2].text(0.1, 0.5, summary_text, transform=axes[1,2].transAxes, 
                  fontsize=12, verticalalignment='center')
    axes[1,2].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Example usage
if __name__ == "__main__":
    # Specify your paths here
    BASE_FOLDER = "turing_experiments"  # Base folder containing experiment directories
    CSV_FILE = "turing_experiments/turing_experiments.csv"  # Your CSV file
    
    # Alternative: if your paths are different, update these
    # BASE_FOLDER = "/path/to/your/base/folder"
    # CSV_FILE = "/path/to/your/csv/file.csv"
    
    print("G-M Parameter Predictor")
    print("=" * 50)
    print(f"Base folder: {BASE_FOLDER}")
    print(f"CSV file: {CSV_FILE}")
    
    # Check if paths exist
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file {CSV_FILE} does not exist!")
        print("Please update CSV_FILE with the correct path to your parameters file")
        
        # Try to find CSV files in current directory
        csv_files = glob.glob("*.csv") + glob.glob("*/*.csv")
        if csv_files:
            print(f"Found these CSV files that might be relevant:")
            for f in csv_files[:5]:  # Show first 5
                print(f"  {f}")
        exit()
    
    if not os.path.exists(BASE_FOLDER):
        print(f"Error: Base folder {BASE_FOLDER} does not exist!")
        print("Please update BASE_FOLDER with the correct path to your experiment folders")
        exit()
    
    # Train the model
    try:
        print("\nStarting training process...")
        
        model, history, predictions, y_test, param_names = train_gm_predictor(
            BASE_FOLDER, 
            CSV_FILE, 
            model_type='classic',  # Using classic G-M as requested
            test_split=0.2
        )
        
        # Save the trained model
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'gm_classic_predictor_{timestamp}.h5'
        model.model.save(model_filename)
        print(f"\nModel saved as '{model_filename}'")
        
        # Save results summary
        results_summary = {
            'timestamp': timestamp,
            'model_type': 'classic',
            'num_training_samples': len(predictions),
            'param_names': param_names,
            'test_predictions': predictions,
            'test_true_values': y_test.tolist()
        }
        
        import json
        results_filename = f'training_results_{timestamp}.json'
        with open(results_filename, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"Results saved as '{results_filename}'")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"You can now use the trained model to predict G-M parameters from new pattern images.")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nDebugging information:")
        print("1. Check that your CSV file has the expected columns:")
        print("   Experiment_Number,Row_Number,Model_Type,a,b,c,d,Du,Dv,Rho,Saturation,Turing_Capable,Status,Output_Directory,Error")
        print("2. Check that Output_Directory paths point to folders containing 'pattern_simulation.png'")
        print("3. Make sure you have some experiments with Model_Type='Classic' and Turing_Capable='YES'")
        
        # Try to load just the CSV to see what's in it
        try:
            df = pd.read_csv(CSV_FILE)
            print(f"\nCSV file loaded successfully with {len(df)} rows")
            print(f"Columns found: {list(df.columns)}")
            print(f"Model types: {df['Model_Type'].value_counts().to_dict()}")
            print(f"Turing capable: {df['Turing_Capable'].value_counts().to_dict()}")
            
            # Show a few sample paths
            print(f"\nSample Output_Directory paths:")
            for i, path in enumerate(df['Output_Directory'].head(3)):
                full_path = os.path.join(BASE_FOLDER, path, 'pattern_simulation.png')
                exists = "✓" if os.path.exists(full_path) else "✗"
                print(f"  {exists} {full_path}")
                
        except Exception as csv_error:
            print(f"Could not load CSV file: {csv_error}")
        
        import traceback
        traceback.print_exc()