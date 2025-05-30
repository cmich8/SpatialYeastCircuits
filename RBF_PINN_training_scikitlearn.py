import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import joblib

class SimpleGMPredictor:
    """Simple G-M parameter predictor using scikit-learn"""
    
    def __init__(self, model_type='neural_network', use_pca=True, n_components=100):
        self.model_type = model_type
        self.use_pca = use_pca
        self.n_components = n_components
        
        # Parameter names for classic G-M
        self.param_names = ['a', 'b', 'c', 'd', 'Du', 'Dv']
        self.num_params = len(self.param_names)
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if use_pca else None
        
        # Initialize model
        if model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(200, 100, 50),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError("model_type must be 'neural_network' or 'random_forest'")
    
    def preprocess_images(self, X):
        """Preprocess images for training/prediction"""
        # Normalize to [0, 1]
        X_processed = X / 255.0
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Apply PCA if requested
        if self.use_pca:
            X_final = self.pca.fit_transform(X_scaled)
            print(f"PCA: Reduced from {X_scaled.shape[1]} to {X_final.shape[1]} features")
        else:
            X_final = X_scaled
            
        return X_final
    
    def preprocess_new_images(self, X):
        """Preprocess new images using already fitted transformers"""
        X_processed = X / 255.0
        X_scaled = self.scaler.transform(X_processed)
        
        if self.use_pca:
            X_final = self.pca.transform(X_scaled)
        else:
            X_final = X_scaled
            
        return X_final
    
    def train(self, X, y):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        
        # Preprocess
        X_processed = self.preprocess_images(X)
        
        # Train model
        self.model.fit(X_processed, y)
        
        print("Training completed!")
        
        # Get feature importance if random forest
        if self.model_type == 'random_forest':
            self.feature_importance = self.model.feature_importances_
    
    def predict(self, X):
        """Predict parameters from images"""
        X_processed = self.preprocess_new_images(X)
        predictions = self.model.predict(X_processed)
        
        # Convert to list of dictionaries
        result = []
        for pred in predictions:
            param_dict = {name: val for name, val in zip(self.param_names, pred)}
            result.append(param_dict)
        
        return result
    
    def save_model(self, filename):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'param_names': self.param_names,
            'model_type': self.model_type,
            'use_pca': self.use_pca
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a trained model"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.param_names = model_data['param_names']
        self.model_type = model_data['model_type']
        self.use_pca = model_data['use_pca']
        print(f"Model loaded from {filename}")

def load_data_from_folder(base_folder, csv_file):
    """Load images and parameters from your folder structure"""
    
    print(f"Loading data from CSV: {csv_file}")
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")
    print(f"CSV columns: {list(df.columns)}")
    
    # Filter for classic GM model only
    classic_mask = df['Model_Type'] == 'Classic'
    df_classic = df[classic_mask].copy()
    print(f"Found {len(df_classic)} Classic G-M experiments")
    
    # Filter for Turing-capable patterns only
    success_mask = df_classic['Status'] == 'success'
    df_turing = df_classic[success_mask].copy()
    print(f"Found {len(df_turing)} Successful Classic G-M patterns")
    
    if len(df_turing) == 0:
        print("Warning: No successful classic simulations found!")
        # Use all classic patterns if no Turing-capable ones found
        df_turing = df_classic.copy()
        print(f"Using all {len(df_turing)} classic patterns instead")
    
    # Load images and parameters
    images = []
    parameters = []
    valid_rows = []
    
    for idx, row in df_turing.iterrows():
        # Get the output directory for this experiment
        output_dir = row['Output_Directory']
        
        # Construct full path to the image
        if os.path.isabs(output_dir):
            image_path = os.path.join(output_dir, 'pattern_simulation.png')
        else:
            image_path = os.path.join(base_folder, output_dir, 'pattern_simulation.png')
        
        if os.path.exists(image_path):
            try:
                # Load image
                img = Image.open(image_path)
                if img.mode != 'L':
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
                        param_row.append(0.0)
                
                parameters.append(param_row)
                valid_rows.append(row)
                
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        else:
            print(f"Image not found: {image_path}")
    
    if len(images) == 0:
        raise ValueError("No images were successfully loaded!")
    
    X = np.array(images, dtype=np.float32)
    y = np.array(parameters, dtype=np.float32)
    
    print(f"\nSuccessfully loaded {len(X)} image-parameter pairs")
    print(f"Image shape: {X.shape}")
    print(f"Parameter shape: {y.shape}")
    
    return X, y, pd.DataFrame(valid_rows)

def train_gm_predictor(base_folder, csv_file, model_type='neural_network', test_split=0.2):
    """Complete training pipeline using scikit-learn"""
    
    print("="*60)
    print(f"TRAINING G-M PARAMETER PREDICTOR ({model_type.upper()})")
    print("="*60)
    
    # Load data
    X, y, metadata = load_data_from_folder(base_folder, csv_file)
    
    if len(X) < 10:
        print(f"Warning: Only {len(X)} samples found. Results may not be reliable.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42
    )
    
    print(f"\nData split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and train model
    model = SimpleGMPredictor(model_type=model_type, use_pca=True, n_components=100)
    model.train(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    predictions = model.predict(X_test)
    
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
            
            if abs(true_val) > 1e-8:
                rel_error = abs(pred_val - true_val) / abs(true_val) * 100
                errors[param_name].append(rel_error)
                
                print(f"{i+1:<8} {param_name:<12} {true_val:<10.4f} {pred_val:<10.4f} {rel_error:<10.1f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY RESULTS:")
    print("="*60)
    
    for param_name in param_names:
        if len(errors[param_name]) > 0:
            mean_error = np.mean(errors[param_name])
            std_error = np.std(errors[param_name])
            print(f"{param_name:4s}: {mean_error:6.2f}% ± {std_error:5.2f}%")
    
    overall_error = np.mean([np.mean(errors[name]) for name in param_names if len(errors[name]) > 0])
    print(f"\nOverall average error: {overall_error:6.2f}%")
    
    # Plot results
    plot_results(predictions, y_test, param_names, model_type)
    
    return model, predictions, y_test, param_names

def plot_results(predictions, y_test, param_names, model_type):
    """Plot prediction results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. True vs Predicted scatter plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))
    
    for i, param_name in enumerate(param_names):
        true_vals = y_test[:, i]
        pred_vals = [pred[param_name] for pred in predictions]
        
        axes[0,0].scatter(true_vals, pred_vals, alpha=0.7, 
                         color=colors[i], label=param_name, s=30)
    
    # Perfect prediction line
    all_true = y_test.flatten()
    all_pred = np.array([[pred[name] for name in param_names] for pred in predictions]).flatten()
    min_val, max_val = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    axes[0,0].set_xlabel('True Parameter Value')
    axes[0,0].set_ylabel('Predicted Parameter Value')
    axes[0,0].set_title(f'Parameter Predictions ({model_type})')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Error distribution
    all_errors = []
    for i, param_name in enumerate(param_names):
        true_vals = y_test[:, i]
        pred_vals = [pred[param_name] for pred in predictions]
        
        for j in range(len(true_vals)):
            if abs(true_vals[j]) > 1e-8:
                rel_error = abs(pred_vals[j] - true_vals[j]) / abs(true_vals[j]) * 100
                all_errors.append(rel_error)
    
    axes[0,1].hist(all_errors, bins=15, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Relative Error (%)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Error Distribution')
    axes[0,1].axvline(np.mean(all_errors), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(all_errors):.1f}%')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Parameter-wise error comparison
    param_errors = {name: [] for name in param_names}
    for i, pred_dict in enumerate(predictions):
        for j, param_name in enumerate(param_names):
            true_val = y_test[i, j]
            pred_val = pred_dict[param_name]
            
            if abs(true_val) > 1e-8:
                rel_error = abs(pred_val - true_val) / abs(true_val) * 100
                param_errors[param_name].append(rel_error)
    
    # Box plot
    error_data = [param_errors[name] for name in param_names if len(param_errors[name]) > 0]
    valid_names = [name for name in param_names if len(param_errors[name]) > 0]
    
    if error_data:
        axes[1,0].boxplot(error_data, labels=valid_names)
        axes[1,0].set_xlabel('Parameter')
        axes[1,0].set_ylabel('Relative Error (%)')
        axes[1,0].set_title('Error by Parameter')
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    axes[1,1].axis('off')
    summary_text = f"Model: {model_type}\n\n"
    summary_text += "Parameter Errors:\n"
    
    for param_name in param_names:
        if len(param_errors[param_name]) > 0:
            mean_err = np.mean(param_errors[param_name])
            summary_text += f"{param_name}: {mean_err:.2f}%\n"
    
    summary_text += f"\nOverall: {np.mean(all_errors):.2f}%"
    summary_text += f"\nSamples: {len(predictions)}"
    
    axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes, 
                  fontsize=12, verticalalignment='center')
    axes[1,1].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Your paths
    BASE_FOLDER = "/home/ec2-user/multicellularcircuits/"
    CSV_FILE = "/home/ec2-user/multicellularcircuits/GM_training_data_targeted_20250528_211446/batch_experiments_summary.csv"
    
    print("Scikit-learn G-M Parameter Predictor")
    print("=" * 50)
    
    # Check paths
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file {CSV_FILE} does not exist!")
        exit()
    
    if not os.path.exists(BASE_FOLDER):
        print(f"Error: Base folder {BASE_FOLDER} does not exist!")
        exit()
    
    try:
        # Try both model types
        for model_type in ['neural_network', 'random_forest']:
            print(f"\n{'='*60}")
            print(f"TESTING {model_type.upper()} MODEL")
            print(f"{'='*60}")
            
            model, predictions, y_test, param_names = train_gm_predictor(
                BASE_FOLDER, 
                CSV_FILE, 
                model_type=model_type,
                test_split=0.2
            )
            
            # Save model
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'gm_{model_type}_model_{timestamp}.joblib'
            model.save_model(filename)
            
            print(f"\n{model_type} model training completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()