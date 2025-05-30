#!/usr/bin/env python3
"""
Fixed version of the batch experiment script with JSON serialization fix
"""

import os
import sys
import csv
import argparse
import datetime
import json
import shutil
from pathlib import Path
import numpy as np

# Import functions from GMmodeling.py
try:
    from GMmodeling import (
        TuringAnalyzer, 
        f_classic, g_classic, 
        f_saturated, g_saturated,
        run_turing_experiment
    )
except ImportError as e:
    print(f"Error: Could not import GMmodeling functions: {e}")
    print("Make sure GMmodeling.py is in the same directory as this script.")
    sys.exit(1)

def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def read_parameters_csv(csv_file):
    """
    Read parameters from CSV file.
    
    Args:
        csv_file: Path to CSV file with parameters
        
    Returns:
        List of parameter dictionaries
    """
    parameters = []
    
    try:
        with open(csv_file, 'r', newline='') as f:
            # Try to detect if there's a header
            sample = f.read(1024)
            f.seek(0)
            
            # Check if first row looks like parameter values (all numeric)
            first_line = f.readline().strip()
            f.seek(0)
            
            has_header = False
            try:
                # Try to parse first line as floats
                values = [float(x.strip()) for x in first_line.split(',')]
                # If successful and we have 6 values, assume no header
                if len(values) == 6:
                    has_header = False
                else:
                    has_header = True
            except ValueError:
                # If parsing fails, assume there's a header
                has_header = True
            
            reader = csv.reader(f)
            
            # Skip header if present
            if has_header:
                next(reader)
                print("Detected header row, skipping...")
            
            for row_num, row in enumerate(reader, start=1):
                if len(row) < 6:
                    print(f"Warning: Row {row_num} has only {len(row)} values, skipping...")
                    continue
                
                try:
                    # Parse the first 6 columns as a, b, c, d, Du, Dv
                    a = float(row[0].strip())
                    b = float(row[1].strip())
                    c = float(row[2].strip())
                    d = float(row[3].strip())
                    Du = float(row[4].strip())
                    Dv = float(row[5].strip())
                    
                    parameters.append({
                        'a': a,
                        'b': b,
                        'c': c,
                        'd': d,
                        'Du': Du,
                        'Dv': Dv,
                        'row_number': row_num
                    })
                    
                except ValueError as e:
                    print(f"Warning: Could not parse row {row_num}: {e}")
                    print(f"Row content: {row}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: Could not find CSV file: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    return parameters

def create_output_directory(base_dir="GM_training_data_sobol"):
    """
    Create the main output directory and return its path.
    
    Args:
        base_dir: Base directory name
        
    Returns:
        Path to created directory
    """
    # Create timestamp for this batch run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_dir = f"{base_dir}_{timestamp}"
    
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
        print(f"Created output directory: {full_dir}")
    
    return full_dir

def run_batch_experiments(csv_file, output_dir, args):
    """
    Run batch experiments from CSV parameters.
    
    Args:
        csv_file: Path to CSV file with parameters
        output_dir: Output directory for results
        args: Command line arguments
        
    Returns:
        Dictionary with batch run statistics
    """
    print(f"Reading parameters from {csv_file}...")
    parameters = read_parameters_csv(csv_file)
    
    if not parameters:
        print("Error: No valid parameters found in CSV file.")
        return None
    
    print(f"Found {len(parameters)} parameter sets to process.")
    
    # Create summary file for this batch
    batch_csv_file = os.path.join(output_dir, "batch_experiments_summary.csv")
    
    # Statistics tracking
    stats = {
        'total_experiments': len(parameters),
        'successful_experiments': 0,
        'failed_experiments': 0,
        'turing_capable_count': 0,
        'start_time': datetime.datetime.now(),
        'experiment_details': []
    }
    
    print(f"\nStarting batch processing of {len(parameters)} experiments...")
    print(f"Results will be saved to: {output_dir}")
    print(f"Model type: {'Saturated' if args.saturated else 'Classic'} Gierer-Meinhardt")
    
    # Process each parameter set
    for i, params in enumerate(parameters, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(parameters)}] Processing experiment {i}")
        print(f"Parameters: a={params['a']:.6f}, b={params['b']:.6f}, c={params['c']:.6f}, d={params['d']:.6f}, Du={params['Du']:.6f}, Dv={params['Dv']:.6f}")
        
        # Create experiment name
        experiment_name = f"Experiment_{i:04d}_Row_{params['row_number']}"
        
        try:
            # Create experiments subdirectory if it doesn't exist
            temp_base_dir = os.path.join(output_dir, "experiments")
            if not os.path.exists(temp_base_dir):
                os.makedirs(temp_base_dir)
            
            # Run the experiment
            experiment_result = run_custom_turing_experiment(
                a=params['a'],
                b=params['b'], 
                c=params['c'],
                d=params['d'],
                Du=params['Du'],
                Dv=params['Dv'],
                rho=args.rho,
                grid_size=args.grid_size,
                time_points=args.time_points,
                dt=args.dt,
                experiment_name=experiment_name,
                use_saturated=args.saturated,
                saturation=args.saturation,
                output_base_dir=temp_base_dir,
                no_animation=True
            )
            
            if experiment_result:
                dir_path, is_turing_capable = experiment_result
                stats['successful_experiments'] += 1
                if is_turing_capable:
                    stats['turing_capable_count'] += 1
                
                # Record experiment details (with numpy type conversion)
                stats['experiment_details'].append({
                    'experiment_number': int(i),
                    'row_number': int(params['row_number']),
                    'parameters': convert_numpy_types(params),
                    'is_turing_capable': bool(is_turing_capable),
                    'output_directory': str(dir_path),
                    'status': 'success'
                })
                
                print(f"✓ Experiment {i} completed successfully")
                print(f"  Turing capable: {'YES' if is_turing_capable else 'NO'}")
                print(f"  Results saved to: {dir_path}")
                
            else:
                raise Exception("Experiment returned None")
                
        except Exception as e:
            print(f"✗ Experiment {i} failed: {str(e)}")
            stats['failed_experiments'] += 1
            
            # Record failed experiment (with numpy type conversion)
            stats['experiment_details'].append({
                'experiment_number': int(i),
                'row_number': int(params['row_number']),
                'parameters': convert_numpy_types(params),
                'is_turing_capable': False,
                'output_directory': None,
                'status': 'failed',
                'error': str(e)
            })
            
            continue
    
    # Calculate final statistics
    stats['end_time'] = datetime.datetime.now()
    stats['total_runtime'] = stats['end_time'] - stats['start_time']
    stats['success_rate'] = stats['successful_experiments'] / stats['total_experiments'] * 100
    stats['turing_rate'] = stats['turing_capable_count'] / stats['successful_experiments'] * 100 if stats['successful_experiments'] > 0 else 0
    
    # Create comprehensive batch summary CSV
    create_batch_summary_csv(batch_csv_file, stats, args)
    
    # Save detailed batch statistics as JSON
    stats_file = os.path.join(output_dir, "batch_statistics.json")
    save_batch_statistics(stats_file, stats, args)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {stats['total_experiments']}")
    print(f"Successful: {stats['successful_experiments']}")
    print(f"Failed: {stats['failed_experiments']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Turing-capable experiments: {stats['turing_capable_count']}")
    print(f"Turing rate (of successful): {stats['turing_rate']:.1f}%")
    print(f"Total runtime: {stats['total_runtime']}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary CSV: {batch_csv_file}")
    print(f"Statistics JSON: {stats_file}")
    
    return stats

def run_custom_turing_experiment(a, b, c, d, Du, Dv, rho=0.01, grid_size=100, time_points=2000, dt=0.1,
                                experiment_name=None, use_saturated=False, saturation=0.01, 
                                output_base_dir="turing_experiments", no_animation=True):
    """
    Custom version of run_turing_experiment with JSON serialization fix.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Create parameters - convert to Python native types
    params = {
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'd': float(d),
        'rho': float(rho)
    }
    
    if use_saturated:
        params['saturation'] = float(saturation)
        f_func, g_func = f_saturated, g_saturated
        model_type = "Saturated"
    else:
        f_func, g_func = f_classic, g_classic
        model_type = "Classic"
    
    # Create directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    if experiment_name:
        clean_name = experiment_name.replace(" ", "_").replace("/", "-").replace("\\", "-")
        dir_name = f"{model_type}_{clean_name}_a{a:.6f}_b{b:.6f}_c{c:.6f}_d{d:.6f}_Du{Du:.6f}_Dv{Dv:.6f}_{timestamp}"
    else:
        dir_name = f"{model_type}_a{a:.6f}_b{b:.6f}_c{c:.6f}_d{d:.6f}_Du{Du:.6f}_Dv{Dv:.6f}_{timestamp}"
    
    dir_path = os.path.join(output_base_dir, dir_name)
    
    # Create directories
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Save experiment parameters - convert all to Python native types
    params_with_diffusion = {
        'model_type': str(model_type),
        'a': float(a), 
        'b': float(b), 
        'c': float(c), 
        'd': float(d),
        'rho': float(rho),
        'Du': float(Du), 
        'Dv': float(Dv),
        'use_saturated': bool(use_saturated),
        'saturation': float(saturation) if use_saturated else None,
        'grid_size': int(grid_size),
        'time_points': int(time_points),
        'dt': float(dt),
        'experiment_name': str(experiment_name or "Unnamed Experiment"),
        'timestamp': str(timestamp)
    }
    
    params_file = os.path.join(dir_path, "parameters.json")
    with open(params_file, 'w') as f:
        json.dump(params_with_diffusion, f, indent=4)
    
    # Initialize analyzer
    analyzer = TuringAnalyzer((f_func, g_func), (Du, Dv), params)
    
    # Try multiple initial guesses to find steady state
    initial_guesses = [
        [1.0, 1.0],
        [0.5, 0.5],
        [2.0, 2.0],
        [0.2, 0.8],
        [0.8, 0.2],
        [np.sqrt(float(a)/float(b)) if b > 0 else 1.0, float(a)/(float(c)*float(b)) if (c*b) > 0 else 1.0]
    ]
    
    # Try each initial guess until we get a valid result
    results = None
    steady_state = None
    
    for i, guess in enumerate(initial_guesses):
        try:
            results = analyzer.analyze_system(initial_guess=guess)
            
            if results and results.get('has_steady_state', False):
                steady_state = results['steady_state']
                break
        except Exception as e:
            continue
    
    if not results or not results.get('has_steady_state', False):
        return None
    
    # Create report and save to file
    try:
        report = analyzer.create_report(results)
        report_path = os.path.join(dir_path, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
    except Exception as e:
        print(f"Warning: Could not create analysis report: {e}")
    
    # Plot dispersion relation and save
    try:
        dispersion_path = os.path.join(dir_path, "dispersion_relation.png")
        analyzer.plot_dispersion_relation(results, save_path=dispersion_path)
        plt.close('all')
    except Exception as e:
        print(f"Warning: Could not create dispersion plot: {e}")
    
    # Run simulation
    try:
        simulation_results = analyzer.simulate_pattern(
            grid_size=grid_size, 
            spatial_size=10.0,
            time_points=time_points,
            dt=dt,
            noise_amplitude=0.05,
            steady_state=steady_state,
            save_frames=2 if no_animation else 10
        )
        
        # Plot simulation results
        simulation_path = os.path.join(dir_path, "pattern_simulation.png")
        fig = analyzer.plot_simulation_results(
            simulation_results, 
            save_path=simulation_path,
            save_animation=False
        )
        plt.close('all')
        
    except Exception as e:
        print(f"Warning: Simulation failed: {str(e)}")
    
    # Convert is_turing_capable to Python bool
    is_turing_capable = bool(results.get('is_turing_capable', False))
    
    return dir_path, is_turing_capable

def create_batch_summary_csv(csv_file, stats, args):
    """
    Create a comprehensive CSV summary of the batch run.
    """
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Experiment_Number', 'Row_Number', 'Model_Type', 'a', 'b', 'c', 'd', 
            'Du', 'Dv', 'Rho', 'Saturation', 'Turing_Capable', 'Status', 
            'Output_Directory', 'Error'
        ])
        
        # Write experiment data
        for exp in stats['experiment_details']:
            params = exp['parameters']
            writer.writerow([
                exp['experiment_number'],
                exp['row_number'],
                'Saturated' if args.saturated else 'Classic',
                params['a'], params['b'], params['c'], params['d'],
                params['Du'], params['Dv'],
                args.rho,
                args.saturation if args.saturated else 'N/A',
                'Yes' if exp['is_turing_capable'] else 'No',
                exp['status'],
                exp['output_directory'] or 'N/A',
                exp.get('error', '')
            ])

def save_batch_statistics(stats_file, stats, args):
    """
    Save detailed batch statistics as JSON with proper type conversion.
    """
    # Convert datetime objects and ensure all types are JSON serializable
    stats_copy = convert_numpy_types(stats.copy())
    stats_copy['start_time'] = stats['start_time'].isoformat()
    stats_copy['end_time'] = stats['end_time'].isoformat()
    stats_copy['total_runtime'] = str(stats['total_runtime'])
    
    # Add run configuration
    stats_copy['run_configuration'] = {
        'model_type': 'Saturated' if args.saturated else 'Classic',
        'grid_size': int(args.grid_size),
        'time_points': int(args.time_points),
        'dt': float(args.dt),
        'rho': float(args.rho),
        'saturation': float(args.saturation) if args.saturated else None,
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_copy, f, indent=4)

def main():
    """Main function to parse arguments and run batch experiments."""
    parser = argparse.ArgumentParser(description='Run batch Turing pattern experiments from CSV file')
    
    # Required parameter
    parser.add_argument('csv_file', help='CSV file with parameters (columns: a, b, c, d, Du, Dv)')
    
    # Model type selection
    parser.add_argument('--saturated', action='store_true', help='Use saturated G-M model (default: classic)')
    parser.add_argument('--saturation', type=float, default=0.01, help='Saturation parameter for saturated model')
    parser.add_argument('--rho', type=float, default=0.01, help='Basal activator production')
    
    # Simulation parameters
    parser.add_argument('--grid_size', type=int, default=100, help='Number of grid points (default: 100)')
    parser.add_argument('--time_points', type=int, default=8000, help='Number of simulation time steps (default: 8000)')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step size (default: 0.05)')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='GM_training_data_targeted', 
                       help='Base output directory name (default: GM_training_data_targeted)')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        sys.exit(1)
    
    print("Batch Turing Pattern Experiment Runner (Fixed Version)")
    print("===================================================")
    print(f"CSV file: {args.csv_file}")
    print(f"Model type: {'Saturated' if args.saturated else 'Classic'} Gierer-Meinhardt")
    print(f"Simulation settings: grid_size={args.grid_size}, time_points={args.time_points}, dt={args.dt}")
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    # Copy the input CSV to the output directory for reference
    input_csv_copy = os.path.join(output_dir, "input_parameters.csv")
    shutil.copy2(args.csv_file, input_csv_copy)
    print(f"Input CSV copied to: {input_csv_copy}")
    
    # Run batch experiments
    try:
        stats = run_batch_experiments(args.csv_file, output_dir, args)
        
        if stats:
            print(f"\nBatch processing completed successfully!")
            print(f"Check {output_dir} for all results.")
        else:
            print("Batch processing failed.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()