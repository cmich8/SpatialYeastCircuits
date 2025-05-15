# Experiment: Basic_Two_Colony

## Notes
Basic two-colony experiment with default parameters.

## Configuration

* **Grid size**: (70, 20) (coarsened to (17, 5))
* **Spatial resolution (dx)**: 1.0 mm (coarsened to 4.0 mm)
* **Simulation time**: 48 hours
* **Number of time points**: 10

## Colony Configuration

### Colony 1 (beta->alpha)
* **Position**: (30, 10) (scaled to (7, 2))
* **Shape**: circle
* **Radius**: 5 (scaled to 1)
* **Concentration**: 0.5

### Colony 2 (alpha->venus)
* **Position**: (40, 10) (scaled to (10, 2))
* **Shape**: circle
* **Radius**: 5 (scaled to 1)
* **Concentration**: 0.5

## Initial Molecule Distributions

### Distribution 1 (beta_estradiol)
* **Position**: (30, 10) (scaled to (7, 2))
* **Shape**: rectangle
* **Width**: 10 (scaled to 2)
* **Height**: 10 (scaled to 2)
* **Concentration**: 100

## Results

* **Status**: Success
* **Time points**: 1
* **Timestamp**: 2025-05-13T20:20:20.695041

## Files

* **`spatial_results.png`**: Final state of the simulation
* **`growth_dashboard.png`**: Overview of strain growth over time
* **`growth_by_location.png`**: Growth at specific locations
* **`alphafactor_heatmap.gif`**: Animation of alpha_factor
* **`betaestradiol_heatmap.gif`**: Animation of beta_estradiol
* **`venus_heatmap.gif`**: Animation of Venus
* **`experiment_log.txt`**: Detailed log of the experiment
* **`experiment_metadata.json`**: Complete experiment configuration in JSON format
