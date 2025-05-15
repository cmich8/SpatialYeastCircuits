# Experiment: Three_Colony_Ring

## Notes
Three colonies arranged in a triangle with different reporter strains.

## Configuration

* **Grid size**: (80, 60) (coarsened to (20, 15))
* **Spatial resolution (dx)**: 1.0 mm (coarsened to 4.0 mm)
* **Simulation time**: 48 hours
* **Number of time points**: 10

## Colony Configuration

### Colony 1 (beta->alpha)
* **Position**: (30, 20) (scaled to (7, 5))
* **Shape**: circle
* **Radius**: 5 (scaled to 1)
* **Concentration**: 0.5

### Colony 2 (alpha->venus)
* **Position**: (20, 30) (scaled to (5, 7))
* **Shape**: circle
* **Radius**: 5 (scaled to 1)
* **Concentration**: 0.5

### Colony 3 (IAA->GFP)
* **Position**: (40, 30) (scaled to (10, 7))
* **Shape**: circle
* **Radius**: 5 (scaled to 1)
* **Concentration**: 0.5

## Initial Molecule Distributions

### Distribution 1 (beta_estradiol)
* **Position**: (30, 20) (scaled to (7, 5))
* **Shape**: circle
* **Radius**: 6 (scaled to 1)
* **Concentration**: 100

## Results

* **Status**: Success
* **Time points**: 10
* **Timestamp**: 2025-05-13T20:20:32.060780

## Files

* **`spatial_results.png`**: Final state of the simulation
* **`growth_dashboard.png`**: Overview of strain growth over time
* **`growth_by_location.png`**: Growth at specific locations
* **`alphafactor_heatmap.gif`**: Animation of alpha_factor
* **`betaestradiol_heatmap.gif`**: Animation of beta_estradiol
* **`iaa_heatmap.gif`**: Animation of IAA
* **`venus_heatmap.gif`**: Animation of Venus
* **`gfp_heatmap.gif`**: Animation of GFP
* **`experiment_log.txt`**: Detailed log of the experiment
* **`experiment_metadata.json`**: Complete experiment configuration in JSON format
