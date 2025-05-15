# Experiment: Complex_Arrangement

## Notes
Complex arrangement with four colonies and three different starting molecules.

## Configuration

* **Grid size**: (100, 60) (coarsened to (25, 15))
* **Spatial resolution (dx)**: 1.0 mm (coarsened to 4.0 mm)
* **Simulation time**: 72 hours
* **Number of time points**: 15

## Colony Configuration

### Colony 1 (beta->alpha)
* **Position**: (25, 15) (scaled to (6, 3))
* **Shape**: rectangle
* **Width**: 10 (scaled to 2)
* **Height**: 6 (scaled to 1)
* **Concentration**: 0.5

### Colony 2 (alpha->venus)
* **Position**: (25, 35) (scaled to (6, 8))
* **Shape**: rectangle
* **Width**: 10 (scaled to 2)
* **Height**: 6 (scaled to 1)
* **Concentration**: 0.5

### Colony 3 (IAA->GFP)
* **Position**: (45, 25) (scaled to (11, 6))
* **Shape**: circle
* **Radius**: 7 (scaled to 1)
* **Concentration**: 0.5

### Colony 4 (alpha->alpha)
* **Position**: (5, 25) (scaled to (1, 6))
* **Shape**: circle
* **Radius**: 7 (scaled to 1)
* **Concentration**: 0.5

## Initial Molecule Distributions

### Distribution 1 (beta_estradiol)
* **Position**: (25, 15) (scaled to (6, 3))
* **Shape**: rectangle
* **Width**: 12 (scaled to 3)
* **Height**: 8 (scaled to 2)
* **Concentration**: 80

### Distribution 2 (alpha_factor)
* **Position**: (5, 25) (scaled to (1, 6))
* **Shape**: circle
* **Radius**: 8 (scaled to 2)
* **Concentration**: 20

### Distribution 3 (IAA)
* **Position**: (45, 25) (scaled to (11, 6))
* **Shape**: circle
* **Radius**: 3 (scaled to 1)
* **Concentration**: 50

## Results

* **Status**: Success
* **Time points**: 1
* **Timestamp**: 2025-05-13T22:26:12.689364

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
