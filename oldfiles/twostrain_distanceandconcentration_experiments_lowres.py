from simulation_v6 import *

#########
# Experimental Plan
# 1. varying distances apart
#   a. touching, 1cm, 2cm, 5cm
#   b. both Alpha and IAA to fluorescence
#   c. 10nM of Beta Estradiol
#   d. all colonies are 1x1 cm at a starting concentration of 0.01 (1:100 dilution of 1.0 OD starter culture)
#   
#2. Varying Concentrations of Beta Estradiol
#   0nM, 1, 10nM
#   1cm apart
#   b. both Alpha and IAA to fluorescence
#
########

outputdir = '/home/ec2-user/multicellularcircuits/firstpassexperiments'
experiment = 'Alpha_0cm'
def quick_two_strain_relay(experiment, outputdir):

    strain_library = create_strain_library()
    # Create smaller model
    model = SpatialMultiStrainModel(grid_size=(70, 20), dx=1)
    # Add strains
    model.add_strain(strain_library['beta->alpha'])  # Sender
    model.add_strain(strain_library['alpha->venus']) # Receiver
    # Place sender strain on the left
    model.place_strain(0, row=30, col=10, shape="circle", radius=5, concentration=0.5)
    # Place receiver strain on the right
    model.place_strain(1, row=40, col=10, shape="circle", radius=5, concentration=0.5)
    # Place beta-estradiol input
    model.place_molecule(BETA, row=30, col=10, shape="circle", radius=5, concentration=100)
    model.set_simulation_time(0, 48)
    # Run simulation with fewer time points
    results = model.simulate(n_time_points=10)
    # Plot just the final state
    figsr = model.plot_spatial_results(results, time_idx=-1, molecules=[ALPHA, BETA, VENUS])
    figsr.savefig(f'{outputdir}/{experiment}_spatialresults.png')
    gfp_anim = model.create_animation(results, molecule=VENUS, 
                                       interval=100, cmap='viridis')
    gfp_anim.save(f'{outputdir}/{experiment}_venus_heatmap.gif', writer='ffmpeg', fps=5)
    alpha_anim = model.create_animation(results, molecule=ALPHA, 
                                       interval=100, cmap='viridis')
    alpha_anim.save(f'{outputdir}/{experiment}_alpha_heatmap.gif', writer='ffmpeg', fps=5)
    beta_anim = model.create_animation(results, molecule=BETA, 
                                       interval=100, cmap='viridis')
    beta_anim.save(f'{outputdir}/{experiment}_beta_heatmap.gif', writer='ffmpeg', fps=5)
    fig = create_growth_dashboard(results, model)
    fig.savefig(f'{outputdir}/{experiment}_growthdashboard.png')
    fig2 = plot_strain_growth(results, model, specific_locations=[(20, 20), (30, 40)])  # With specific locations
    fig2.savefig(f'{outputdir}/{experiment}_growthbylocation.png')

    return results, model

quick_two_strain_relay(experiment, outputdir)
