from simulation_v5_strainspecificgrowthrates import *
#########
# Experimental Plan
# 1. varying distances apart
#
#
#
#
#
#
#
#
#
#
########
outputdir = '/home/ec2-user/multicellularcircuits/firstpassexperiments'
experiment = 'test'
def quick_two_strain_relay(experiment, outputdir):
    strain_library = create_strain_library()
    
    # Create smaller model
    model = SpatialMultiStrainModel(grid_size=(10, 10), dx=1)
    
    # Add strains
    model.add_strain(strain_library['beta->alpha'])  # Sender
    model.add_strain(strain_library['alpha->venus']) # Receiver
    
    # Place sender strain on the left
    model.place_strain(0, row=5, col=5, shape="circle", radius=2, concentration=10.0)
    
    # Place receiver strain on the right
    model.place_strain(1, row=5, col=5, shape="circle", radius=2, concentration=10.0)
    
    # Place beta-estradiol input
    model.place_molecule(BETA, 5, 5, radius=2, concentration=100.0)
    
    # Set shorter simulation time
    model.set_simulation_time(0, 24)
    
    # Run simulation with fewer time points
    results = model.simulate(n_time_points=24)
    
    # Plot just the final state
    figsr = model.plot_spatial_results(results, time_idx=-1, molecules=[ALPHA, BETA, VENUS])
    figsr.savefig(f'{outputdir}/{experiment}_spatialresults.png')
    gfp_anim = model.create_animation(results, molecule=VENUS, 
                                       interval=100, cmap='viridis')
    gfp_anim.save(f'{outputdir}/{experiment}_venus_heatmap.gif', writer='ffmpeg', fps=5)
    alpha_anim = model.create_animation(results, molecule=ALPHA, 
                                       interval=100, cmap='viridis')
    alpha_anim.save('{outputdir}/{experiment}_alpha_heatmap.gif', writer='ffmpeg', fps=5)
    beta_anim = model.create_animation(results, molecule=BETA, 
                                       interval=100, cmap='viridis')
    beta_anim.save('{outputdir}/{experiment}_beta_heatmap.gif', writer='ffmpeg', fps=5)
    
    fig = create_growth_dashboard(results, model)
    fig.savefig(f'{outputdir}/{experiment}_growthdashboard.png')
    fig2 = plot_strain_growth(results, model, specific_locations=[(20, 20), (30, 40)])  # With specific locations
    fig2.savefig(f'{outputdir}/{experiment}_growthbylocation.png')

    return results, model

quick_two_strain_relay(experiment, outputdir)
