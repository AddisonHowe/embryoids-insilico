import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from insilemb.bounded_voronoi import BoundedVoronoi
from insilemb.embryoids import TopologicalEmbryoid
import insilemb.pl as pl


def run_voronoi_simulation(ncells, nu, alpha, beta, 
                           ic=None, dt=0.1, nsteps=100, **kwargs):
    """Simulate a Voronoi partition of cells.
     
    Arguments:
        ncells: int -- number of cells in the partition.
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    boundary_type = kwargs.get('boundary_type', None)
    saverate = kwargs.get('saverate', 1)
    outdir = kwargs.get('outdir', "out/sims/voronoi")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if outdir: 
        os.makedirs(outdir, exist_ok=True)
    
    if boundary_type is None:
        boundary_idx = None
        nnodes = ncells
    elif boundary_type == "ambient":
        boundary_idx = -1
        nnodes = ncells + 1
    else:
        msg = f"Boundary type {boundary_type} not implemented."
        raise NotImplementedError(msg)
    
    if ic is None:
        baselevel = 1
        sigma = 0.1  # noise level for initial conditions
        ic = baselevel * np.ones(nnodes) + sigma * np.random.randn(nnodes) 

    vor = BoundedVoronoi(ncells, boundary_type=boundary_type)
    locations = vor.get_points()
    emb = TopologicalEmbryoid(
        ncells, vor.get_adjacency_matrix(), data=ic,
        diffusivities=[nu], alphas=[alpha], betas=[beta],
        boundary_idx=boundary_idx, locations=locations,
    )
    
    print("Running simulation...")
    history = np.empty([nsteps, *emb.get_data().shape])
    plt_history = [emb.get_data()]
    plt_times = [0]
    t = 0
    for i in range(nsteps):
        t += dt
        emb.step(dt=dt)
        history[i] = emb.get_data(0)
        if (i+1) % saverate == 0:
            plt_history.append(history[i])
            plt_times.append(t)
    
    print("Plotting...")
    norm = Normalize(vmin=history.min(), vmax=history.max())
    for i, data in enumerate(plt_history):
        t = plt_times[i]
        pl.plot_bounded_voronoi(
            vor, 
            plot_points = False,
            plot_centroids = False,
            plot_vertices = False,
            plot_data = True,
            data = data[0],
            norm=norm,
            cmap = 'jet',
            title=f"$t={t:.4g}$",
            saveas=f"{outdir}/img{str(i).zfill(len(str(nsteps)))}.png",
        )
        plt.close()
    print("Done!")


def main():
    ncells = 1000
    tfin = 10
    dt = 0.01
    nu = 10
    alpha = 0.1
    beta = 0.05
    boundary_type=None
    n = ncells
    ic = np.ones(n) + 0.1 * np.random.randn(n)
    nsteps = int(tfin/dt)
    run_voronoi_simulation(
        ncells, nu, alpha, beta, 
        ic=ic, dt=dt, nsteps=nsteps, 
        boundary_type=boundary_type,
        saverate = 10
    )

if __name__ == "__main__":
    main()
