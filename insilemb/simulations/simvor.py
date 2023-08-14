import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from insilemb.bounded_voronoi import BoundedVoronoi
from insilemb.embryoids import TopologicalEmbryoid
import insilemb.pl as pl
from tqdm import tqdm


def run_voronoi_simulation(ncells, nu, alpha, beta, 
                           ic=None, dt=0.1, nsteps=100, **kwargs):
    """Simulate a Voronoi partition of cells.
     
    Arguments:
        ncells: int -- number of cells in the partition.
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    boundary_type = kwargs.get('boundary_type', None)
    regularization = kwargs.get('regularization', None)
    burnin = kwargs.get('burnin', 0)
    fix_cells = kwargs.get('fix_cells', None)
    nonlinearity = kwargs.get('nonlinearity', None)
    saverate = kwargs.get('saverate', 1)
    show_pbar = kwargs.get('show_pbar', True)
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
        ic = baselevel*np.ones([2, nnodes]) + sigma*np.random.randn(2, nnodes)

    vor = BoundedVoronoi(ncells, boundary_type=boundary_type)
    if regularization:
        for i in tqdm(range(regularization), desc="Regularizing", 
                      disable=not show_pbar):
            vor = vor.recenter()

    locations = vor.get_points()
    emb = TopologicalEmbryoid(
        ncells, vor.get_adjacency_matrix(), fields=ic,
        diffusivities=nu, alphas=alpha, betas=beta,
        boundary_idx=boundary_idx, locations=locations,
        nonlinearity=nonlinearity
    )

    # if fix_cells:
    #     emb.fix_cells(vor.get_edge_cells(1), 100, 0)
    #     emb.fix_cells(vor.get_edge_cells(2), 100, 1)
    
    print("Running simulation...")
    # history = np.empty([nsteps, *emb.get_fields().shape], dtype=np.float32)
    plt_history = [] #[emb.get_fields().copy()]
    plt_times = [] #[0]
    t = 0
    for i in tqdm(range(nsteps), desc="Simulating", disable=not show_pbar):
        t += dt
        emb.step(dt=dt)
        # history[i] = emb.get_fields().astype(np.float32)
        if t >= burnin and (i+1) % saverate == 0:
            # plt_history.append(history[i])
            plt_history.append(emb.get_fields().astype(np.float32))
            plt_times.append(t)
    plt_history = np.array(plt_history)
    print("Plotting...")
    norm0 = Normalize(vmin=plt_history[:,0].min(), vmax=plt_history[:,0].max())
    norm1 = Normalize(vmin=plt_history[:,1].min(), vmax=plt_history[:,1].max())
    for i in tqdm(range(len(plt_history)), desc="Plotting", 
                  disable=not show_pbar):
        t = plt_times[i]
        data = plt_history[i]
        pl.plot_bounded_voronoi(
            vor, 
            plot_points=False,
            plot_centroids=False,
            plot_vertices=False,
            plot_data=True,
            data=data[0],
            norm=norm0,
            cmap='jet',
            title=f"$t={t:.4g}$",
            saveas=f"{outdir}/imga{str(i).zfill(len(str(nsteps)))}.png",
        )
        plt.close()
        pl.plot_bounded_voronoi(
            vor, 
            plot_points=False,
            plot_centroids=False,
            plot_vertices=False,
            plot_data=True,
            data=data[1],
            norm=norm1,
            cmap='jet',
            title=f"$t={t:.4g}$",
            saveas=f"{outdir}/imgb{str(i).zfill(len(str(nsteps)))}.png",
        )
        plt.close()
    print("Done!")


def main():
    ncells = 10000
    tfin = 5000
    dt = 0.01
    saverate = 2000
    nu = [0.014, 0.4]  # diffusion coefficients
    alpha = [0.0, 0.0]  # constant production term
    beta = [0.001, 0.008]  # linear degradation term
    boundary_type=None
    n = ncells
    
    # Specify initial condition
    # ic = np.ones([2, n]) + 0.1 * np.random.randn(2, n)
    ic = np.random.rand(2, n)

    # Define nonlinear term of dynamics
    ka = 0
    ki = 1
    sa = 0.01
    si = 0.01
    def nonlinearity(data, idx):
        if idx == 0:
            return sa * data[0]*data[0] \
                / (1 + ki * data[1]) \
                / (1 + ka * data[0]*data[0])
        elif idx == 1:
            return si * data[0] * data[0]

    nsteps = int(tfin/dt)
    run_voronoi_simulation(
        ncells, nu, alpha, beta, 
        ic=ic, dt=dt, nsteps=nsteps, saverate=saverate,
        boundary_type=boundary_type, regularization=20,
        nonlinearity=nonlinearity, show_pbar=False,
    )

if __name__ == "__main__":
    main()
