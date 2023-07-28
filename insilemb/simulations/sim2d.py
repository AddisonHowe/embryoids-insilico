import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from insilemb.embryoids import TopologicalEmbryoid
import insilemb.pl as pl


def run_2d_simulation(nrows, ncols, nu, alpha, beta, 
                      ic=None, dt=0.1, nsteps=100, **kwargs):
    """Simulate a grid arrangement of cells.
     
    Arguments:
        nrows: int -- number of rows in the grid
        ncols: int -- number of columns in the grid
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    boundary_type = kwargs.get('boundary_type', None)
    dx = kwargs.get('dx', 1)
    dy = kwargs.get('dy', 1)
    saverate = kwargs.get('saverate', 1)
    outdir = kwargs.get('outdir', "out/sims/sim2d")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if outdir: 
        os.makedirs(outdir, exist_ok=True)
    ncells = nrows * ncols
    if boundary_type is None:
        arr = np.zeros(ncols)
        arr[1] = 1
        offdi = scipy.linalg.toeplitz(arr)
        id = np.eye(nrows)
        adj = np.kron(offdi, id) + np.kron(id, offdi)
        boundary_idx = None
        nnodes = ncells
    elif boundary_type == "periodic":
        arr = np.zeros(ncols)
        arr[1] = 1
        arr[-1] = 1
        offdi = scipy.linalg.circulant(arr)
        id = np.eye(nrows)
        adj = np.kron(offdi, id) + np.kron(id, offdi)
        boundary_idx = None
        nnodes = ncells
    elif boundary_type == "ambient":
        arr = np.zeros(ncols)
        arr[1] = 1
        offdi = scipy.linalg.toeplitz(arr)
        id = np.eye(nrows)
        adj = np.zeros([ncells + 1, ncells + 1])
        adj[0:-1,0:-1] = np.kron(offdi, id) + np.kron(id, offdi)
        adj[-1, 0:ncols] = 1
        adj[-1, -ncols-1:-1] = 1
        for i in range(1, nrows-1):
            adj[-1, i*ncols] = 1
            adj[-1, (i+1)*ncols-1] = 1
        adj[:,-1] = adj[-1,:]
        boundary_idx = -1
        nnodes = ncells + 1
    else:
        msg = f"Boundary type {boundary_type} not implemented."
        raise NotImplementedError(msg)
    
    if ic is None:
        baselevel = 1
        sigma = 0.1  # noise level for initial conditions
        ic = baselevel * np.ones(nnodes) + sigma * np.random.randn(nnodes) 
    
    locations = np.zeros([ncells, 2])
    for rowidx in range(nrows):
        locations[ncols*rowidx:ncols*(rowidx+1),0] = dx * np.arange(ncols)
        locations[ncols*rowidx:ncols*(rowidx+1),1] = dy * (nrows - rowidx)
        
    emb = TopologicalEmbryoid(
        ncells, adj, data=ic, 
        diffusivities=[nu], alphas=[alpha], betas=[beta], 
        boundary_idx=boundary_idx,
        locations=locations,
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
        pl.plot_embryoid_cells(
            emb, 
            data=data[0],
            locations=locations,
            cell_radius=0.5, 
            norm=norm,
            boundary_type=boundary_type,
            title=f"$t={t:.4g}$",
            saveas=f"{outdir}/img{i}.png"
        )
    
    print(emb.get_data(0).sum())
    print("Done!")


def main():
    nrows = 30
    ncols = 30
    tfin = 3
    dt = 0.01
    nu = 1
    alpha = 0.1
    beta = 0.00
    boundary_type=None
    # boundary_type="ambient"
    n = nrows * ncols
    if boundary_type == "ambient":
        n += 1
    ic = np.ones(n) + 0.1 * np.random.randn(n)
    for rowidx in range(nrows//2):
        ic[ncols*(rowidx) + ncols//2] = 2
    # ic[-1] = 10
    nsteps = int(tfin/dt)
    run_2d_simulation(
        nrows, ncols, nu, alpha, beta, 
        ic=ic, dt=dt, nsteps=nsteps, 
        boundary_type=boundary_type,
        saverate = 10
    )

if __name__ == "__main__":
    main()
