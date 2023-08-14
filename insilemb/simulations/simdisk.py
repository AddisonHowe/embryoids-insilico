import os, sys, time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from insilemb.bounded_voronoi import BoundedVoronoi
from insilemb.embryoids import TopologicalEmbryoid
import insilemb.pl as pl
from tqdm import tqdm

cupy_success = True
try:
    import cupy as cp
except ImportError:
    cupy_success = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--ncells', type=int, default=10000)
    parser.add_argument('-t', '--tfin', type=float, default=5000)
    parser.add_argument('-dt', '--dt', type=float, default=0.01)
    parser.add_argument('-s', '--saverate', type=int, default=2000)
    parser.add_argument('-b', '--boundary_type', type=str, default=None)
    parser.add_argument('-r', '--regularization', type=int, default=10)
    parser.add_argument('-o', '--outdir', type=str, default="out/sims/simdisk")
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('--use_gpu', action="store_true")
    args = parser.parse_args()
    return args
    
def run_voronoi_simulation(ncells, 
                           nus_inner, nus_outer,  
                           alphas_inner, alphas_outer, 
                           betas_inner, betas_outer, 
                           sa_inner, sa_outer, 
                           si_inner, si_outer, 
                           ka, ki, 
                           ic=None, dt=0.1, nsteps=100, **kwargs):
    """Simulate a Voronoi partition of cells in a disk.
     
    Arguments:
        ncells: int -- number of cells in the partition.
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    boundary_type = kwargs.get('boundary_type', None)
    regularization = kwargs.get('regularization', None)
    burnin = kwargs.get('burnin', 0)
    r_box = kwargs.get('r_box', 2)
    r_emb = kwargs.get('r_emb', 1)
    emb_center = kwargs.get('emb_center', [0, 0])
    fix_cells = kwargs.get('fix_cells', None)
    nonlinearity = kwargs.get('nonlinearity', None)
    saverate = kwargs.get('saverate', 1)
    show_pbar = kwargs.get('show_pbar', True)
    use_gpu = kwargs.get('use_gpu', False)
    outdir = kwargs.get('outdir', "out/sims/disk")
    verbosity = kwargs.get('verbosity', 0)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if outdir: 
        os.makedirs(outdir, exist_ok=True)

    print('ncells:', ncells)
    print('nsteps:', nsteps)
    print('dt:', dt)
    print('saverate:', saverate)
    print('r_box:', r_box)
    print('r_emb:', r_emb)
    print('emb_center:', emb_center)
    print('regularization:', regularization)
    print('boundary_type:', boundary_type)
    print('use_gpu:', use_gpu)
    
    xp = np if not use_gpu else cp

    if boundary_type is None or boundary_type.lower() == 'none':
        boundary_type = None
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

    vor = BoundedVoronoi(ncells, boundary_type=boundary_type, 
                         bounding_box = [-r_box, r_box, -r_box, r_box])
    if regularization:
        print("Regularizing Voronoi patterning...", flush=True)
        sys.stdout.flush()
        for i in tqdm(range(regularization), desc="Regularizing", 
                      disable=not show_pbar):
            vor = vor.recenter()
        print("Regularizing complete.", flush=True)
        sys.stdout.flush()

    locations = vor.get_points()

    # Determine inner and outer cells
    inner_cells = np.linalg.norm(locations - emb_center, axis=1) <= r_emb
    outer_cells = np.linalg.norm(locations - emb_center, axis=1) > r_emb

    alphas = [inner_cells * ai + outer_cells * ao 
              for ai, ao in zip(alphas_inner, alphas_outer)]
    betas = [inner_cells * ai + outer_cells * ao 
             for ai, ao in zip(betas_inner, betas_outer)]
    nus = [inner_cells * ai + outer_cells * ao 
           for ai, ao in zip(nus_inner, nus_outer)]
    
    sa = xp.array(inner_cells * sa_inner + outer_cells * sa_outer)
    si = xp.array(inner_cells * si_inner + outer_cells * si_outer)

    def nonlinearity(data, idx):
        if idx == 0:
            return sa * data[0]*data[0] \
                / (1 + ki * data[1]) \
                / (1 + ka * data[0]*data[0])
        elif idx == 1:
            return si * data[0] * data[0]

    emb = TopologicalEmbryoid(
        ncells, vor.get_adjacency_matrix(), fields=ic,
        diffusivities=nus, alphas=alphas, betas=betas,
        boundary_idx=boundary_idx, locations=locations,
        nonlinearity=nonlinearity, use_gpu=use_gpu,
    )

    # if fix_cells:
    #     emb.fix_cells(vor.get_edge_cells(1), 100, 0)
    #     emb.fix_cells(vor.get_edge_cells(2), 100, 1)
    
    print("Running simulation...", flush=True)
    sys.stdout.flush()
    # history = np.empty([nsteps, *emb.get_fields().shape], dtype=np.float32)
    plt_history = [] #[emb.get_fields().copy()]
    plt_times = [] #[0]
    t = 0
    time0 = time.time()
    time1 = time0
    for i in tqdm(range(nsteps), desc="Simulating", disable=not show_pbar):
        t += dt
        emb.step(dt=dt)
        # history[i] = emb.get_fields().astype(np.float32)
        if t >= burnin and (i+1) % saverate == 0:
            if verbosity > 0:
                ctime = time.time()
                elapsed = ctime - time0
                avg_iter_time = (ctime - time1) / saverate
                time1 = ctime
                print(f"[iter {i+1}/{nsteps}]\t t={t:.5g}")
                print(f"\tTotal elapsed time: {elapsed:.5g} sec")
                print(f"\tAverage time per iteration: {avg_iter_time:.5g} sec")
                sys.stdout.flush()
            # plt_history.append(history[i])
            plt_history.append(emb.get_fields().astype(np.float32))
            plt_times.append(t)
    plt_history = np.array(plt_history)
    print(f"Simulation complete. Elasped time: {time.time()-time0:.5g} sec")

    print("Plotting...", flush=True)
    sys.stdout.flush()
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
    
    print("Done!", flush=True)


def main():
    args = parse_args()
    ncells = args.ncells
    tfin = args.tfin
    dt = args.dt
    saverate = args.saverate
    boundary_type = args.boundary_type
    regularization = args.regularization
    outdir = args.outdir
    verbosity = args.verbosity
    use_gpu = args.use_gpu
    
    # Number of nodes
    n = ncells

    # Specify initial condition
    # ic = np.ones([2, n]) + 0.1 * np.random.randn(2, n)
    ic = np.random.rand(2, n)

    # Dish and embryoid radii
    r_box = 1.2
    r_emb = 1
    emb_center = [0, 0]

    # Define model parameters
    nus_inner = [0.014, 0.4]
    nus_outer = [0.014, 0.4]
    alphas_inner = [0.0, 0.0]
    alphas_outer = [0.0, 0.0]
    betas_inner = [0.001, 0.008]
    betas_outer = [0.01, 0.01]
    sa_inner = 0.01
    sa_outer = 0.0
    si_inner = 0.01
    si_outer = 0.0
    ka = 0
    ki = 1

    nsteps = int(tfin/dt)
    run_voronoi_simulation(
        ncells, nus_inner, nus_outer, alphas_inner, alphas_outer, betas_inner, 
        betas_outer, sa_inner, sa_outer, si_inner, si_outer, ka, ki, 
        r_emb=r_emb, r_box=r_box, emb_center=emb_center,
        ic=ic, dt=dt, nsteps=nsteps, saverate=saverate,
        boundary_type=boundary_type, 
        regularization=regularization, 
        show_pbar=False,
        verbosity=verbosity,
        outdir=outdir, 
        use_gpu=use_gpu
    )

if __name__ == "__main__":
    main()
