"""
Voronoi simulation.
"""

import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import scipy as sp
from insilemb.embryoids import TopologicalEmbryoid
from insilemb.pl import plot_bounded_voronoi
from .bounded_voronoi import BoundedVoronoi

DO_PLOT = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--npoints', type=int, default=1000)
    parser.add_argument('-bt', '--boundary_type', type=str, default=None)
    parser.add_argument('-m', '--mirroring', type=str, default='bbox')
    parser.add_argument('-b', '--bbox', type=float, nargs=4, 
                        default=[0, 1, 0, 1])
    parser.add_argument('-i', '--maxiter', type=int, default=16)
    parser.add_argument('-r', '--saverate', type=int, default=4)
    parser.add_argument('-t', '--tol', type=float, default=1e-4)
    parser.add_argument('-o', '--outdir', type=str, default="out/voronoi/")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    imgdir = args.outdir
    maxiter = args.maxiter
    tol = args.tol
    ncells = args.npoints
    mirroring = args.mirroring
    boundary_type = args.boundary_type
    bounding_box = np.array(args.bbox) # [x_min, x_max, y_min, y_max]
    seed = args.seed

    tfin = 20
    dt = 0.01
    nsteps = int(tfin/dt)
    saverate = 10
    cmap = 'jet'

    os.makedirs(imgdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    points = rng.random([ncells, 2])

    if boundary_type == "ambient":
        boundary_idx = -1
        n = ncells + 1
    else:
        boundary_idx = None
        n = ncells

    # Create Bounded Voronoi partition
    vor = BoundedVoronoi(
        points, 
        mirroring=mirroring,
        bounding_box=bounding_box,
        boundary_type=boundary_type,
    )
    
    vor.plot_2d()
    plt.savefig(f"{imgdir}/mirrored_voronoi_initial.png")
    plot_bounded_voronoi(vor, bbox=bounding_box, 
                         saveas=f"{imgdir}/bounded_voronoi_0.png")
    
    # Get the centroids of the filtered regions
    centroids = vor.get_centroids()
    dmax = np.linalg.norm(centroids - vor.get_points(), axis=1).max()

    iter = 0
    while iter < maxiter and dmax >= tol:
        # New points are the previous centroids
        points = centroids
        vor = vor.recenter()
        if iter % 100 == 0:
            plot_bounded_voronoi(
                vor, bbox=bounding_box, 
                saveas=f"{imgdir}/bounded_voronoi_{iter}.png"
            )
        # Get the centroids of the filtered regions
        centroids = vor.get_centroids()
        dmax = np.linalg.norm(centroids - vor.get_points(), axis=1).max()
        iter += 1

    if dmax < tol:
        print(f"Reached tol={tol:.3g} after {iter} iterations.")
    else:
        print(f"Failed to reach tol={tol:.3g} after {iter} iterations." + \
              f"\n\t max distance between center and centroid: {dmax:.5e}")

    vor.plot_2d()
    plt.savefig(f"{imgdir}/mirrored_voronoi_final.png")
    plot_bounded_voronoi(
        vor, bbox=bounding_box, 
        saveas=f"{imgdir}/bounded_voronoi_final.png"
    )

    adj = vor.get_adjacency_matrix()

    nu = 2
    alpha = 0
    beta = 0
    ic = 1 + 0.1 * np.random.randn(n)
    ic[vor.get_edge_cells()] = 100

    # locations = vor.filtered_points
    emb = TopologicalEmbryoid(
        ncells, adj, 
        boundary_idx=boundary_idx,
        locations=centroids,
        data=ic,
        diffusivities=[nu], alphas=[alpha], betas=[beta],
    )

    print("Running simulation...")
    history = np.empty([nsteps, *emb.get_data().shape])
    plt_history = [emb.get_data()]
    plt_times = [0]
    t = 0
    edge_idxs = vor.get_edge_cells()
    for i in range(nsteps):
        t += dt
        emb.step(dt=dt)
        emb.data[0,edge_idxs] = ic[edge_idxs]
        history[i] = emb.get_data(0)
        if (i+1) % saverate == 0:
            plt_history.append(history[i])
            plt_times.append(t)
            print(history[i].sum(), history[i].min(), history[i].max())
    
    if DO_PLOT:
        print("Plotting...")
        norm = Normalize(vmin=history.min(), vmax=history.max())
        for i, data in enumerate(plt_history):
            t = plt_times[i]
            ax = plot_bounded_voronoi(
                vor, bbox=bounding_box, saveas=None,
                plot_points=False, plot_centroids=False, plot_vertices=False)
            sc = ax.scatter(
                centroids[:,0], centroids[:,1], 
                s=8, c=data[0], norm=norm, cmap=cmap
            )
            ax.set_title(f"$t={t:.4g}$")
            fig = ax.get_figure()
            fig.colorbar(sc, ax=ax, fraction=0.015)
            plt.savefig(f"{imgdir}/img{str(i).zfill(len(str(nsteps)))}.png")
            plt.close()
        
        print("Done!")
    

if __name__ == "__main__":
    main()
