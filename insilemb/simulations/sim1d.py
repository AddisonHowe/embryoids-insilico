import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from insilemb.embryoids import TopologicalEmbryoid
import insilemb.pl as pl


def run_1d_simulation(ncells, nu, alpha, beta, ic=None, dt=0.1, nsteps=100):
    """Simulate a linear arrangement of cells.
     
    Arguments:
        ncells: int -- number of cells
    """
    adj = np.diag(np.ones(ncells), k=1) + np.diag(np.ones(ncells), k=-1)
    adj[0,-1] = 1
    adj[-1,0] = 1
    
    if ic is None:
        baselevel = 1
        sigma = 0.1  # noise level for initial conditions
        ic = baselevel * np.ones(ncells + 1) + sigma * np.random.randn(ncells+1) 
    
    locations = np.array([2*np.arange(ncells), np.zeros(ncells)]).T
    
    emb = TopologicalEmbryoid(
        ncells, adj, ic, [nu], [alpha], [beta], 
        locations=locations)
    
    print("Running simulation...")
    history = np.empty([nsteps, *emb.get_data().shape])
    plt_history = [emb.get_data()]
    plt_times = [0]
    t = 0
    for i in range(nsteps):
        t += dt
        emb.step(dt=dt)
        history[i] = emb.get_data(0)
        if i % 10 == 0:
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
            title=f"$t={t:.4g}$",
            saveas=f"out/images/sim1d/img{i}.png"
        )

    print("Done!")

  
def main():
    ncells = 15
    tfin = 10
    dt = 0.01
    nu = 1
    alpha = 0.1
    beta = 0
    ic = np.zeros(ncells + 1)
    ic[ncells//2] = 10    
    nsteps = int(tfin/dt)
    run_1d_simulation(ncells, nu, alpha, beta, ic=ic, dt=dt, nsteps=nsteps)


if __name__ == "__main__":
    main()
    