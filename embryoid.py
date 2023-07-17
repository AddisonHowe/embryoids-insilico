"""
Dedalus script implementing the Gierer-Meinhardt Model.

This script simulates the reaction-diffusion system on a 2-d periodic domain.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python embryoid.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from dedalus.extras.plot_tools import plot_bot_2d
import logging

figkw = {'figsize':(6,4), 'dpi':100}
logger = logging.getLogger(__name__)

outdir = "out"
imgdir = "out/images"

os.makedirs(outdir, exist_ok=True)
os.makedirs(imgdir, exist_ok=True)

# Simulation Parameters
R = 1024
Lx, Ly = R, R

Nx, Ny = 1024, 1024

dealias = 3/2
stop_sim_time = 1000
timestepper = d3.RK222
timestep = 0.05
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Substitutions
x, y = dist.local_grids(xbasis, ybasis)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~  Fields  ~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Signal Fields
s1 = dist.Field(name='s1', bases=(xbasis, ybasis))  # s1: Activator
s2 = dist.Field(name='s2', bases=(xbasis, ybasis))  # s2: Inhibitor

# Cell Fields
cells1 = dist.Field(name='c1', bases=(xbasis, ybasis))  # cells producing s1
cells2 = dist.Field(name='c2', bases=(xbasis, ybasis))  # cells producing s2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~  Initial conditions  ~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

rcell = 16  # cell radius
ncellsx = 20  # number of cells in x dimension
ncellsy = 20  # number of cells in y dimension

gamma0 = 1/4               # cell boundary width factor
gamma = gamma0 * rcell      # cell boundary width
kcell = 1/gamma             # tanh factor

epsilon = gamma

print("dt:", timestep)
print("dx:", Lx / Nx)
print("eps:", epsilon)


# # Add cells to the domain
# cell_centers_x = np.linspace(0, Lx, ncellsx + 2)[1:-1]
# cell_centers_y = np.linspace(0, Ly, ncellsy + 2)[1:-1]
cell_centers_x = np.linspace(0, Lx, ncellsx, endpoint=False)
cell_centers_y = np.linspace(0, Ly, ncellsy, endpoint=False)
cell_centers = np.zeros([ncellsx * ncellsy, 2])

i=0
for cx in cell_centers_x:
    for cy in cell_centers_y:
        cell_centers[i] = (cx, cy)
        i += 1

for (cx, cy) in cell_centers:
    rhos = np.zeros([9, Nx, Ny])
    k = 0
    for i in range(-1, 2):
        xs = x + Lx*i
        for j in range(-1, 2):
            ys = y + Ly*j
            rhos[k] = np.sqrt((xs - cx)**2 + (ys - cy)**2)
            k += 1
    rho = np.min(rhos, axis=0)
    d = rho - rcell
    cells1['g'] += 0.5 * (1 - np.tanh(kcell * d))
    cells2['g'] += 0.5 * (1 - np.tanh(kcell * d))

cells1['g'] = np.minimum(cells1['g'], 1)
cells2['g'] = np.minimum(cells2['g'], 1)
cells1['g'] = np.maximum(cells1['g'], 0)
cells2['g'] = np.maximum(cells2['g'], 0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~  Problem Definition  ~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

d1 = 1e-1       # diffusion constant for s1
d2 = 2          # diffusion constant for s2
r = 0.5
c = 1
a = 0.9

s = 0.708
fp_s1 = s
fp_s2 = s*s/a
p = (a-s*c)/(c*s**3)

problem = d3.IVP([s1, s2], namespace=locals())
problem.add_equation(
    "dt(s1) = d1*lap(s1) + cells1 * (r*(s1*s1 / ((1 + p*s1*s1) * s2) - c*s1))"
)
problem.add_equation(
    "dt(s2) = d2*lap(s2) + cells2 * (r*(s1*s1 - a*s2))"
)
# problem.add_equation("dt(cells1) = 0")
# problem.add_equation("dt(cells2) = 0")

s1['g'] = fp_s1 + 0.05 * np.random.rand(*s1['g'].shape)
s2['g'] = fp_s2 + 0.05 * np.random.rand(*s2['g'].shape)

# Plot grid values
plot = True
if plot:
    ax, _ = plot_bot_2d(cells1, figkw=figkw, title="cells1['g']")
    ax.axis('equal')
    plt.savefig(f"{imgdir}/cells1_grid_0.png")
    
    ax, _ = plot_bot_2d(cells2, figkw=figkw, title="cells2['g']")
    ax.axis('equal')
    plt.savefig(f"{imgdir}/cells2_grid_0.png")
    
    ax, _ = plot_bot_2d(s1, figkw=figkw, title="s1['g']")
    ax.axis('equal')
    plt.savefig(f"{imgdir}/s1_grid_0.png")
    
    ax, _ = plot_bot_2d(s2, figkw=figkw, title="s2['g']")
    ax.axis('equal')
    plt.savefig(f"{imgdir}/s2_grid_0.png")
    

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler(f"{outdir}/snapshots", sim_dt=10, max_writes=10)
snapshots.add_task(s1, name='s1')
snapshots.add_task(s2, name='s2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration - 1) % 100 == 0:
            max_s1 = s1['g'].max()
            max_s2 = s2['g'].max()
            logger.info(
                "Iter={:d}, Time={:e}, dt={:e}, max_s1={:e}, max_s2={:e}".format(
                solver.iteration, solver.sim_time, timestep, max_s1, max_s2)
            )
            if plot:

                ax, _ = plot_bot_2d(s1, figkw=figkw, 
                                    title=f"s1['g'] $t={solver.sim_time:.4g}$")
                ax.axis('equal')
                plt.savefig(f"{imgdir}/s1_grid_{solver.iteration}.png")
                plt.close()
                
                ax, _ = plot_bot_2d(s2, figkw=figkw,
                                    title=f"s2['g'] $t={solver.sim_time:.4g}$")
                ax.axis('equal')
                plt.savefig(f"{imgdir}/s2_grid_{solver.iteration}.png")
                plt.close()
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
