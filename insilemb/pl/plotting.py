import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

def plot_embryoid_cells(emb, data=None, **kwargs):
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    cell_radius = kwargs.get('cell_radius', 2)
    cell_boundarycolor = kwargs.get('cell_boundarycolor', 'k')
    edgewidth = kwargs.get('edgewidth', 2)
    coords = kwargs.get('coords', None)
    figsize = kwargs.get('figsize', (8, 5))
    ax = kwargs.get('ax', None)
    title = kwargs.get('title', None)
    saveas = kwargs.get('saveas', None)
    show = kwargs.get('show', False)
    cmap = kwargs.get('cmap', 'viridis')
    norm = kwargs.get('norm', None)
    data_idx = kwargs.get('data_idx', 0)
    boundary_type = kwargs.get('boundary_type', None)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get positions of cells
    if coords is None:
        coords = emb.get_locations()
        if coords is None:
            coords = 10 * np.random.rand(emb.ncells, emb.ndim)

    # Get data to plot
    if data is None:
        data = emb.get_data(data_idx)
    
    if norm is None:
        norm = Normalize(vmin=data.min(), vmax=data.max())
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(norm(data))

    circles = []
    for i, xy in enumerate(coords):
        draw_circle(xy, cell_radius, ax, 
                    color=cell_boundarycolor, linewidth=edgewidth)
        circle = plt.Circle(xy, cell_radius, color=colors[i])
        circles.append(circle)

    s  = f"cell min: {data[:-1].min():.4g}\n"
    s += f"cell max: {data[:-1].max():.4g}"
    
    if boundary_type == "ambient":
        cx = coords[:,0].min() - 2 * cell_radius
        cy = coords[:,1].min() - 2 * cell_radius
        w = coords[:,0].max() - coords[:,0].min() + 4 * cell_radius
        h = coords[:,1].max() - coords[:,1].min() + 4 * cell_radius
        bounding_patch = plt.Rectangle((cx, cy), w, h, color=colors[-1])
        ax.add_patch(bounding_patch)
        s += f"\nboundary: {data[-1]:.4g}"
    
    ax.text(1.01, 0, s, fontsize='x-small', transform=ax.transAxes)
   
    for c in circles:
        ax.add_patch(c)

    p = PatchCollection(circles, cmap=cmap, norm=norm, alpha=0.9)
    fig.colorbar(p, ax=ax, fraction=0.015)

    if title: ax.set_title(title)
    if saveas: plt.savefig(saveas)
    if show: plt.show()
    plt.close()


def draw_circle(c, r, ax, **kwargs):
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    color = kwargs.get('color', 'k')
    linewidth = kwargs.get('linewidth', 2)
    npoints = kwargs.get('npoints', 100)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    pts = np.ones([npoints, 2]) * c
    thetas = np.linspace(0, 2*np.pi, npoints)
    pts += np.array([r * np.cos(thetas), r * np.sin(thetas)]).T
    ax.plot(pts[:,0], pts[:,1], linewidth=linewidth, color=color)
    ax.axis('equal')
    
    
def plot_bounded_voronoi(vor, **kwargs):
    xlims = kwargs.get('xlims', None)
    ylims = kwargs.get('ylims', None)
    plot_points = kwargs.get('plot_points', True)
    plot_centroids = kwargs.get('plot_centroids', True)
    plot_vertices = kwargs.get('plot_vertices', True)
    plot_data = kwargs.get('plot_data', False)
    plot_edges = kwargs.get('plot_edges', False)
    data = kwargs.get('data', None)
    norm = kwargs.get('norm', None)
    cmap = kwargs.get('cmap', 'jet')
    size = kwargs.get('size', 1)
    title = kwargs.get('title', None)
    saveas = kwargs.get('saveas', None)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
    fig, ax = plt.subplots(1, 1)
    # Plot initial points
    if plot_points:
        ax.plot(vor.get_points()[:, 0], vor.get_points()[:, 1], 'b.')
    # Plot ridges points
    if plot_vertices:
        for region in vor.get_regions():
            vertices = vor.get_vertices(region=region)
            ax.plot(vertices[:, 0], vertices[:, 1], 'go')
    # Plot ridges
    if plot_edges:
        for region in vor.get_regions():
            vertices = vor.get_vertices(region=region+[region[0]])
            # vertices = vor.vertices[region + [region[0]], :]
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
    # Compute and plot centroids
    if plot_centroids:
        centroids = vor.get_centroids()
        ax.plot(centroids[:, 0], centroids[:, 1], 'r.')
    if plot_data:
        sc = ax.scatter(
            vor.get_points()[:,0], vor.get_points()[:,1], 
            s=size, c=data, norm=norm, cmap=cmap
        )
        fig.colorbar(sc, ax=ax, fraction=0.015)
    if xlims is not None: ax.set_xlim(xlims)
    if ylims is not None: ax.set_ylim(ylims)
    if title: 
        ax.set_title(title)
    if saveas:
        plt.savefig(saveas)
    return ax
