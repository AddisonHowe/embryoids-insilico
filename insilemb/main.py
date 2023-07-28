import numpy as np
from insilemb.embryoids import TopologicalEmbryoid
from insilemb.pl import plot_embryoid_cells

def main(): 
    ncells = 8
    nnodes = ncells + 1  # include the boundary
    edges = [
        [0, 1], [0, 3], [0, 8],
        [1, 2], [1, 3], [1, 8],
        [2, 4], [2, 8],
        [3, 4], [3, 5],
        [4, 7], [4, 8],
        [5, 6], [5, 8],
        [6, 7], [6, 8],
        [7, 8],
    ]
    adj = np.zeros([nnodes, nnodes], dtype=int)
    for e in edges:
        adj[e[0], e[1]] = 1
        adj[e[1], e[0]] = 1

    emb = TopologicalEmbryoid(ncells, adj)

    vals = np.random.rand(nnodes)

    coords = [
        [0, 0],
        [5, 0],
        [10, 0],
        [15, 0],
        [20, 0],
        [25, 0],
        [30, 0],
        [35, 0],
    ]
    
    plot_embryoid_cells(emb, coords=coords, show=True)


if __name__ == "__main__":
    main()