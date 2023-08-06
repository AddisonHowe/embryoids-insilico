"""
Topological Embryoid
"""

import numpy as np
import scipy

class TopologicalEmbryoid:
    """A topologically defined embryoid. Each cell in the embroid is indexed.
    Attributes:
        ncells -- number of cells constituting the embryoid
        adj -- adjacency matrix
    """

    def __init__(self, ncells, adj, 
                 data=None, diffusivities=None, alphas=None, betas=None,
                 boundary_idx=-1, locations=None, ndim=2, nonlinearity=None):
        #~~~~~~~~~~~~~~~~~ process kwargs ~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.ndim = ndim  # dimension of the embryo
        self.ncells = ncells  # number of cells in the embryo
        if boundary_idx == -1:
            # Include boundary as a node, with index ncells + 1
            boundary_idx = ncells + 1
            self.boundary_idx = boundary_idx
            self.n = ncells + 1
        elif boundary_idx is None:
            self.boundary_idx = boundary_idx
            self.n = ncells
        else:
            msg = f"Haven't implemented boundary_idx={boundary_idx}."
            raise NotImplementedError(msg)
        # Store topological information
        self.nodes = np.arange(self.n, dtype=int)
        self.adj = scipy.sparse.csr_matrix(adj, dtype=int)
        assert self.adj.shape == (self.n, self.n), \
            f"Bad adj shape. Got: {adj.shape}. Expected: ({self.n}, {self.n})"
        # Process and given data information
        self.data = data
        self.nfields = 0
        self.diffusivities = diffusivities
        self.alphas = alphas
        self.betas = betas
        if self.data is not None:
            if np.ndim(self.data) == 1:
                self.data = self.data[None,:]
            self.nfields = len(self.data)  # number of signals comprising data
            self.data = np.array(self.data)
            assert self.data.shape == (self.nfields, self.n), \
                "Bad shape for data."
        if self.diffusivities is not None: 
            self.diffusivities = np.array(diffusivities)
            assert self.diffusivities.shape == (self.nfields,), \
                "Wrong shape for diffusivities."
        if self.alphas is not None:
            self.alphas = np.array(alphas)
            assert self.alphas.shape == (self.nfields,), \
                "Wrong shape for alphas."
        if self.betas is not None:
            self.betas = np.array(betas)
            assert self.betas.shape == (self.nfields,), \
                "Wrong shape for betas."
        # Spatial information
        self.locations = locations
        if self.locations is not None:
            self.locations = np.array(locations)
            assert self.locations.shape == (self.ncells, self.ndim)
        # Initialize fixed cell rules
        self.fixed_cells = []
        # Handle nonlinear term
        self.nonlinearity = nonlinearity if nonlinearity else lambda d, i: 0

    def __str__(self) -> str:
        # return "TopologicalEmbryoid[ncells={}, bidx={}, fields: {}]".format(
        return "TopologicalEmbryoid[{0}, {1}, {2}]".format(
            f"ncells={self.ncells}", 
            f"bidx={self.boundary_idx}", 
            f"nfields={self.nfields}"
        )

    def __repr__(self) -> str:
        return f"<TopologicalEmbryoid>"
    
    ######################
    ##  Getter Methods  ##
    ######################

    def get_data(self, idx=None):
        return self.data if idx is None else self.data[idx]
    
    def get_locations(self):
        return self.locations

    ######################
    ##  Update Methods  ##
    ######################

    def fix_cells(self, idxs, values, dataidx=None):
        if dataidx is None:
            for k in range(self.nfields):
                rule = (k, idxs, values)
                self.fixed_cells.append(rule)
        elif isinstance(dataidx, int):
            rule = (dataidx, idxs, values)
            self.fixed_cells.append(rule)
        self._set_fixed_cells()

    def _set_fixed_cells(self):
        if self.fixed_cells:
            for dataidx, cellidxs, values in self.fixed_cells:
                self.data[dataidx,cellidxs] = values

    def step(self, dt):
        newdata = np.empty(self.data.shape)
        for i in range(self.nfields):
            newdata[i] = self._update_layer(i, dt)
        self.data = newdata
        self._set_fixed_cells()
        
    def _update_layer(self, idx, dt):
        x = self.data[idx]
        nu = self.diffusivities[idx]
        a = self.alphas[idx]
        b = self.betas[idx]
        laplac = np.sum(
            self.adj.multiply(x) - scipy.sparse.diags(x, 0) * self.adj, 
            axis=1).A1
        nonlinterm = self.nonlinearity(self.data, idx)
        dxdt = a - b*x + nu*laplac + nonlinterm
        return np.maximum(x + dt * dxdt, 0)
    