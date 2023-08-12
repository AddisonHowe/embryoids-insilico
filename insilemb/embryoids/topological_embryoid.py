"""
Topological Embryoid
"""

import numpy as np
import scipy

class TopologicalEmbryoid:
    """A topologically defined embryoid. Each cell in the embroid is indexed.
    Attributes:
        ncells : number of cells constituting the embryoid.
        adj : adjacency matrix.
        ndim : int : dimension of the embryoid.
        fields : list : 
    """

    def __init__(self, ncells, adj, **kwargs):
        #~~~~~~~~~~~~~~~~~ process kwargs ~~~~~~~~~~~~~~~~~#
        ndim = kwargs.get('ndim', 2)
        fields = kwargs.get('fields', None)
        locations = kwargs.get('locations', None)
        alphas = kwargs.get('alphas', None)
        betas = kwargs.get('betas', None)
        diffusivities = kwargs.get('diffusivities', None)
        boundary_idx = kwargs.get('boundary_idx', -1)
        nonlinearity = kwargs.get('nonlinearity', None)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        # Number and dimension of cells constituting the embryoid
        self.ndim = ndim  # dimension of the embryoid
        self.ncells = ncells  # number of cells in the embryoid
        
        # Process boundary rule
        if boundary_idx == -1:
            # Include boundary as a node, with index ncells + 1
            boundary_idx = ncells + 1
            self.boundary_idx = boundary_idx
            self.n = ncells + 1
        elif boundary_idx is None:
            # No boundary
            self.boundary_idx = boundary_idx
            self.n = ncells
        else:
            msg = f"Haven't implemented boundary_idx={boundary_idx}."
            raise NotImplementedError(msg)
        
        # Topological information
        self.nodes = np.arange(self.n, dtype=int)
        self.adj = scipy.sparse.csr_matrix(adj, dtype=int)
        assert self.adj.shape == (self.n, self.n), \
            f"Bad adj shape. Got: {adj.shape}. Expected: ({self.n}, {self.n})"
        
        # Initialize fields information
        self.fields = fields
        self.nfields = 0
        self.alphas = alphas
        self.betas = betas
        self.diffusivities = diffusivities
        if self.fields is not None:
            if np.ndim(self.fields) == 1:
                self.fields = self.fields[None,:]
            self.nfields = len(self.fields)  # number of signals comprising fields
            self.fields = np.array(self.fields)
            assert self.fields.shape == (self.nfields, self.n), \
                "Bad shape for fields."
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

    def get_fields(self, idx=None):
        return self.fields if idx is None else self.fields[idx]
    
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
            for fieldidx, cellidxs, values in self.fixed_cells:
                self.fields[fieldidx, cellidxs] = values

    def step(self, dt):
        newfields = np.empty(self.fields.shape)
        for i in range(self.nfields):
            newfields[i] = self._update_layer(i, dt)
        self.fields = newfields
        self._set_fixed_cells()
        
    def _update_layer(self, idx, dt):
        x = self.fields[idx]
        nu = self.diffusivities[idx]
        a = self.alphas[idx]
        b = self.betas[idx]
        laplac = np.sum(
            self.adj.multiply(x) - scipy.sparse.diags(x, 0) * self.adj, 
            axis=1).A1
        nonlinterm = self.nonlinearity(self.fields, idx)
        dxdt = a - b*x + nu*laplac + nonlinterm
        return np.maximum(x + dt * dxdt, 0)
    