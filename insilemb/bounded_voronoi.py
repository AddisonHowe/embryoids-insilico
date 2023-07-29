"""
Bounded Voronoi Class
Adapted from https://stackoverflow.com/questions/28665491/...
    getting-a-bounded-polygon-coordinates-from-voronoi-cells
"""

import numpy as  np
import scipy

class BoundedVoronoi:
    """A bounded Voronoi partition. 
    Defined by a set of points and a bounding box. Points are reflected across
    the edges of the bounding box in order to construct a bounded Voronoi 
    partition. A wrapper class for the scipy.spatial.voronoi class. Provides
    the following attributes:

    Attributes:
        ncells: int - the number of cells making up the bounded region.
        nridges: int - the number of ridges in the Voronoi partition.
        ndim: int - the dimension of the ambient space.
        points: ndarray[float] (ncells, d) - coordinates of primary points.
        vertices: ndarray[float] (nvertices, d) - coordinates of primary 
            Voronoi vertices.
        ridge_points: ndarray[int] (nridges, 2) - indices of points between each 
            Voronoi ridge.
        ridge_vertices: list of lists of int: (nridges, *) - indices of Voronoi 
            vertices defining each Voronoi ridge.
        regions: list of lists of int: (ncells, *) - indices of Voronoi vertices 
            forming each Voronoi region. -1 indicates a vertex in the exterior.
        centroids: ndarray (ncells, d) - coordinates of centroids of regions. 
        adj: ??? - adjacency matrix of the Voronoi partition
    """

    ndim = 2

    def __init__(self, points, **kwargs):
        #~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~~~~~~~~~~~~~~~#
        mirroring = kwargs.get('mirroring', 'bbox')
        bounding_box = kwargs.get('bounding_box', [0, 1, 0, 1])
        boundary_type = kwargs.get('boundary_type', None)
        self.mirroring = mirroring
        self.bounding_box = bounding_box
        self.boundary_type = boundary_type
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # Primary points
        if isinstance(points, int):
            self.ncells = points
            dx = self.bounding_box[1] - self.bounding_box[0]
            dy = self.bounding_box[3] - self.bounding_box[2]
            points = np.random.rand(points, self.ndim)
            points[:,0] = points[:,0] * dx + self.bounding_box[0]
            points[:,1] = points[:,1] * dy + self.bounding_box[2]
            self.points = points
        else:
            self.points = np.array(points)
            self.ncells = len(points)
        # Reflect the primary points
        if mirroring == "bbox" or mirroring == "bounding_box":
            # Check that points are inside the bounding box.
            assert np.all(in_box(self.points, bounding_box)), \
                f"All given points should be inside the box: {bounding_box}."+\
                f"\nGot:\n{self.points}"
            self.bbox = bounding_box
            # Get tuple consisting of mirrored point coordinates.
            mirrored_points, mirror_idxs = self._get_mirrored_points(mirroring)
            self.mirror_idxs = mirror_idxs
        else:
            raise RuntimeError(f"Unknown mirroring scheme: {mirroring}")
        # Combine all points and compute Voronoi pattern
        pts = np.concatenate([*mirrored_points], axis=0)
        vor = scipy.spatial.Voronoi(pts)
        self._vor = vor
        # Save primary regions
        primary_regions = vor.point_region[:self.ncells]
        self.regions = [vor.regions[idxs] for idxs in primary_regions]
        self.centroids = self._compute_centroids()
        self.vertices = self._get_filtered_vertices()
        self.ridge_points, self.ridge_vertices = self._get_filtered_ridges()
        self.nridges = len(self.ridge_points)
        self.adj, self.boundaries = self._compute_adj()
        
    def recenter(self):
        """Return a BoundedVoronoi with points at the current centroids.
        """
        return BoundedVoronoi(
            self.centroids.copy(), 
            mirroring=self.mirroring,
            bounding_box=self.bounding_box,
            boundary_type=self.boundary_type,
        )

    def plot_2d(self):
        scipy.spatial.voronoi_plot_2d(self._vor)

    ######################
    ##  Helper Methods  ##
    ######################

    def _compute_centroids(self):
        centroids = np.empty([self.ncells, self.ndim])
        for i, region in enumerate(self.regions):
            vertices = self._vor.vertices[region + [region[0]], :]
            centroids[i] = centroid_region(vertices)
        return centroids      

    def _get_mirrored_points(self, mirroring):
        """Mirror primary points across edges.
        Args:
            mirroring (str): mirroring scheme
        Returns:
            pts: (k+1)-tuple, where k is the number of reflection edges. Each 
                tuple consists of an ndarray of shape (ncells, ndim). Primary 
                points are the 0th element 0, with mirrored sets following.
            mirror_edge_map: map of point indices 0:(k+1)*ncells mapped to the 
                edgeidx.
        """
        if mirroring == "bbox":
            self.nedges = 4
            # Mirror points
            pts_c = self.points
            pts_l = np.copy(pts_c)
            pts_l[:, 0] = self.bbox[0] - (pts_l[:, 0] - self.bbox[0])
            pts_r = np.copy(pts_c)
            pts_r[:, 0] = self.bbox[1] + (self.bbox[1] - pts_r[:, 0])
            pts_d = np.copy(pts_c)
            pts_d[:, 1] = self.bbox[2] - (pts_d[:, 1] - self.bbox[2])
            pts_u = np.copy(pts_c)
            pts_u[:, 1] = self.bbox[3] + (self.bbox[3] - pts_u[:, 1])
            pts = (pts_c, pts_l, pts_r, pts_d, pts_u)
            # Map primary and reflected cell indices to indicators of which edge
            # they were reflected over.
            mirror_edge_map = {}
            n = self.ncells
            for edgeidx in range(5):
                mirror_edge_map.update(
                    {i:edgeidx for i in range(n*edgeidx, n*(edgeidx+1))}
                )
        else:
            raise NotImplementedError(f"Unknown mirroring scheme: {mirroring}")
        return pts, mirror_edge_map

    def _get_filtered_vertices(self):
        vertices = set()
        for reg in self.regions:
            vertices.update(reg)
        return np.array([self._vor.vertices[vidx] for vidx in vertices])
    
    def _get_filtered_ridges(self):
        ridge_points = []
        ridge_vertices = []
        for i, (p0, p1) in enumerate(self._vor.ridge_points):
            g0 = self.mirror_idxs[p0]
            g1 = self.mirror_idxs[p1]
            if g0 == 0 or g1 == 0:
                ridge_points.append([p0, p1])
                ridge_vertices.append(self._vor.ridge_vertices[i])
        return np.array(ridge_points), np.array(ridge_vertices)
    
    def _compute_adj(self):
        btype = self.boundary_type
        n = self.ncells + 1 if btype == "ambient" else self.ncells
        adj = scipy.sparse.lil_matrix((n, n), dtype=int)
        edges = self.ridge_points
        boundaries = {k: [] for k in range(1, self.nedges + 1)}
        for v0, v1 in edges:
            if v0 >= self.ncells and v1 >= self.ncells:
                # both nodes are exterior, so skip
                pass  
            elif v0 < self.ncells and v1 < self.ncells:
                # both cells in the interior, so connect with an edge
                adj[v0, v1] = 1
            else:  
                vmin = min(v0, v1)
                vmax = max(v0, v1)
                # vmax is an exterior cell, so vmin is connected to the boundary
                boundaries[self.mirror_idxs[vmax]].append(vmin)
                if btype == "ambient":
                    adj[vmin, -1] = 1
                elif btype is None:
                    pass
                else:
                    msg = f"Not sure how to handle boundary type '{btype}'"
                    raise NotImplementedError(msg)
        adj = (adj + adj.T > 0).astype(int)  # make symmetric
        return adj, boundaries
    
    ######################
    ##  Getter Methods  ##
    ######################

    def get_points(self):
        return self.points
    
    def get_regions(self):
        return self.regions

    def get_centroids(self):
        return self.centroids
    
    def get_vertices(self, region=None):
        if region:
            return self._vor.vertices[region,:]
        return self.vertices

    def get_edge_cells(self, edge='all'):
        """Get the indices of primary cells on the edge."""
        if edge == 'all':
            idxs = [self.boundaries[k] for k in self.boundaries]
            return np.concatenate(idxs)
        else:
            return self.boundaries[edge]
    
    def get_ridge_points(self):
        return self.ridge_points
    
    def get_ridge_vertices(self):
        return self.ridge_vertices
    
    def get_ridge_point_coords(self):
        return np.array([
            [self._vor.points[pidx] for pidx in ps] 
            for ps in self.ridge_points
        ])
    
    def get_ridge_vertex_coords(self):
        return np.array([
            [self._vor.vertices[vidx] for vidx in vs] 
            for vs in self.ridge_vertices
        ])
    
    def get_adjacency_matrix(self):
        return self.adj
    

########################
##  Helper Functions  ##
########################

def in_box(points, bounding_box):
    return np.logical_and(
        np.logical_and(bounding_box[0] <= points[:, 0], 
                       points[:, 0] <= bounding_box[1]),
        np.logical_and(bounding_box[2] <= points[:, 1], 
                       points[:, 1] <= bounding_box[3]))

def centroid_region(vertices):
    # Polygon's signed area
    area = 0
    # Centroid's x and y
    cx, cy = 0, 0
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0]     * vertices[i + 1, 1] - \
             vertices[i + 1, 0] * vertices[i, 1])
        area += s
        cx += (vertices[i, 0] + vertices[i + 1, 0]) * s
        cy += (vertices[i, 1] + vertices[i + 1, 1]) * s
    area = 0.5 * area
    cx = (1.0 / (6.0 * area)) * cx
    cy = (1.0 / (6.0 * area)) * cy
    return np.array([cx, cy])