import pytest
import numpy as np
from insilemb.voronoi import BoundedVoronoi
from insilemb.embryoids import TopologicalEmbryoid


class TestBoundedVoronoi:

    bbox = [-0.5, 2.5, -0.5, 2.5]
    points = [
        [0., 2], [1., 2.], [2., 2.],
        [0., 1], [1., 1.], [2., 1.],
        [0., 0], [1., 0.], [2., 0.],
    ]
    bvor = BoundedVoronoi(points, bounding_box=bbox)
    
    def test_points(self):
        assert self.bvor.ncells == 9
        assert self.bvor.get_points().shape == (9, 2)
        for xy in self.points:
            assert np.any(xy == self.bvor.get_points())
    
    def test_regions(self):
        regions = self.bvor.get_regions()
        assert len(regions) == 9

    def test_centroids(self):
        centroids = self.bvor.get_centroids()
        assert np.allclose(self.points, centroids)

    def test_vertices(self):
        vertices = self.bvor.get_vertices()
        assert len(vertices) == 16
        xs = [-0.5, 0.5, 1.5, 2.5]
        expected = []
        for x in xs:
            for y in xs:
                expected.append([x, y])
        for xy in expected:
            assert xy in vertices.tolist()

    def test_ridge_points(self):
        ridge_points = self.bvor.get_ridge_points()
        ridge_point_coords = self.bvor.get_ridge_point_coords()
        assert len(ridge_points) == 24
        assert len(ridge_point_coords) == 24
        expected_ridge_points = [
            [0,1],[1,2],
            [3,4],[4,5],
            [6,7],[7,8],
            [0,3],[3,6],
            [1,4],[4,7],
            [2,5],[5,8],
            [9,0],[12,3],[15,6],
            [2,20],[5,23],[8,26],
            [0,36],[1,37],[2,38],
            [6,33],[7,34],[8,35],
        ]
        for xy in expected_ridge_points:
            assert xy in ridge_points.tolist() or \
                np.flip(xy).tolist() in ridge_points.tolist()

    def test_ridge_vertices(self):
        ridge_vertices = self.bvor.get_ridge_vertices()
        ridge_vertex_coords = self.bvor.get_ridge_vertex_coords()
        assert len(ridge_vertices) == 24
        assert len(ridge_vertex_coords) == 24

    def test_edge_cells(self):
        edge_cells_l = self.bvor.get_edge_cells(1)
        edge_cells_r = self.bvor.get_edge_cells(2)
        edge_cells_d = self.bvor.get_edge_cells(3)
        edge_cells_u = self.bvor.get_edge_cells(4)
        edge_cells_a = self.bvor.get_edge_cells('all')
        idxs_l = [0,3,6]
        idxs_r = [2,5,8]
        idxs_d = [6,7,8]
        idxs_u = [0,1,2]
        idxs_a = [0,1,2,3,5,6,7,8]
        for i in idxs_l:
            assert i in edge_cells_l
        for i in idxs_r:
            assert i in edge_cells_r
        for i in idxs_d:
            assert i in edge_cells_d
        for i in idxs_u:
            assert i in edge_cells_u
        for i in idxs_a:
            assert i in edge_cells_a

    
class TestDiffusion:
    
    bbox = [-0.5, 4.5, -0.5, 4.5]
    points = [
        [0., 4], [1., 4.], [2., 4.], [3., 4.], [4., 4.],
        [0., 3], [1., 3.], [2., 3.], [3., 3.], [4., 3.],
        [0., 2], [1., 2.], [2., 2.], [3., 2.], [4., 2.], 
        [0., 1], [1., 1.], [2., 1.], [3., 1.], [4., 1.],
        [0., 0], [1., 0.], [2., 0.], [3., 0.], [4., 0.],
    ]
    

    def test_diffusion(self):
        ncells = len(self.points)
        vor = BoundedVoronoi(
            self.points, 
            bounding_box=self.bbox,
            boundary_type=None
        )
        ic = np.ones(ncells)
        ic[vor.get_edge_cells()] = 10
        nu = 2
        alpha = 0
        beta = 0
        adj = vor.get_adjacency_matrix()
        emb = TopologicalEmbryoid(
            ncells, adj, boundary_idx=None, data=ic,
            diffusivities=[nu], alphas=[alpha], betas=[beta]
        )
        emb.step(dt=0.1)
        expected = np.array([
            10,  8.2, 8.2, 8.2, 10,
            8.2, 4.6, 2.8, 4.6, 8.2,
            8.2, 2.8, 1.0, 2.8, 8.2,
            8.2, 4.6, 2.8, 4.6, 8.2,
            10,  8.2, 8.2, 8.2, 10,
        ])
        assert np.allclose(emb.get_data()[0], expected)


