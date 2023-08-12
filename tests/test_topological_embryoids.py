import pytest
import numpy as np
from insilemb.embryoids import TopologicalEmbryoid

ADJ1 = [
    [0,1,0,1,0,1],
    [1,0,1,0,0,1],
    [0,1,0,1,1,1],
    [1,0,1,0,1,1],
    [0,0,1,1,0,1],
    [1,1,1,1,1,0],
]

@pytest.mark.parametrize("ncells, adj, nu, a, b, data0, expected", [
    [5, ADJ1, 
     [1], [0], [0],
     [[3,2,4,0,6,5]],
     [[1,8,1,18,0,0]]
    ],
    [5, ADJ1,
     [1, 1], [0, 1], [0, 0],
     [[3,2,4,0,6,5], [3,2,4,0,6,5]],
     [[1,8,1,18,0,0], [2,9,2,19,0,0]]
    ],
    [5, ADJ1,
     [1, 1, 1], [0, 1, 1], [0, 0, 1],
     [[3,2,4,0,6,5], [3,2,4,0,6,5], [3,2,4,0,6,5]],
     [[1,8,1,18,0,0], [2,9,2,19,0,0], [0,7,0,19,0,0]]
    ],
])
def test_update1(ncells, adj, nu, a, b, data0, expected):
    emb = TopologicalEmbryoid(ncells, adj, fields=data0, diffusivities=nu,
                              alphas=a, betas=b)
    print(emb.adj.todense())
    emb.step(1)
    assert np.allclose(emb.get_fields(), expected), \
        f"Expected:\n{expected}\nGot:\n{emb.get_fields()}"
