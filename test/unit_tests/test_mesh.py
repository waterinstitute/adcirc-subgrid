import numpy as np
from AdcircSubgrid.mesh import Mesh
from shapely.geometry import Polygon


def test_mesh() -> None:
    """
    Test the mesh class for adcirc ascii mesh I/O
    """
    mesh_filename = "test/test_files/fort.14"
    mesh = Mesh(mesh_filename)

    assert mesh.num_nodes() == 8303
    assert mesh.num_elements() == 14761
    assert np.allclose(
        mesh.nodes()[3], np.array([-76.3602229888, 39.1975901868, 3.8347489834e00])
    )
    assert np.all(
        mesh.elements()[3] == np.array([8020 - 1, 8036 - 1, 8000 - 1])
    )  # Note we convert to 0-based indexing

    assert np.isclose(mesh.centroids()[3][0], -83.50949621966667)
    assert np.isclose(mesh.centroids()[3][1], 16.27066179943333)

    expected_neighbors = np.array([2541, 8253, 462, 8254, 7422, -1, -1, -1])
    assert np.array_equal(
        mesh.element_neighbor_table()["neighbors"][3], expected_neighbors
    )
    assert mesh.element_neighbor_table()["neighbor_count"][3] == 5

    assert mesh.find_element(-89.39544, 28.930391) == 7276
    assert mesh.find_element(0, 0) == mesh.ELEMENT_NOT_FOUND

    poly = mesh.subarea_polygons().polygon(0)
    assert poly == Polygon(
        [
            [-76.40035379066666, 39.2403234329],
            [-76.42041919159999, 39.26169005595],
            [-76.3689180345, 39.313526649],
            [-76.29028342065, 39.309472034500004],
            [-76.31359661003334, 39.27217808526667],
            [-76.40035379066666, 39.2403234329],
        ]
    )

    poly = mesh.subarea_polygons().polygon(mesh.num_nodes() - 1)
    assert poly == Polygon(
        [
            [-88.6294424012, 15.7518381087],
            [-88.7282586848, 15.809296777499998],
            [-88.70796100813332, 15.833680145666667],
            [-88.61176751403333, 15.826354432099999],
            [-88.58396844365001, 15.79830820715],
            [-88.6294424012, 15.7518381087],
        ]
    )
