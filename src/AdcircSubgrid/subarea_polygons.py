import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from numba import njit
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


@njit
def _compute_subarea_arrays(
    nodes: np.ndarray,
    element_neighbor_table: np.ndarray,
    element_neighbor_table_count: np.ndarray,
    elements: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the subareas for each node

    Subareas are the influence area for each node. It is computed
    as the polygon formed by linking the centroids of the elements
    that contain the node and the midpoints of the edges of the
    elements.

    Returns:
        The subareas for each node as an array of shapely polygons
    """
    array_size = _compute_array_dimensions(
        nodes, element_neighbor_table, element_neighbor_table_count
    )

    poly_pts = np.zeros((array_size, 2), dtype=np.float64)
    poly_pts_label = np.zeros(array_size, dtype=np.int32)
    poly_box = np.zeros((nodes.shape[0], 4), dtype=np.float64)

    count = 0
    for node_idx, _this_node in enumerate(nodes):
        node_elements = element_neighbor_table[node_idx][
            : element_neighbor_table_count[node_idx]
        ]

        this_centroids = centroids[node_elements]
        node_elements = elements[node_elements]

        this_pts, this_label, bounds = __generate_subarea_polygons_for_node(
            node_idx,
            nodes,
            this_centroids,
            node_elements,
        )
        sz = len(this_pts)

        poly_pts[count : count + sz] = this_pts
        poly_pts_label[count : count + sz] = this_label
        poly_box[node_idx] = bounds

        count += sz

    return poly_pts, poly_pts_label, poly_box


@njit
def _compute_array_dimensions(
    nodes: np.ndarray,
    element_neighbor_table: np.ndarray,
    element_neighbor_table_count: np.ndarray,
) -> int:
    """
    Compute the size of the array needed to store the subareas

    Args:
        nodes: Nodes of the mesh
        element_neighbor_table: Neighbors of each element
        element_neighbor_table_count: Number of neighbors for each element

    Returns:
        The size of the array needed to store the subareas points
    """
    counter = 0
    for node_idx, _node in enumerate(nodes):
        node_elements = element_neighbor_table[node_idx][
            : element_neighbor_table_count[node_idx]
        ]
        counter += len(node_elements) * 5
    return counter


@njit
def __generate_subarea_polygons_for_node(
    node_idx: int,
    mesh_nodes: np.ndarray,
    centroids: np.ndarray,
    node_elements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the subarea polygons for a node

    Args:
        node_idx: Index of the node
        mesh_nodes: Nodes of the mesh
        centroids: Centroids of the elements that contain the node
        node_elements: Elements that contain the node


    Returns:
        The subarea polygons for the node

    """
    bounds = np.full(4, np.inf, dtype=float)
    bounds[1] = -np.inf
    bounds[3] = -np.inf

    out_pts = np.zeros((5 * len(node_elements), 2), dtype=float)
    out_label = np.full(5 * len(node_elements), node_idx, dtype=int)

    node = mesh_nodes[node_idx]

    idx = 0
    for element, centroid in zip(node_elements, centroids):
        n1, n2, n3 = element
        if n2 == node_idx:
            n1, n2 = n2, n1
        elif n3 == node_idx:
            n1, n3 = n3, n1

        m2 = ((node + mesh_nodes[n2]) / 2)[:2]
        m3 = ((node + mesh_nodes[n3]) / 2)[:2]

        out_pts[idx] = [node[0], node[1]]
        out_pts[idx + 1] = [m2[0], m2[1]]
        out_pts[idx + 2] = [centroid[0], centroid[1]]
        out_pts[idx + 3] = [m3[0], m3[1]]
        out_pts[idx + 4] = [node[0], node[1]]

        idx += 5

        bounds[0] = np.min(np.array([bounds[0], node[0], m2[0], centroid[0], m3[0]]))
        bounds[1] = np.max(np.array([bounds[1], node[0], m2[0], centroid[0], m3[0]]))
        bounds[2] = np.min(np.array([bounds[2], node[1], m2[1], centroid[1], m3[1]]))
        bounds[3] = np.max(np.array([bounds[3], node[1], m2[1], centroid[1], m3[1]]))

    return out_pts, out_label, bounds


class SubareaPolygons:
    """
    Class which handles the sub-areas around the nodes in a mesh
    """

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        centroids: np.ndarray,
        element_neighbor_table: dict,
    ) -> None:
        """
        Constructor for the SubareaPolygons class

        Computes the subarea polygons for each node in the mesh

        Args:
            nodes: Nodes of the mesh
            elements: Elements of the mesh
            centroids: Element centroids
            element_neighbor_table: Element neighbors of each node
        """
        self.__polygon_dataframe, self.__poly_box = self.__compute_subarea_polygons(
            nodes,
            elements,
            centroids,
            element_neighbor_table,
        )

    def polygons(self) -> gpd.GeoDataFrame:
        """
        Get the subarea polygons

        Returns:
            The subarea polygons as a GeoDataFrame
        """
        return self.__polygon_dataframe

    def bounding_boxes(self) -> np.ndarray:
        """
        Get the bounding boxes for the subarea polygons

        Returns:
            The bounding boxes for the subarea polygons
        """
        return self.__poly_box

    def polygon(self, node_idx: int) -> Polygon:
        """
        Get the subarea polygon for a node

        Args:
            node_idx: Index of the node

        Returns:
            The subarea polygon for the node as a GeoSeries
        """
        return self.__polygon_dataframe.iloc[node_idx]

    def bounding_box(self, node_idx: int) -> np.ndarray:
        """
        Get the bounding box for a node

        Args:
            node_idx: Index of the node

        Returns:
            The bounding box for the
        """
        return self.__poly_box[node_idx]

    @staticmethod
    def __compute_subarea_polygons(
        nodes: np.ndarray,
        elements: np.ndarray,
        centroids: np.ndarray,
        element_neighbor_table: dict,
    ) -> tuple[gpd.GeoDataFrame, np.ndarray]:
        """
        Compute the subarea polygons for each node and return them as a GeoDataFrame

        Args:
            nodes: Nodes of the mesh
            elements: Elements of the mesh
            centroids: Element centroids
            element_neighbor_table: Element neighbors of each node

        Returns:
            A GeoDataFrame containing the subarea polygons for each node
        """
        import time

        tic = time.time()

        poly_pts, poly_pts_label, poly_box = _compute_subarea_arrays(
            nodes,
            element_neighbor_table["neighbors"],
            element_neighbor_table["neighbor_count"],
            elements,
            centroids,
        )

        labels_df = pd.DataFrame({"label": poly_pts_label})

        point_df = gpd.GeoDataFrame(
            labels_df, geometry=gpd.points_from_xy(poly_pts[:, 0], poly_pts[:, 1])
        )

        out_poly = point_df.dissolve(by="label").convex_hull

        toc = time.time()
        logger.info(f"Computed subarea polygons in {toc - tic:.2f} seconds")

        return out_poly, poly_box
