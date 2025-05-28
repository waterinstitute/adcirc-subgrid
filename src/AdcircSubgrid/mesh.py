# Copyright 2025 The Water Institute of the Gulf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Optional

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from .subarea_polygons import SubareaPolygons

logger = logging.getLogger(__name__)


class Mesh:
    """
    Class to represent an Adcirc mesh and provide utility functions
    """

    ELEMENT_NOT_FOUND = -1
    KDTREE_SEARCH_DEPTH = 3

    def __init__(self, filename: str) -> None:
        """
        Initialize the Mesh class using an Adcirc mesh file

        Args:
            filename: File path to the Adcirc mesh file
        """
        self.__filename: str = filename
        self.__nodes: Optional[np.ndarray] = None
        self.__elements: Optional[np.ndarray] = None
        self.__centroids: Optional[np.ndarray] = None
        self.__read_mesh()
        self.__centroids: np.ndarray = self.__compute_centroids(
            self.__elements, self.__nodes
        )

        # self.__node_neighbor_table: Dict[
        #     str, np.ndarray
        # ] = self.__compute_node_neighbor_table()

        self.__element_neighbor_table: dict = self.__compute_element_neighbor_table()

        self.__subarea_polygons = SubareaPolygons(
            self.__nodes,
            self.__elements,
            self.__centroids,
            self.__element_neighbor_table,
        )

        self.__kdtree: cKDTree = cKDTree(self.__centroids)

    def __read_mesh(self) -> None:
        """
        Read the Adcirc mesh file

        Returns:
            None
        """
        logger.debug("Reading mesh file %s", self.__filename)

        with open(self.__filename) as file:
            _ = file.readline().strip().split()
            sizing_info = file.readline().strip().split()
            num_nodes = int(sizing_info[1])
            num_elements = int(sizing_info[0])

            self.__nodes = np.zeros((num_nodes, 3), dtype=float)
            self.__elements = np.zeros((num_elements, 3), dtype=int)

            for i in range(num_nodes):
                line = file.readline().strip().split()
                self.__nodes[i] = [float(line[1]), float(line[2]), float(line[3])]

            for i in range(num_elements):
                line = file.readline().strip().split()
                self.__elements[i] = [
                    int(line[2]) - 1,
                    int(line[3]) - 1,
                    int(line[4]) - 1,
                ]

    @staticmethod
    @njit
    def __compute_centroids(elements: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """
        Compute the centroids of each element

        Returns:
            The centroids of each element as a numpy array
        """
        centroids = np.zeros((elements.shape[0], 2), dtype=float)
        for i, element in enumerate(elements):
            x1, y1 = nodes[element[0]][:2]
            x2, y2 = nodes[element[1]][:2]
            x3, y3 = nodes[element[2]][:2]

            centroids[i] = [(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3]

        return centroids

    def nodes(self) -> np.ndarray:
        """
        Get the nodes from the mesh

        Returns:
            The nodes from the mesh as a numpy array
        """
        return self.__nodes

    def elements(self) -> np.ndarray:
        """
        Get the elements from the mesh

        Returns:
            The elements from the mesh as a numpy array
        """
        return self.__elements

    def num_nodes(self) -> int:
        """
        Get the number of nodes in the mesh

        Returns:
            The number of nodes in the mesh
        """
        return self.__nodes.shape[0]

    def num_elements(self) -> int:
        """
        Get the number of elements in the mesh

        Returns:
            The number of elements in the mesh
        """
        return self.__elements.shape[0]

    def centroids(self) -> np.ndarray:
        """
        Get the centroids of the elements

        Returns:
            The centroids of the elements as a numpy array
        """
        return self.__centroids

    def subarea_polygons(self) -> SubareaPolygons:
        """
        Get the subarea polygons

        Returns:
            The subarea polygons
        """
        return self.__subarea_polygons

    # def node_neighbor_table(self) -> Dict[str, np.ndarray]:
    #     """
    #     Get the neighbor table
    #
    #     Returns:
    #         The neighbor table as a numpy array
    #     """
    #     return self.__node_neighbor_table

    def element_neighbor_table(self) -> dict[str, np.ndarray]:
        """
        Get the neighbor table

        Returns:
            The neighbor table as a numpy array
        """
        return self.__element_neighbor_table

    def kdtree(self) -> cKDTree:
        """
        Get the kdtree

        Returns:
            The kdtree
        """
        return self.__kdtree

    @staticmethod
    @njit
    def tri_area(
        x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
    ) -> float:
        """
        Calculate the area of a triangle

        Args:
            x1: The x-coordinate of the first point
            y1: The y-coordinate of the first point
            x2: The x-coordinate of the second point
            y2: The y-coordinate of the second point
            x3: The x-coordinate of the third point
            y3: The y-coordinate of the third point

        Returns:
            The area of the triangle
        """
        return 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    def is_inside_element(self, element: int, x: float, y: float) -> bool:
        """
        Check if a point is inside an element

        Args:
            element: The element to check
            x: The x-coordinate of the point
            y: The y-coordinate of the point

        Returns:
            True if the point is inside the element, False otherwise
        """
        return Mesh.__is_inside_element_impl(
            self.__nodes, self.__elements, element, x, y
        )

    @staticmethod
    def __is_inside_element_impl(
        nodes: np.ndarray, elements: np.ndarray, element: int, x: float, y: float
    ) -> bool:
        """
        Check if a point is inside an element

        Args:
            nodes: The nodes of the mesh
            elements: The elements of the mesh
            element: The element to check
            x: The x-coordinate of the point
            y: The y-coordinate of the point

        Returns:
            True if the point is inside the element, False otherwise
        """
        x1, y1 = nodes[elements[element][0]][:2]
        x2, y2 = nodes[elements[element][1]][:2]
        x3, y3 = nodes[elements[element][2]][:2]

        a_total = Mesh.tri_area(x1, y1, x2, y2, x3, y3)
        subarea_1 = Mesh.tri_area(x, y, x2, y2, x3, y3)
        subarea_2 = Mesh.tri_area(x1, y1, x, y, x3, y3)
        subarea_3 = Mesh.tri_area(x1, y1, x2, y2, x, y)

        return np.abs(a_total - (subarea_1 + subarea_2 + subarea_3)) < 1e-8

    def find_element(self, x: float, y: float) -> int:
        """
        Find the element containing a point

        We use the kdtree of element centroids to quickly get ourselves
        in the correct region of the mesh, and then we check a few elements.
        In general, the first element we check will be the correct one, but we can't
        rule out bad geometries, so we check a few.

        Args:
            x: The x-coordinate of the point
            y: The y-coordinate of the point

        Returns:
            The element containing the point
        """
        _, nearest_pts = self.__kdtree.query([x, y], k=Mesh.KDTREE_SEARCH_DEPTH)
        for e_idx in np.array(nearest_pts):
            if self.is_inside_element(e_idx, x, y):
                return e_idx
        return Mesh.ELEMENT_NOT_FOUND

    def __compute_node_neighbor_table(self) -> dict[str, np.ndarray]:
        """
        Compute the neighbor table and the count of neighbors for each node
        and return in a dict with the structure:
        {
            "neighbors": np.ndarray,
            "neighbor_count": np.ndarray
        }

        Returns:
            A dictionary with the neighbor table and the count of neighbors for each node
        """
        neighbor_table = Mesh.__compute_node_neighbor_table_array(
            self.__elements, self.num_nodes()
        )
        return {
            "neighbors": neighbor_table,
            "neighbor_count": np.sum(neighbor_table != -1, axis=1),
        }

    @staticmethod
    @njit
    def __compute_node_neighbor_table_array(
        elements: np.ndarray, node_count: int
    ) -> np.ndarray:
        """
        Compute a table of neighbors for each node

        Args:
            elements: The elements of the mesh
            node_count: The number of nodes in the mesh

        Returns:
            A numpy array with the neighbor table
        """
        # Create an over-allocated neighbor table array (20 neighbors per node)
        # If you have more then 20 neighbors, your mesh is bad, and you
        # should feel shame. No subgrid for you.
        neighbor_table = np.full((node_count, 20), -1, dtype=int)

        for element in elements:
            for i in range(3):
                node = element[i]
                next_node = element[(i + 1) % 3]

                # Find the first empty spot in the neighbor table
                empty_spot = np.where(neighbor_table[node] == -1)[0][0]

                # Add the next node to the neighbor table
                neighbor_table[node][empty_spot] = next_node

        # Find the maximum number of neighbors
        max_neighbors = np.max(np.sum(neighbor_table != -1, axis=1))

        # Compact the neighbor table
        return neighbor_table[:, :max_neighbors]

    def __compute_element_neighbor_table(self) -> dict[str, np.ndarray]:
        """
        Compute elements that each node participates in and return in a dict with the structure:
        {
            "neighbors": np.ndarray,
            "neighbor_count": np.ndarray
        }

        Returns:
            A dictionary with the neighbor table and the count of neighbors for each element
        """
        neighbor_table = Mesh.__compute_element_neighbor_table_array(
            self.__elements, self.num_nodes(), self.centroids()
        )
        return {
            "neighbors": neighbor_table,
            "neighbor_count": np.sum(neighbor_table != -1, axis=1),
        }

    @staticmethod
    def __compute_element_neighbor_table_array(
        elements: np.array, node_count: int, centroids: np.ndarray
    ) -> np.ndarray:
        """
        Compute a table of neighbors for each element

        Args:
            elements: The elements of the mesh
            node_count: The number of nodes in the mesh
            centroids: The centroids of the elements

        Returns:
            A numpy array with the neighbor table
        """
        # Create an over-allocated neighbor table array (20 neighbors per node)
        # If you have more then 20 neighbors, your mesh is bad, and you
        # should feel shame. No subgrid for you.
        neighbor_table = np.full((node_count, 20), -1, dtype=int)

        for element_idx, element in enumerate(elements):
            for i in range(3):
                node = element[i]

                # Find the first empty spot in the neighbor table
                empty_spot = np.where(neighbor_table[node] == -1)[0][0]

                # Add the next node to the neighbor table
                neighbor_table[node][empty_spot] = element_idx

        max_neighbors = np.max(np.sum(neighbor_table != -1, axis=1))

        neighbor_table = neighbor_table[:, :max_neighbors]

        return Mesh.__sort_elements_counterclockwise(centroids, neighbor_table)

    @staticmethod
    def __sort_elements_counterclockwise(
        centroids: np.ndarray, neighbor_table: np.ndarray
    ) -> np.ndarray:
        """
        Sort the elements counterclockwise for each node

        Args:
            centroids: Centroids of the elements
            neighbor_table: Table of neighbors for each node

        Returns:
            The neighbor table with the elements sorted counterclockwise
        """
        for node_idx, node in enumerate(neighbor_table):
            node_elements = node[: np.sum(node != -1)]
            node_centroids = centroids[node_elements]

            # Insert the node's centroid at the beginning of the list
            node_centroids = np.insert(
                node_centroids, 0, centroids[node_elements[0]], axis=0
            )

            # Calculate the angles between the node and the centroids
            angles = np.arctan2(
                node_centroids[:, 1] - centroids[node_idx][1],
                node_centroids[:, 0] - centroids[node_idx][0],
            )
            angles = np.where(angles < 0, angles + 2 * np.pi, angles)

            # Remove the node's centroid from the list
            angles = angles[1:]

            # Sort the angles
            sorted_indices = np.argsort(angles)

            # Reorder the neighbors based on the sorted indices and pad to the max number of neighbors
            neighbor_table[node_idx][: len(sorted_indices)] = node_elements[
                sorted_indices
            ]

        return neighbor_table
