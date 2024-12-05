import logging

import numpy as np

from .input_file import InputFile
from .mesh import Mesh
from .raster import Raster
from .raster_region import RasterRegion

logger = logging.getLogger(__name__)


class SubgridPreprocessor:
    """
    The SubgridPreprocessor class is responsible for preprocessing the subgrid data
    into a format that can be used by the ADCIRC model
    """

    def __init__(self, config: InputFile) -> None:
        """
        Initialize the SubgridPreprocessor class

        Args:
            config: Configuration dictionary (parsed from YAML)
        """
        self.__config = config
        self.__adcirc_mesh = Mesh(config.data()["adcirc_mesh"])
        self.__dem_raster = Raster(config.data()["dem"])
        dynamic_overlap_size = self.__compute_overlap_size()
        self.__processing_windows = self.__generate_raster_windows(
            256, dynamic_overlap_size
        )

    def __compute_overlap_size(self) -> float:
        """
        The overlap size is the maximum width or height of the bounding box of a node's polygon.
        We use this rather than some arbitrary overlap size to ensure that we don't miss any nodes

        Note that the overlap is only computed from nodes that are fully within the raster's bounding box

        Returns:
            The overlap size in raster units
        """
        xmin = self.__dem_raster.bounds()[0]
        ymin = self.__dem_raster.bounds()[3]
        xmax = self.__dem_raster.bounds()[2]
        ymax = self.__dem_raster.bounds()[1]

        overlap_size = 0
        for poly in self.__adcirc_mesh.subarea_polygons().polygons():
            if (
                poly.bounds[0] < xmin
                or poly.bounds[1] < ymin
                or poly.bounds[2] > xmax
                or poly.bounds[3] > ymax
            ):
                continue

            overlap_size = max(overlap_size, poly.bounds[2] - poly.bounds[0])
            overlap_size = max(overlap_size, poly.bounds[3] - poly.bounds[1])

        return overlap_size

    def process(self) -> None:
        """
        Process the subgrid preprocessor
        """
        self.__find_nodes_in_window()

    def __generate_raster_windows(
        self, window_size: int, overlap: float
    ) -> list[RasterRegion]:
        """
        Generate raster windows based on the DEM raster

        Args:
            window_size: The size of the window
            overlap: The overlap between windows
        """
        return self.__dem_raster.generate_windows(window_size, overlap)

    def __find_nodes_in_window(self) -> np.ndarray:
        """
        Find the raster window that contains the subgrid polygon
        """
        logger.info("Finding nodes in raster windows")

        nodes_found = 0
        node_window_index = np.full(self.__adcirc_mesh.num_nodes(), -1, dtype=int)
        for node_idx, poly in enumerate(
            self.__adcirc_mesh.subarea_polygons().polygons()
        ):
            for window_idx, window in enumerate(self.__processing_windows):
                if window.contains(poly):
                    node_window_index[node_idx] = window_idx
                    nodes_found += 1
                    break

        logger.info(
            "Found %d of %d nodes in the raster",
            nodes_found,
            self.__adcirc_mesh.num_nodes(),
        )

        return node_window_index
