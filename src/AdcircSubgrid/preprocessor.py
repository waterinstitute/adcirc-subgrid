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
import warnings
from typing import Optional

import numpy as np
import rasterio as rio
from numba import njit
from rasterio import features
from scipy.constants import g as g_constant

from .calculation_levels import CalculationLevels
from .input_file import InputFile
from .jit_helpers import nan_mean_jit, nan_sum_jit, reciprocal_jit
from .lookup_table import LookupTable
from .mesh import Mesh
from .progress_bar import ProgressBar
from .raster import Raster
from .raster_region import RasterRegion
from .subgrid_data import SubgridData
from .subgrid_output_file import SubgridOutputFile

logger = logging.getLogger(__name__)

# We need to suppress the numpy empty slice warning. The numpy package
# issues warnings for slices that are all nan, but our slices aren't
# empty, they're all nan, which is completely valid. We could check the
# array for validity, however, that would be slow. And slow is bad.
warnings.filterwarnings("ignore", message="Mean of empty slice")


class SubgridPreprocessor:
    """
    The SubgridPreprocessor class is responsible for preprocessing the subgrid data
    into a format that can be used by the ADCIRC model
    """

    # For areas that are dry, we compute the corrections
    # on an 8cm water depth
    DRY_PIXEL_WATER_DEPTH = 0.08

    # Minimum quadratic friction coefficient
    MIN_CF = 0.0025

    # Valid pixel counts
    # These are the number of valid pixels required to proceed with the calc
    N_VALID_PIXELS_DEM = (
        10  # Minimum number of valid pixels in a sub area for the DEM raster
    )
    N_VALID_PIXELS_LULC = (
        10  # Minimum number of valid pixels in a sub area for the LULC raster
    )
    N_UNIQUE_VALUES_DEM = (
        5  # Minimum number of unique pixels in a sub area for the DEM raster
    )
    N_UNIQUE_VALUES_LULC = (
        1  # Minimum number of unique pixels in a sub area for the LULC raster
    )

    def __init__(self, config: InputFile, max_memory: int) -> None:
        """
        Initialize the SubgridPreprocessor class

        Args:
            config: Configuration dictionary (parsed from YAML)
            max_memory: Approximate maximum memory in MB to use for a single raster window
            subgrid_table: Existing subgrid table to be added to
        """
        self.__config = config
        self.__max_memory = max_memory
        self.__adcirc_mesh = Mesh(config.data()["input"]["adcirc_mesh"])
        self.__lulc_lut = LookupTable(config.data()["input"]["manning_lookup"])
        self.__dem_raster = Raster(config.data()["input"]["dem"])
        self.__lulc_raster = Raster(
            config.data()["input"]["land_cover"], lookup_table=self.__lulc_lut
        )
        self.__check_raster_projection()

        # Read the existing subgrid table if specified
        if config.data()["options"]["existing_subgrid"] is None:
            self.__output = SubgridData(
                self.__adcirc_mesh.num_nodes(),
                self.__config.data()["options"]["n_subgrid_levels"],
                self.__config.data()["options"]["n_phi_levels"],
            )
        else:
            subgrid_table = SubgridOutputFile.read(
                config.data()["options"]["existing_subgrid"]
            )
            self.__output = subgrid_table

        self.__calculation_levels = CalculationLevels(
            self.__config.data()["options"]["subgrid_level_distribution"],
            self.__config.data()["options"]["n_subgrid_levels"],
            self.__config.data()["options"]["distribution_factor"],
            self.DRY_PIXEL_WATER_DEPTH,
        )

        self.__processing_windows = self.__dem_raster.generate_windows(
            max_memory, self.__compute_overlap_size()
        )

        logger.info(
            "Subgrid parameters will be computed on {} levels".format(
                self.__config.data()["options"]["n_subgrid_levels"]
            )
        )
        logger.info(
            "Subgrid parameters will be written to {} phi levels".format(
                self.__config.data()["options"]["n_phi_levels"]
            )
        )

    def __check_raster_projection(self) -> None:
        """
        The raster projections should match so that we can more easily interpolate between them

        The main issue is that when the projections are different, the coordinates do not fall
        on the same grid and the interpolation between them becomes quite expensive
        """
        if (
            self.__dem_raster.coordinate_system()
            != self.__lulc_raster.coordinate_system()
        ):
            msg = "The DEM and land cover rasters have different coordinate systems"
            raise RuntimeError(msg)

    def max_memory(self) -> int:
        """
        Get the maximum memory in MB to use for a single raster window
        """
        return self.__max_memory

    def __compute_overlap_size(self) -> float:
        """
        The overlap size is the maximum width or height of the bounding box of a node's polygon.
        We use this rather than some arbitrary overlap size to ensure that we don't miss any nodes

        Note that the overlap is only computed from nodes that are fully within the raster's bounding box

        Returns:
            The overlap size in raster units
        """
        x_min = self.__dem_raster.bounds()[0]
        y_min = self.__dem_raster.bounds()[1]
        x_max = self.__dem_raster.bounds()[2]
        y_max = self.__dem_raster.bounds()[3]

        overlap_size = 0
        for poly in self.__adcirc_mesh.subarea_polygons().polygons():
            if (
                poly.bounds[0] < x_min
                or poly.bounds[1] < y_min
                or poly.bounds[2] > x_max
                or poly.bounds[3] > y_max
            ):
                continue

            dp = 1.1 * max(
                poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]
            )
            overlap_size = max(overlap_size, dp)

        return overlap_size

    def process(self) -> None:
        """
        Process the subgrid preprocessor
        """
        self.__process_mesh_nodes()

    def write(self) -> None:
        """
        Write the output to a file
        """
        from .subgrid_output_file import SubgridOutputFile

        filename = self.__config.data()["output"]["filename"]

        logger.info(f"Writing output to {filename}")
        SubgridOutputFile.write(self.__output, self.__adcirc_mesh, filename)

    def __find_nodes_in_window(self) -> tuple[np.ndarray, int]:
        """
        Find the raster window that contains the subgrid polygon
        """
        from shapely.geometry import Polygon

        logger.info("Finding nodes in raster windows")

        nodes_found = 0
        node_window_index = np.full(self.__adcirc_mesh.num_nodes(), -1, dtype=int)
        for node_idx, poly in enumerate(
            self.__adcirc_mesh.subarea_polygons().polygons()
        ):
            vertex_status = self.__output.vertex_flag()[node_idx]

            # Skip nodes which have already been computed (i.e. from a prior run)
            if vertex_status == 1:
                continue
            else:
                node_x = self.__adcirc_mesh.nodes()[node_idx][0]
                node_y = self.__adcirc_mesh.nodes()[node_idx][1]

                for window_idx, window in enumerate(self.__processing_windows):
                    # Do the fast check - are we in the bounding box?
                    if (
                        node_x < window.xll()
                        or node_x > window.xur()
                        or node_y < window.yll()
                        or node_y > window.yur()
                    ):
                        continue

                    # Now, do the more expensive check - are we in the polygon?
                    if window.contains(Polygon(poly)):
                        node_window_index[node_idx] = window_idx
                        nodes_found += 1
                        break

        logger.info(
            f"Found {nodes_found} of {self.__adcirc_mesh.num_nodes()} nodes "
            f"({nodes_found / self.__adcirc_mesh.num_nodes() * 100:.2f}%) within the raster"
        )

        return node_window_index, nodes_found

    def __process_mesh_nodes(self) -> None:
        """
        Process the mesh nodes
        """
        node_index, node_count = self.__find_nodes_in_window()

        progress = ProgressBar(
            node_count, self.__config.data()["output"]["progress_bar_increment"], logger
        )

        for window_idx, window in enumerate(self.__processing_windows):
            logger.debug(
                f"Processing raster window {window_idx + 1} of {len(self.__processing_windows)}"
            )

            self.__process_raster_window(window_idx, window, node_index, progress)

    def __process_raster_window(
        self,
        window_index: int,
        window: RasterRegion,
        node_index: np.ndarray,
        progress: ProgressBar,
    ) -> None:
        """
        Process the nodes within the given raster window

        Args:
            window_index: The index of the raster window
            window: The raster window
            node_index: The index for the window that each node falls into
            progress: The progress bar
        """
        # If there is at least one node in the window, read the data from the raster
        # otherwise, we can skip
        if not np.any(node_index == window_index):
            return

        window_data = self.__read_window_data(window)

        # Create a list of the indices of the nodes that fall into the window
        for node in np.where(node_index == window_index)[0]:
            result = self.__process_node(window, window_data, node)

            if result is not None:
                self.__output.add_vertex(
                    node,
                    result["wse_levels"],
                    result["wet_fraction"],
                    result["dp_wet"],
                    result["dp_tot"],
                    result["c_f"],
                    result["c_bf"],
                    result["c_adv"],
                    result["man_avg"],
                )

            progress.increment()

    def __read_window_data(self, window: RasterRegion) -> dict:
        """
        Read the data from the raster window and interpolate the dem and land use to
        a common grid, returned as a single xarray
        """
        dem_data = self.__dem_raster.get_region(
            window.xll(), window.yll(), window.xur(), window.yur(), target_name="dem"
        )
        lulc_data = self.__lulc_raster.get_region(
            window.xll(),
            window.yll(),
            window.xur(),
            window.yur(),
            target_name="manning_n",
        ).interp_like(dem_data)

        if dem_data is None or lulc_data is None:
            msg = "Error reading data from raster window {window}"
            raise RuntimeError(msg)

        return {
            "dem": dem_data["dem"].to_numpy(),
            "manning_n": lulc_data["manning_n"].to_numpy(),
        }

    def __process_node(
        self, window: RasterRegion, window_data: dict, node_index: int
    ) -> Optional[dict]:
        """
        Process the node within the given raster window

        Args:
            window: The raster window
            window_data: The data from the raster window
            node_index: The index of the node to process
        """
        subset_data = self.__subset_raster_data_for_node(
            node_index, window, window_data
        )

        if subset_data is None:
            return None
        else:
            return self.__compute_subgrid_variables_at_node(subset_data)

    def __subset_raster_data_for_node(
        self, node_index: int, window: RasterRegion, window_data: dict
    ) -> Optional[dict]:
        """
        Subset the raster data to the polygon for the node

        Args:
            node_index: The index of the node
            window: The raster window
            window_data: The data from the raster window

        Returns:
            A dictionary with the subset data for the node
        """
        # Compute the minimum index set for this window that the node would
        # need based on the bounds of its polygon
        sub_window = self.__generate_sub_window(node_index, window)

        # Create a mask for the polygon that represents this node
        node_mask = self.__generate_sub_window_mask(node_index, sub_window)

        # Generate numpy arrays for only the data in the polygon
        subset = self.__subset_raster_arrays_to_stencil(
            node_mask, sub_window, window_data["dem"], window_data["manning_n"]
        )

        # Count the number of valid pixels in the dem and manning arrays
        valid_pixels_dem = np.isfinite(subset["dem"]).sum()
        valid_pixels_manning = np.isfinite(subset["manning_n"]).sum()

        # If there isn't enough data to compute the subgrid variables, then we
        # duck out here and continue to the next node
        if valid_pixels_dem < SubgridPreprocessor.N_VALID_PIXELS_DEM:
            logger.debug(
                f"Node {node_index} has too few valid dem pixels ({valid_pixels_dem}) to compute subgrid variables"
            )
            return None
        elif valid_pixels_manning < SubgridPreprocessor.N_VALID_PIXELS_LULC:
            logger.debug(
                f"Node {node_index} has too few valid lulc pixels ({valid_pixels_manning}) to compute subgrid variables"
            )
            return None

        unique_values_dem = len(
            np.unique(subset["dem"][~np.isnan(subset["dem"])].flatten())
        )
        unique_values_manning = len(
            np.unique(subset["manning_n"][~np.isnan(subset["manning_n"])].flatten())
        )
        if unique_values_dem < SubgridPreprocessor.N_UNIQUE_VALUES_DEM:
            logger.debug(
                f"Node {node_index} has too few unique dem values ({unique_values_dem}) to compute subgrid variables"
            )
            return None
        elif unique_values_manning < SubgridPreprocessor.N_UNIQUE_VALUES_LULC:
            logger.debug(
                f"Node {node_index} has too few unique lulc values ({unique_values_manning}) to compute subgrid variables"
            )
            return None

        return {
            "node": node_index,
            "data": subset,
            "sub_window": sub_window,
        }

    @staticmethod
    def __subset_raster_arrays_to_stencil(
        node_mask: np.ndarray, sub_window: dict, dem: np.ndarray, manning_n: np.ndarray
    ) -> dict:
        """
        Subset the DEM and Manning arrays to the stencil for the node

        Args:
            node_mask: The mask for the node
            sub_window: The sub-window for the node
            dem: The DEM array
            manning_n: The Manning's n array

        Returns:
            An xarray dataset with only the subset data
        """
        dem_subset = np.copy(
            dem[
                sub_window["j_start"] : sub_window["j_end"],
                sub_window["i_start"] : sub_window["i_end"],
            ]
        )
        dem_subset[~node_mask] = np.nan

        manning_subset = np.copy(
            manning_n[
                sub_window["j_start"] : sub_window["j_end"],
                sub_window["i_start"] : sub_window["i_end"],
            ]
        )
        manning_subset[~node_mask] = np.nan

        return {
            "dem": dem_subset,
            "manning_n": manning_subset,
        }

    def __generate_sub_window_mask(
        self, node_index: int, sub_window: dict
    ) -> np.ndarray:
        """
        Generate a mask for the sub-window that represents the polygon for the node

        Args:
            node_index: The index of the node
            sub_window: The sub-window to generate the mask for

        Returns:
            A mask for the sub-window
        """
        return features.geometry_mask(
            geometries=[self.__adcirc_mesh.subarea_polygons().polygons()[node_index]],
            out_shape=(sub_window["j_size"], sub_window["i_size"]),
            transform=sub_window["transform"],
            invert=True,
            all_touched=True,
        )

    def __generate_sub_window(self, node_index: int, window: RasterRegion) -> dict:
        """
        Generate the sub-window for the node based on the raster window. This
        is done so that we can be much more efficient in our stencil generation

        Args:
            node_index: The index of the node
            window: The raster window

        Returns:
            A dictionary with the sub-window information
        """
        node_bounds = (
            self.__adcirc_mesh.subarea_polygons().polygons()[node_index].bounds
        )
        i_start = int((node_bounds[0] - window.xll()) / window.cell_size())
        j_end = int((window.yur() - node_bounds[1]) / window.cell_size())
        i_end = int((node_bounds[2] - window.xll()) / window.cell_size())
        j_start = int((window.yur() - node_bounds[3]) / window.cell_size())
        i_size = i_end - i_start
        j_size = j_end - j_start
        this_transform = rio.transform.from_bounds(
            *node_bounds,
            i_size,
            j_size,
        )

        return {
            "node": node_index,
            "i_start": i_start,
            "j_start": j_start,
            "i_end": i_end,
            "j_end": j_end,
            "i_size": i_size,
            "j_size": j_size,
            "transform": this_transform,
            "i_size_full": window.i_size(),
            "j_size_full": window.j_size(),
            "transform_full": window.affine_transform(),
        }

    def __compute_subgrid_variables_at_node(
        self,
        subset_data: dict,
    ) -> Optional[dict]:
        """
        Compute the subgrid variables at the node

        Args:
            subset_data: The subset data for the node

        Returns:
            A dictionary with the subgrid variables
        """
        wse_levels = self.__calculation_levels.get_levels(subset_data["data"]["dem"])

        depth_info = self.__compute_water_depth_at_levels(
            subset_data["data"]["dem"], wse_levels
        )

        if depth_info is None:
            return None

        default_cf = SubgridPreprocessor.__compute_default_cf(
            subset_data["data"]["manning_n"], SubgridPreprocessor.DRY_PIXEL_WATER_DEPTH
        )

        man_avg = SubgridPreprocessor.__compute_man_avg(
            subset_data["data"]["manning_n"]
        )

        cf, cf_avg = self.__compute_cf_at_levels(
            depth_info["depth"],
            subset_data["data"]["manning_n"],
        )

        rv = self.__compute_rv_at_levels(
            depth_info["depth"], depth_info["depth_avg_wet"], cf
        )

        c_adv = self.__compute_advection_correction(
            depth_info["depth_avg_wet"],
            depth_info["depth"],
            rv,
            cf,
        )

        c_bf = self.__compute_bottom_friction_correction(
            depth_info["depth_avg_wet"], rv
        )

        cf_avg[np.isnan(cf_avg)] = default_cf
        c_bf[np.isnan(c_bf)] = default_cf

        cf_avg[cf_avg < self.MIN_CF] = self.MIN_CF
        c_bf[c_bf < self.MIN_CF] = self.MIN_CF

        return {
            "wse_levels": wse_levels,
            "wet_fraction": depth_info["wet_fraction"],
            "dp_wet": depth_info["depth_avg_wet"],
            "dp_tot": depth_info["depth_avg_tot"],
            "c_f": cf_avg,
            "c_adv": c_adv,
            "c_bf": c_bf,
            "man_avg": man_avg,
        }

    @staticmethod
    def __compute_water_depth_at_levels(
        dem_values: np.ndarray,
        wse_levels: np.ndarray,
    ) -> Optional[dict]:
        """
        Compute the water depth at each water level

        Args:
            dem_values: The DEM values
            wse_levels: The water surface elevation levels

        Returns:
            The mean pixel water depth at each water level
        """
        total_pixels = np.isfinite(dem_values).sum()
        if total_pixels == 0:
            return None

        wet_depth_levels = wse_levels[:, np.newaxis, np.newaxis] - dem_values
        wet_depth_levels[
            wet_depth_levels < SubgridPreprocessor.DRY_PIXEL_WATER_DEPTH
        ] = np.nan
        wet_mask = np.isfinite(wet_depth_levels)
        wet_counts = nan_sum_jit(np.isfinite(wet_depth_levels))
        wet_depth_sum = nan_sum_jit(wet_depth_levels)

        # Compute the mean wet depth and the mean depth
        dp_wet_agg = np.divide(
            wet_depth_sum,
            wet_counts,
            out=np.zeros(wet_depth_levels.shape[0]),
            where=wet_counts != 0,
        )

        dp_tot_agg = wet_depth_sum / float(total_pixels)
        dp_tot_agg[np.isnan(dp_tot_agg)] = 0.0
        dp_wet_agg[np.isnan(dp_wet_agg)] = 0.0

        return {
            "depth": wet_depth_levels,
            "depth_avg_wet": dp_wet_agg,
            "depth_avg_tot": dp_tot_agg,
            "wet_mask": wet_mask,
            "wet_counts": wet_counts,
            "wet_fraction": wet_counts / total_pixels,
            "total_pixels": total_pixels,
        }

    @staticmethod
    @njit
    def __compute_bottom_friction_correction(
        dp: np.ndarray,
        rv: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the bottom friction correction for the subgrid

        Args:
            dp: The mean water depth at each pixel per water level
            rv: The R_v at each water

        Returns:
            The bottom friction correction as a 1D vector
        """
        return dp * rv

    @staticmethod
    @njit
    def __compute_advection_correction(
        dp_avg: np.ndarray,
        depth: np.ndarray,
        rv: np.ndarray,
        cf: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the advection correction for the subgrid

        Args:
            dp_avg: The mean water depth at each pixel per water level
            depth: The water depth at each pixel per water level
            rv: The R_v at each water level
            cf: The friction coefficient at each pixel per water level

        Returns:
            The advection correction as a 1D vector
        """
        c_adv = reciprocal_jit(dp_avg) * nan_mean_jit(depth**2 / cf) * rv
        c_adv[np.isnan(c_adv)] = 1.0

        return c_adv

    @staticmethod
    @njit
    def __compute_default_cf(manning: np.ndarray, dry_pixel_depth: float) -> float:
        """
        Compute the default friction coefficient for the subgrid

        Args:
            manning: The Manning's n values
            dry_pixel_depth: The dry pixel water depth

        Returns:
            The default friction coefficient
        """
        return g_constant * np.nanmean(manning**2) / dry_pixel_depth ** (1.0 / 3.0)

    @staticmethod
    @njit
    def __compute_man_avg(manning: np.ndarray) -> float:
        """
        Compute the grid averaged Manning's n values

        Args:
            manning: The Manning's n values

        Returns:
            The grid averaged Manning's n for a vertex area
        """
        return np.nanmean(manning)

    @staticmethod
    @njit
    def __compute_rv_at_levels(
        dp: np.ndarray,
        dp_avg: np.ndarray,
        cf: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the R_v at each water level

        Args:
            dp: The water depth at each pixel per water level
            dp_avg: The mean water depth at each pixel per water level
            cf: The friction coefficient at each pixel per water level

        Returns:
            The R_v at each water level as a 2D array
        """
        return (dp_avg / nan_mean_jit(dp**1.5 * np.sqrt(1.0 / cf))) ** 2

    @staticmethod
    @njit
    def __compute_cf_at_levels(
        depth: np.ndarray,
        manning_values: np.ndarray,
    ) -> tuple:
        """
        Compute the friction coefficient at each water level

        Args:
            depth: The water depth at each pixel per water level
            manning_values: The Manning's n values

        Returns:
            The friction coefficient at each water level
        """
        cf = g_constant * manning_values**2 / depth ** (1.0 / 3.0)
        cf_avg = nan_mean_jit(cf)

        return cf, cf_avg
