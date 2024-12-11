import logging
import warnings

import numpy as np
import xarray as xr

from .input_file import InputFile
from .lookup_table import LookupTable
from .mesh import Mesh
from .output import SubgridOutput
from .progress_bar import ProgressBar
from .raster import Raster
from .raster_region import RasterRegion

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

    def __init__(self, config: InputFile, max_memory: int = 64) -> None:
        """
        Initialize the SubgridPreprocessor class

        Args:
            config: Configuration dictionary (parsed from YAML)
            max_memory: Maximum memory in MB to use for a single raster window
        """
        self.__config = config
        self.__max_memory = max_memory
        self.__adcirc_mesh = Mesh(config.data()["adcirc_mesh"])
        self.__lulc_lut = LookupTable(config.data()["manning_lookup"])
        self.__dem_raster = Raster(config.data()["dem"])
        self.__lulc_raster = Raster(
            config.data()["land_cover"], lookup_table=self.__lulc_lut
        )
        self.__check_raster_projection()
        self.__output = SubgridOutput(
            self.__adcirc_mesh.num_nodes(),
            self.__config.data()["n_subgrid_levels"],
            self.__config.data()["n_phi_levels"],
        )

        self.__processing_windows = self.__generate_raster_windows(
            max_memory, self.__compute_overlap_size()
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
        y_min = self.__dem_raster.bounds()[3]
        x_max = self.__dem_raster.bounds()[2]
        y_max = self.__dem_raster.bounds()[1]

        overlap_size = 0
        for poly in self.__adcirc_mesh.subarea_polygons().polygons():
            if (
                poly.bounds[0] < x_min
                or poly.bounds[1] < y_min
                or poly.bounds[2] > x_max
                or poly.bounds[3] > y_max
            ):
                continue

            overlap_size = max(overlap_size, poly.bounds[2] - poly.bounds[0])
            overlap_size = max(overlap_size, poly.bounds[3] - poly.bounds[1])

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
        logger.info(f"Writing output to {self.__config.data()['output_filename']}")
        self.__output.write(self.__config.data()["output_filename"])

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
            for window_idx, window in enumerate(self.__processing_windows):
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
            node_count, self.__config.data()["progress_bar_increment"], logger
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

            # Add the results to the output
            self.__output.add_vertex(
                node,
                result["wet_fraction"],
                result["dp"],
                result["dp"],  # TODO: Fix this to be the total depth
                result["cf"],
                result["c_bf"],
                result["c_adv"],
            )

            progress.increment()

    def __read_window_data(self, window: RasterRegion) -> xr.Dataset:
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

        return xr.merge([dem_data, lulc_data])

    def __process_node(
        self, window: RasterRegion, window_data: xr.Dataset, node_index: int
    ) -> dict:
        """
        Process the node within the given raster window

        Args:
            window: The raster window
            window_data: The data from the raster window
            node_index: The index of the node to process
        """
        plot_diagnostic = False

        subset_data = self.__subset_raster_data_for_node(
            node_index, window, window_data
        )

        # Compute the list of water levels where calculations will be performed
        wse_levels = self.__generate_calculation_intervals(subset_data["dem"])

        subgrid_variables = self.__compute_subgrid_variables_at_node(
            subset_data, wse_levels
        )

        if plot_diagnostic:
            self.__plot_diagnostics(
                window,
                subset_data["sub_window"],
                wse_levels,
                subgrid_variables,
            )

        return subgrid_variables

    def __compute_subgrid_variables_at_node(
        self, subset_data: dict, wse_levels: np.ndarray
    ) -> dict:
        """
        Compute the subgrid variables at the node

        Args:
            subset_data: The subset data for the node
            wse_levels: The water surface elevation levels to compute the variables at

        Returns:
            A dictionary with the subgrid variables
        """
        wet_level_counts, wet_masks, total_pixels = self.__compute_wet_pixel_mask(
            subset_data["dem"], wse_levels
        )

        dp, dp_2d = self.__compute_water_depth_at_levels(
            subset_data["dem"], wet_masks, wse_levels
        )

        cf, cf_2d = self.__compute_cf_at_levels(
            dp_2d,
            wet_masks,
            subset_data["manning"],
        )

        rv = self.__compute_rv_at_levels(dp, dp_2d, cf_2d)
        c_adv = self.__compute_advection_correction(
            dp,
            dp_2d,
            rv,
            cf_2d,
        )
        c_bf = self.__compute_bottom_friction_correction(rv, subset_data["manning"])

        # Now, scale the percent of wet pixels
        # NOTE: This previously was done using areas, but I assume since it is just a
        # scaling, this should amount to the same thing
        wet_fraction = wet_level_counts / float(total_pixels)
        c_adv = c_adv * wet_fraction
        c_bf = c_bf * wet_fraction
        cf = cf * wet_fraction

        # We need to set some minimum values for dp, cf, and c_bf
        dp[
            np.logical_or(dp < SubgridPreprocessor.DRY_PIXEL_WATER_DEPTH, np.isnan(dp))
        ] = SubgridPreprocessor.DRY_PIXEL_WATER_DEPTH
        cf[np.logical_or(cf < SubgridPreprocessor.MIN_CF, np.isnan(cf))] = (
            SubgridPreprocessor.MIN_CF
        )
        c_bf[np.logical_or(c_bf < SubgridPreprocessor.MIN_CF, np.isnan(cf))] = (
            SubgridPreprocessor.MIN_CF
        )

        return {
            "wet_level_counts": wet_level_counts,
            "wet_masks": wet_masks,
            "wet_fraction": wet_fraction,
            "dp": dp,
            # "dp_2d": dp_2d,
            "cf": cf,
            # "cf_2d": cf_2d,
            "c_adv": c_adv,
            "c_bf": c_bf,
        }

    @staticmethod
    def __compute_bottom_friction_correction(
        rv: np.ndarray, manning: np.ndarray
    ) -> np.ndarray:
        """
        Compute the bottom friction correction for the subgrid

        Returns:
            The bottom friction correction as a 2D array
        """
        bf = np.square(rv)
        bf[np.isnan(bf)] = SubgridPreprocessor.__compute_default_cf(manning)

        return bf

    @staticmethod
    def __compute_advection_correction(
        dp: np.ndarray,
        dp2d: np.ndarray,
        rv: np.ndarray,
        cf2d: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the advection correction for the subgrid

        Returns:
            The advection correction as a 2D array
        """
        c_adv = np.multiply(
            np.multiply(
                np.reciprocal(dp),
                np.nanmean(np.divide(np.square(dp2d), cf2d), axis=(1, 2)),
            ),
            np.square(rv),
        )

        c_adv[np.isnan(c_adv)] = 1.0

        return c_adv

    @staticmethod
    def __compute_default_cf(manning: np.ndarray) -> float:
        """
        Compute the default friction coefficient for the subgrid

        Returns:
            The default friction coefficient as a 2D array
        """
        from scipy.constants import g

        return (
            g
            * np.nanmean(np.square(manning))
            / np.cbrt(SubgridPreprocessor.DRY_PIXEL_WATER_DEPTH)
        )

    def __subset_raster_data_for_node(
        self, node_index: int, window: RasterRegion, window_data: xr.Dataset
    ) -> dict:
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
        dem_values, manning_values = self.__subset_raster_arrays_to_stencil(
            node_mask, sub_window, window_data
        )

        return {
            "dem": dem_values,
            "manning": manning_values,
            "sub_window": sub_window,
        }

    @staticmethod
    def __compute_wet_pixel_mask(
        dem_values: np.ndarray, wse_levels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Compute the wet pixel mask for each water level

        Args:
            dem_values: The DEM values
            wse_levels: The water surface elevation levels

        Returns:
            The number of wet pixels at each water level, the 2d wet pixel mask, and the total number of pixels
        """
        wet_masks = np.where(
            np.array([dem_values < level for level in wse_levels], dtype=float),
            1.0,
            np.nan,
        )
        wet_level_counts = np.nansum(wet_masks, axis=(1, 2), dtype=float)
        total_pixels = np.isfinite(dem_values).sum()

        return wet_level_counts, wet_masks, total_pixels

    @staticmethod
    def __compute_rv_at_levels(
        dp: np.ndarray,
        dp_levels: np.ndarray,
        cf_levels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the relative velocity at each water level

        Args:
            dp: The mean water depth at each pixel per water level
            dp_levels: The water depth at each pixel per water level
            cf_levels: The friction coefficient at each pixel per water level

            h / dp ** (3/2) *  1/ sqrt(cf)

        Returns:
            The relative velocity at each water level as a 2D array
        """
        return np.divide(
            dp,
            np.nanmean(
                np.multiply(np.pow(dp_levels, 1.5), np.sqrt(np.reciprocal(cf_levels))),
                axis=(1, 2),
            ),
        )

    @staticmethod
    def __compute_cf_at_levels(
        dp_levels: np.ndarray,
        wet_masks: np.ndarray,
        manning_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the friction coefficient at each water level

        Args:
            dp_levels: The water depth at each pixel per water level
            wet_masks: The wet mask for each water level
            manning_values: The Manning's n values

        Returns:
            The friction coefficient at each water level aggregated and as a 2D array
        """
        from scipy.constants import g

        manning_levels_sq = wet_masks * np.square(manning_values)
        cf_2d = np.divide(np.multiply(g, manning_levels_sq), np.cbrt(dp_levels))
        cf_aggregated = np.nanmean(cf_2d, axis=(1, 2))

        return cf_aggregated, cf_2d

    @staticmethod
    def __compute_water_depth_at_levels(
        dem_values: np.ndarray,
        wet_masks: np.ndarray,
        wse_levels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the water depth at each water level

        Args:
            dem_values: The DEM values
            wet_masks: The wet mask for each water level
            wse_levels: The water surface elevation levels

        Returns:
            The mean pixel water depth at each water level
        """
        depth_levels = np.array(
            [
                wet_masks[idx] * (level - dem_values)
                for idx, level in enumerate(wse_levels)
            ]
        )

        # Check that water depths are greater than the dry pixel water depth
        depth_levels = np.where(
            depth_levels < SubgridPreprocessor.DRY_PIXEL_WATER_DEPTH,
            np.nan,
            depth_levels,
        )

        dp_agg = np.nanmean(depth_levels, axis=(1, 2))

        return dp_agg, depth_levels

    def __generate_calculation_intervals(
        self, dem_elevations: np.ndarray
    ) -> np.ndarray:
        """
        Generate the calculation intervals for the water surface elevation

        Returns:
            An array of water surface elevation levels
        """
        min_elevation = np.nanmin(dem_elevations)
        max_elevation = np.nanmax(dem_elevations)
        return np.linspace(
            min_elevation, max_elevation, self.__config.data()["n_subgrid_levels"]
        )

    @staticmethod
    def __subset_raster_arrays_to_stencil(
        node_mask: np.ndarray, sub_window: dict, window_data: xr.Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Subset the DEM and Manning arrays to the stencil for the node

        Args:
            node_mask: The mask for the node
            sub_window: The sub-window for the node
            window_data: The data from the raster window

        Returns:
            A tuple of the DEM and Manning arrays for the node
        """
        dem_values = np.where(
            node_mask,
            window_data.dem.to_numpy()[
                sub_window["j_start"] : sub_window["j_end"],
                sub_window["i_start"] : sub_window["i_end"],
            ],
            np.nan,
        )

        manning_values = np.where(
            node_mask,
            window_data.manning_n.to_numpy()[
                sub_window["j_start"] : sub_window["j_end"],
                sub_window["i_start"] : sub_window["i_end"],
            ],
            np.nan,
        )

        return dem_values, manning_values

    def __generate_sub_window_mask(
        self, node_index: int, sub_window: dict
    ) -> np.ndarray:
        """
        Generate a mask for the sub-window that represents the polygon for the node

        Args:
            node_index: The index of the node
            sub_window: The sub-window to generate the mask for
        """
        from rasterio import features

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
        import rasterio as rio

        node_bounds = (
            self.__adcirc_mesh.subarea_polygons().polygons()[node_index].bounds
        )
        i_start = int((node_bounds[0] - window.xll()) / window.cell_size())
        j_start = int((node_bounds[1] - window.yll()) / window.cell_size())
        i_end = int((node_bounds[2] - window.xll()) / window.cell_size())
        j_end = int((node_bounds[3] - window.yll()) / window.cell_size())
        i_size = i_end - i_start
        j_size = j_end - j_start
        this_transform = rio.transform.from_bounds(
            node_bounds[0],
            node_bounds[3],
            node_bounds[2],
            node_bounds[1],
            i_size,
            j_size,
        )

        return {
            "i_start": i_start,
            "j_start": j_start,
            "i_end": i_end,
            "j_end": j_end,
            "i_size": i_size,
            "j_size": j_size,
            "transform": this_transform,
        }

    @staticmethod
    def __plot_diagnostics(
        window: RasterRegion,
        sub_window: dict,
        wse_levels: np.ndarray,
        data: dict,
    ) -> None:
        """
        Quick diagnostic plot to check the results of the calculations

        Args:
            window: The raster window
            sub_window: The sub-window for the node
            wse_levels: The water surface elevation levels
            data: The data dictionary containing the subgrid variables for the node
        """
        import matplotlib.pyplot as plt

        # xg, yg = np.meshgrid(
        #     np.linspace(window.xll(), window.xur(), sub_window["i_size"]),
        #     np.linspace(window.yll(), window.yur(), sub_window["j_size"]),
        # )

        fig, ax = plt.subplots(2, 3, figsize=(15, 15))
        ax[0, 0].plot(wse_levels, data["wet_fraction"])
        ax[0, 0].set_xlabel("Water Level (m)")
        ax[0, 0].set_ylabel("Wet Area (%)")
        ax[0, 0].set_xlim([min(wse_levels), max(wse_levels)])
        ax[0, 0].grid()

        ax[0, 1].plot(wse_levels, data["dp"])
        ax[0, 1].set_xlabel("Water Level (m)")
        ax[0, 1].set_ylabel("Mean Water Depth (m)")
        ax[0, 1].set_xlim([min(wse_levels), max(wse_levels)])
        ax[0, 1].grid()

        ax[0, 2].plot(wse_levels, data["cf"])
        ax[0, 2].set_xlabel("Water Level (m)")
        ax[0, 2].set_ylabel("Quadratic Friction Coefficient (c_f)")
        ax[0, 2].set_xlim([min(wse_levels), max(wse_levels)])
        ax[0, 2].grid()

        ax[1, 0].plot(wse_levels, data["c_bf"])
        ax[1, 0].set_xlabel("Water Level (m)")
        ax[1, 0].set_ylabel("Correction (Bottom Friction)")
        ax[1, 0].set_xlim([min(wse_levels), max(wse_levels)])
        ax[1, 0].grid()

        ax[1, 1].plot(wse_levels, data["c_adv"])
        ax[1, 1].set_xlabel("Water Level (m)")
        ax[1, 1].set_ylabel("Correction (Advection)")
        ax[1, 1].set_xlim([min(wse_levels), max(wse_levels)])
        ax[1, 1].grid()

        # Convert the wet mask to a binary mask
        # wm = np.where(data["wet_masks"][-1] > 0, 1, 0)
        # f0 = ax[1, 0].pcolormesh(xg, yg, wm, vmin=0, vmax=1, cmap="jet")
        # ax[1, 0].set_title(f"Pixel Mask for WL {wse_levels[-1]}")
        # fig.colorbar(f0, ax=ax[1, 0], orientation="horizontal")
        #
        # f1 = ax[1, 1].pcolormesh(
        #     xg, yg, data["dp_2d"][-1], vmin=-3, vmax=10, cmap="jet"
        # )
        # ax[1, 1].set_title(f"Depth for WL {wse_levels[-1]}")
        # fig.colorbar(f1, ax=ax[1, 1], orientation="horizontal")
        #
        # f2 = ax[1, 2].pcolormesh(
        #     xg, yg, data["cf_2d"][-1], vmin=0.0, vmax=0.2, cmap="jet"
        # )
        # ax[1, 2].set_title(f"CF for WL {wse_levels[-1]}")
        # fig.colorbar(f2, ax=ax[1, 2], orientation="horizontal")

        plt.tight_layout()
        plt.show()
