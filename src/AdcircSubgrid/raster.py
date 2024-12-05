from typing import Optional

import numpy as np
import xarray as xr
from matplotlib.patches import Polygon
from osgeo import gdal
from pyproj import CRS, Transformer

from .lookup_table import LookupTable
from .raster_region import RasterRegion

gdal.UseExceptions()


class Raster:
    """
    Class to handle raster data
    """

    def __init__(
        self, filename: str, lookup_table: Optional[LookupTable] = None
    ) -> None:
        """
        Initialize the DEM class using a DEM file

        Args:
            filename: The filename of the raster file
            lookup_table: The lookup table to apply to the raster data (default: None)
        """
        self.__filename = filename
        self.__lookup_table = lookup_table
        self.__handle = None
        self.__geo_transform = None
        self.__bounds = None
        self.__bbox = None
        self.__nodata_value = None
        self.__coordinate_system = None
        self.__pj_transform = None
        self.__i_size = None
        self.__j_size = None
        self.__x_resolution = None
        self.__y_resolution = None
        self.__x_start = None
        self.__y_start = None
        self.__x_end = None
        self.__y_end = None
        self.__initialize_raster()

    def __repr__(self) -> str:
        """
        Get a string representation of the Raster object

        Returns:
            A string representation of the Raster object
        """
        nodata = (
            "None" if self.__nodata_value is None else f"{self.__nodata_value:0.3f}"
        )
        return (
            f"Raster("
            f"Bounds: ({self.__bounds[0]:0.3f}, {self.__bounds[1]:0.3f}, {self.__bounds[2]:0.3f}, {self.__bounds[3]:0.3f}),"
            f" Resolution: ({self.__geo_transform[1]:0.3e}, {self.__geo_transform[5]:0.3e}),"
            f" NoData Value: {nodata},"
            f" Lookup Table: {self.__lookup_table is not None})"
        )

    def __initialize_raster(self) -> None:
        """
        Initialize the DEM class using a DEM file

        Returns:
            None
        """
        self.__handle = gdal.Open(self.__filename)
        self.__geo_transform = self.__handle.GetGeoTransform()
        self.__nodata_value = self.__handle.GetRasterBand(1).GetNoDataValue()
        self.__i_size = self.__handle.RasterXSize
        self.__j_size = self.__handle.RasterYSize
        self.__bounds = (
            self.__geo_transform[0],
            self.__geo_transform[3],
            self.__geo_transform[0]
            + self.__geo_transform[1] * self.__handle.RasterXSize,
            self.__geo_transform[3]
            + self.__geo_transform[5] * self.__handle.RasterYSize,
        )
        self.__bbox = Polygon(
            [
                (self.__bounds[0], self.__bounds[1]),
                (self.__bounds[2], self.__bounds[1]),
                (self.__bounds[2], self.__bounds[3]),
                (self.__bounds[0], self.__bounds[3]),
            ]
        )

        self.__x_resolution = self.__geo_transform[1]
        self.__y_resolution = self.__geo_transform[5]
        self.__x_start = self.__bounds[0]
        self.__y_start = self.__bounds[1]
        self.__x_end = self.__bounds[2]
        self.__y_end = self.__bounds[3]

        self.__coordinate_system = self.__handle.GetProjection()
        self.__pj_transform = Transformer.from_crs(
            CRS.from_string("EPSG:4326"),
            CRS.from_string(self.__coordinate_system),
            always_xy=True,
        )

    def bounds(self) -> list[float]:
        """
        Get the bounds of the raster

        Returns:
            The bounds of the raster
        """
        return self.__bounds

    def bbox(self) -> Polygon:
        """
        Get the bounding box of the raster sd

        Returns:
            The bounding box of the raster
        """
        return self.__bbox

    def get_region(self, x1: float, y1: float, x2: float, y2: float) -> xr.Dataset:
        """
        Get a region from the raster data

        Args:
            x1: The minimum x-coordinate
            y1: The minimum y-coordinate
            x2: The maximum x-coordinate
            y2: The maximum y-coordinate

        Returns:
            A xarray Dataset with the region data
        """
        # Transform the coordinates to the raster coordinate system
        x1p, y1p = self.__pj_transform.transform(x1, y1)
        x2p, y2p = self.__pj_transform.transform(x2, y2)

        region = RasterRegion(
            self.__geo_transform, self.__i_size, self.__j_size, x1p, y1p, x2p, y2p
        )

        data = self.__handle.ReadAsArray(
            region.x_start(),
            region.y_start(),
            region.x_size(),
            region.y_size(),
        )
        data = np.where(data == self.__nodata_value, np.nan, data)

        if self.__lookup_table:
            data = self.__apply_lookup_table(data)

        x_pt = region.x_pts()
        y_pt = region.y_pts()

        xg, yg = np.meshgrid(x_pt, y_pt)

        # Transform the coordinates back to lat/lon
        xg_ll, yg_ll = self.__pj_transform.transform(xg, yg, direction="INVERSE")

        return xr.Dataset(
            {
                "values": (("y", "x"), data),
            },
            coords={"lon": (("y", "x"), xg_ll), "lat": (("y", "x"), yg_ll)},
        )

    def __apply_lookup_table(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the lookup table to the data

        Args:
            data: The data to apply the lookup table to

        Returns:
            The data with the lookup table applied
        """
        lookup_table = self.__lookup_table.lookup_table()

        def lookup_function(x: np.ndarray) -> np.ndarray:
            return lookup_table[x]

        func = np.vectorize(lookup_function)
        return func(data)

    def generate_windows(
        self, memory_mb_per_window: float, overlap_width: float
    ) -> list[RasterRegion]:
        """
        Generates windows for the raster based on the user's memory requirements
        and applies a user specified overlap region in both the x and y directions

        Args:
            memory_mb_per_window: The amount of memory in MB that each window should consume
            overlap_width: The width of the overlap between windows in raster units

        Returns:
            A list of RasterRegion objects
        """
        # Calculate the number of windows in the x and y directions
        window_size = int(np.sqrt(memory_mb_per_window * 1024 * 1024 / 8))
        overlap = int(overlap_width / self.__x_resolution / 2)

        windows = []
        for i in range(0, self.__i_size - window_size + overlap, window_size):
            region_i_start = max(0, i - overlap)
            region_i_end = min(self.__i_size, i + window_size + overlap)
            region_x_start = self.__x_start + region_i_start * self.__x_resolution
            region_x_end = self.__x_start + region_i_end * self.__x_resolution

            for j in range(0, self.__j_size - window_size + overlap, window_size):
                region_j_start = max(0, j - overlap)
                region_j_end = min(self.__j_size, j + window_size + overlap)
                region_y_start = self.__y_start + region_j_start * self.__y_resolution
                region_y_end = self.__y_start + region_j_end * self.__y_resolution

                windows.append(
                    RasterRegion(
                        self.__geo_transform,
                        self.__i_size,
                        self.__j_size,
                        region_x_start,
                        region_y_start,
                        region_x_end,
                        region_y_end,
                    )
                )

        return windows
