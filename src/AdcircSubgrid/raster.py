from typing import Optional

import numpy as np
import rasterio as rio
import rioxarray  # noqa: F401
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
        self.__bbox = None
        self.__x_resolution = None
        self.__y_resolution = None
        self.__i_size = None
        self.__j_size = None
        self.__x_start = None
        self.__y_start = None
        self.__x_end = None
        self.__y_end = None
        self.__pj_transform = None
        self.__initialize_raster()

    def __repr__(self) -> str:
        """
        Get a string representation of the Raster object

        Returns:
            A string representation of the Raster object
        """
        return str(self.__handle)

    def __initialize_raster(self) -> None:
        """
        Initialize the DEM class using a DEM file

        Returns:
            None
        """
        self.__handle = rio.open(self.__filename)
        self.__bbox = Polygon(
            [
                (self.__handle.bounds.left, self.__handle.bounds.bottom),
                (self.__handle.bounds.left, self.__handle.bounds.top),
                (self.__handle.bounds.right, self.__handle.bounds.top),
                (self.__handle.bounds.right, self.__handle.bounds.bottom),
            ]
        )
        self.__x_resolution, self.__y_resolution = self.__handle.res
        self.__i_size = self.__handle.width
        self.__j_size = self.__handle.height
        self.__x_start, self.__y_start = (
            self.__handle.bounds.left,
            self.__handle.bounds.bottom,
        )
        self.__x_end, self.__y_end = (
            self.__handle.bounds.right,
            self.__handle.bounds.top,
        )
        self.__nodata_value = self.__handle.nodata

        self.__pj_transform = Transformer.from_crs(
            CRS.from_string("EPSG:4326"),
            self.__handle.crs,
            always_xy=True,
        )

    def x_resolution(self) -> float:
        """
        Get the x resolution of the raster

        Returns:
            The x resolution of the raster
        """
        return self.__x_resolution

    def y_resolution(self) -> float:
        """
        Get the y resolution of the raster

        Returns:
            The y resolution of the raster
        """
        return self.__y_resolution

    def transform(self) -> rio.transform.Affine:
        """
        Get the geo transform of the raster

        Returns:
            The geo transform of the raster
        """
        return self.__handle.transform

    def geo_transform(self) -> list[float]:
        """
        Get the geo transform of the raster

        Returns:
            The geo transform of the raster
        """
        return self.__handle.transform.to_gdal()

    def coordinate_system(self) -> rio.crs.CRS:
        """
        Get the coordinate system of the raster

        Returns:
            The coordinate system of the raster
        """
        return self.__handle.crs

    def bounds(self) -> rio.coords.BoundingBox:
        """
        Get the bounds of the raster

        Returns:
            The bounds of the raster
        """
        return self.__handle.bounds

    def bbox(self) -> Polygon:
        """
        Get the bounding box of the raster sd

        Returns:
            The bounding box of the raster
        """
        return self.__bbox

    def get_region(
        self, x1: float, y1: float, x2: float, y2: float, target_name: str = "values"
    ) -> xr.Dataset:
        """
        Get a region from the raster data

        Args:
            x1: The minimum x-coordinate in raster units
            y1: The minimum y-coordinate in raster units
            x2: The maximum x-coordinate in raster units
            y2: The maximum y-coordinate in raster units
            target_name: The name of the target variable (default: "values")

        Returns:
            A xarray Dataset with the region data
        """
        region = RasterRegion(
            self.geo_transform(), self.__i_size, self.__j_size, x1, y1, x2, y2
        )

        data = self.__handle.read(1, window=region.rio_window())
        if self.__lookup_table:
            data = self.__apply_lookup_table(data)
        else:
            data = np.where(data == self.__nodata_value, np.nan, data)

        x_pt = np.linspace(region.xll(), region.xur(), region.i_size())
        y_pt = np.linspace(region.yur(), region.yll(), region.j_size())

        # Return a dataset with the appropriate coordinates and rio transform
        out_ds = xr.Dataset(
            {
                target_name: (("lat", "lon"), data),
            },
            coords={"lon": x_pt, "lat": y_pt},
        )

        out_ds.rio.set_spatial_dims("lon", "lat", inplace=True)
        out_ds.rio.write_crs(self.__handle.crs, inplace=True)
        transform_this = rio.transform.from_bounds(
            region.xll(),
            region.yll(),
            region.xur(),
            region.yur(),
            region.i_size(),
            region.j_size(),
        )
        out_ds.rio.write_transform(transform_this, inplace=True)
        return out_ds

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
                        self.geo_transform(),
                        self.__i_size,
                        self.__j_size,
                        region_x_start,
                        region_y_start,
                        region_x_end,
                        region_y_end,
                    )
                )

        return windows
