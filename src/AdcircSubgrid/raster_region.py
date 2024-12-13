from typing import Union

import numpy as np
from rasterio.windows import Window
from shapely.geometry import Point, Polygon


class RasterRegion:
    """
    Class that handles the window over a region of a raster
    """

    def __init__(
        self,
        geo_transform: list[float],
        x_size: int,
        y_size: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        """
        Initialize the RasterRegion class

        Args:
            geo_transform: The geo-transform of the raster
            x_size: The x size of the raster
            y_size: The y size of the raster
            x1: First x-coordinate
            y1: First y-coordinate
            x2: Second x-coordinate
            y2: Second y-coordinate
        """
        self.__geo_transform = geo_transform
        self.__xll = None
        self.__yll = None
        self.__xur = None
        self.__yur = None
        self.__i_start = None
        self.__j_start = None
        self.__i_end = None
        self.__j_end = None

        xll_region, xur_region, yll_region, yur_region = self.__compute_region_geometry(
            x1, y1, x2, y2, x_size, y_size
        )

        # Set a flag if this region is valid
        self.__valid = self.__check_valid()

        # Set a flag if the region was clamped
        self.__clamped = self.__check_clamped(
            xll_region, yll_region, xur_region, yur_region
        )

        # Generate a polygon for the region
        self.__polygon = self.__generate_region_polygon()

    def __generate_region_polygon(self) -> Polygon:
        """
        Generate a polygon for the region

        Returns:
            A shapely Polygon object
        """
        return Polygon(
            [
                (self.__xll, self.__yll),
                (self.__xur, self.__yll),
                (self.__xur, self.__yur),
                (self.__xll, self.__yur),
            ]
        )

    def __compute_region_geometry(
        self, x1: float, y1: float, x2: float, y2: float, x_size: int, y_size: int
    ) -> tuple[float, float, float, float]:
        """
        Compute the geometry of the region and clamp it to the raster bounds

        Args:
            x1: x-coordinate of the first point
            y1: y-coordinate of the first point
            x2: x-coordinate of the second point
            y2: y-coordinate of the second point
            x_size: x_size of the raster
            y_size: y_size of the raster

        Returns:
            The bounds of the region

        """
        # Compute the bounds of the raster as a whole for clamping
        xll_raster = self.__geo_transform[0]
        yur_raster = self.__geo_transform[3]
        xur_raster = xll_raster + x_size * self.__geo_transform[1]
        yll_raster = yur_raster + y_size * self.__geo_transform[5]

        # Compute the bounds of the region
        xll_region = min(x1, x2)
        yll_region = min(y1, y2)
        xur_region = max(x1, x2)
        yur_region = max(y1, y2)

        # Clamp the region within the bounds of the raster
        self.__xll = max(xll_raster, xll_region)
        self.__yll = max(yll_raster, yll_region)
        self.__xur = min(xur_raster, xur_region)
        self.__yur = min(yur_raster, yur_region)

        # Compute the starting and ending indices of the region
        self.__i_start = int(
            (self.__xll - self.__geo_transform[0]) / self.__geo_transform[1]
        )
        self.__i_end = int(
            (self.__xur - self.__geo_transform[0]) / self.__geo_transform[1]
        )
        self.__j_start = int(
            (self.__geo_transform[3] - self.__yur) / -self.__geo_transform[5]
        )
        self.__j_end = int(
            (self.__geo_transform[3] - self.__yll) / -self.__geo_transform[5]
        )

        # Clamp the region within the bounds of the raster
        self.__i_start = max(0, self.__i_start)
        self.__j_start = max(0, self.__j_start)
        self.__i_end = min(self.__i_end, x_size - 1)
        self.__j_end = min(self.__j_end, y_size - 1)

        return xll_region, xur_region, yll_region, yur_region

    def __check_valid(self) -> bool:
        """
        Check if the region is valid.

        This is a simple check to ensure that the region is not empty.

        Returns:
            True if the region is valid, False otherwise
        """
        return self.__i_start < self.__i_end and self.__j_start < self.__j_end

    def __check_clamped(
        self, xll_region: float, yll_region: float, xur_region: float, yur_region: float
    ) -> bool:
        """
        Check if the region was clamped from the user input

        Args:
            xll_region: x-coordinate of the lower left corner of the region
            yll_region: y-coordinate of the lower left corner of the region
            xur_region: x-coordinate of the upper right corner of the region
            yur_region: y-coordinate of the upper right corner of the region

        Returns:
            True if the region was clamped, False otherwise
        """
        return (
            self.__xll != xll_region
            or self.__yll != yll_region
            or self.__xur != xur_region
            or self.__yur != yur_region
        )

    def __repr__(self) -> str:
        """
        Get a string representation of the RasterRegion object

        Returns:
            A string representation of the RasterRegion
        """
        return (
            f"RasterRegion(Bounds({self.__xll:0.3f}, {self.__yll:0.3f}, {self.__xur:0.3f}, {self.__yur:0.3f}),"
            f" Start({self.__i_start}, {self.__j_start}),"
            f" End({self.__i_end}, {self.__j_end}),"
            f" Size({self.i_size()}, {self.j_size()}))"
        )

    def valid(self) -> bool:
        """
        Check if the region is valid

        Returns:
            True if the region is valid, False otherwise
        """
        return self.__valid

    def clamped(self) -> bool:
        """
        Check if the region was clamped

        Returns:
            True if the region was clamped, False otherwise
        """
        return self.__clamped

    def xll(self) -> float:
        """
        Get the lower left x-coordinate

        Returns:
            The lower left x-coordinate
        """
        return self.__xll

    def yll(self) -> float:
        """
        Get the lower left y-coordinate

        Returns:
            The lower left y-coordinate
        """
        return self.__yll

    def xur(self) -> float:
        """
        Get the upper right x-coordinate

        Returns:
            The upper right x-coordinate
        """
        return self.__xur

    def yur(self) -> float:
        """
        Get the upper right y-coordinate

        Returns:
            The upper right y-coordinate
        """
        return self.__yur

    def i_start(self) -> int:
        """
        Get the starting i index

        Returns:
            The starting i index
        """
        return self.__i_start

    def j_start(self) -> int:
        """
        Get the starting j index

        Returns:
            The starting j index
        """
        return self.__j_start

    def i_end(self) -> int:
        """
        Get the ending i index

        Returns:
            The ending i index
        """
        return self.__i_end

    def j_end(self) -> int:
        """
        Get the ending j index

        Returns:
            The ending j index
        """
        return self.__j_end

    def i_size(self) -> int:
        """
        Get the i size

        Returns:
            The i size
        """
        return self.__i_end - self.__i_start

    def j_size(self) -> int:
        """
        Get the j size

        Returns:
            The j size
        """
        return self.__j_end - self.__j_start

    def cell_size(self) -> float:
        """
        Get the cell size

        Returns:
            The cell size
        """
        return self.__geo_transform[1]

    def x_pts(self) -> np.ndarray:
        """
        Get the x points

        Returns:
            The x points
        """
        return np.linspace(
            self.__geo_transform[0]
            + self.__i_start * self.__geo_transform[1]
            + 0.5 * self.__geo_transform[1],
            self.__geo_transform[0]
            + self.__i_end * self.__geo_transform[1]
            - 0.5 * self.__geo_transform[1],
            self.__i_end - self.__i_start,
        )

    def y_pts(self) -> np.ndarray:
        """
        Get the y points

        Returns:
            The y points
        """
        return np.linspace(
            self.__geo_transform[3]
            + self.__j_start * self.__geo_transform[5]
            + 0.5 * self.__geo_transform[5],
            self.__geo_transform[3]
            + self.__j_end * self.__geo_transform[5]
            - 0.5 * self.__geo_transform[5],
            self.__j_end - self.__j_start,
        )

    def polygon(self) -> Polygon:
        """
        Get the polygon for the region

        Returns:
            The polygon for the region
        """
        return self.__polygon

    def contains(self, geometry: Union[Polygon, Point]) -> bool:
        """
        Check if a geometry is within the region

        Args:
            geometry: The geometry to check

        Returns:
            True if the geometry is within the region, False otherwise
        """
        return self.__polygon.contains(geometry)

    def rio_window(self) -> Window:
        """
        Get the rasterio window for the region

        Returns:
            The rasterio window for the region
        """
        return Window(self.__i_start, self.__j_start, self.i_size(), self.j_size())

    def affine_transform(self) -> np.ndarray:
        """
        Get the affine transform for the region

        Returns:
            The affine transform for the region
        """
        from rasterio import transform

        return transform.from_bounds(
            *self.__polygon.bounds, self.i_size(), self.j_size()
        )
