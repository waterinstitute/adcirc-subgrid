from typing import Union

import numpy as np
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
        self.__x_start = None
        self.__y_start = None
        self.__x_end = None
        self.__y_end = None

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
        self.__x_start = int(
            (self.__xll - self.__geo_transform[0]) / self.__geo_transform[1]
        )
        self.__x_end = int(
            (self.__xur - self.__geo_transform[0]) / self.__geo_transform[1]
        )
        self.__y_start = int(
            (self.__geo_transform[3] - self.__yur) / -self.__geo_transform[5]
        )
        self.__y_end = int(
            (self.__geo_transform[3] - self.__yll) / -self.__geo_transform[5]
        )

        # Clamp the region within the bounds of the raster
        self.__x_start = max(0, self.__x_start)
        self.__y_start = max(0, self.__y_start)
        self.__x_end = min(self.__x_end, x_size - 1)
        self.__y_end = min(self.__y_end, y_size - 1)

        return xll_region, xur_region, yll_region, yur_region

    def __check_valid(self) -> bool:
        """
        Check if the region is valid.

        This is a simple check to ensure that the region is not empty.

        Returns:
            True if the region is valid, False otherwise
        """
        return self.__x_start < self.__x_end and self.__y_start < self.__y_end

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
            f" Start({self.__x_start}, {self.__y_start}),"
            f" End({self.__x_end}, {self.__y_end}),"
            f" Size({self.x_size()}, {self.y_size()}))"
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

    def x_start(self) -> int:
        """
        Get the starting x index

        Returns:
            The starting x index
        """
        return self.__x_start

    def y_start(self) -> int:
        """
        Get the starting y index

        Returns:
            The starting y index
        """
        return self.__y_start

    def x_end(self) -> int:
        """
        Get the ending x index

        Returns:
            The ending x index
        """
        return self.__x_end

    def y_end(self) -> int:
        """
        Get the ending y index

        Returns:
            The ending y index
        """
        return self.__y_end

    def x_size(self) -> int:
        """
        Get the x size

        Returns:
            The x size
        """
        return self.__x_end - self.__x_start

    def y_size(self) -> int:
        """
        Get the y size

        Returns:
            The y size
        """
        return self.__y_end - self.__y_start

    def x_pts(self) -> np.ndarray:
        """
        Get the x points

        Returns:
            The x points
        """
        return np.linspace(
            self.__geo_transform[0]
            + self.__x_start * self.__geo_transform[1]
            + 0.5 * self.__geo_transform[1],
            self.__geo_transform[0]
            + self.__x_end * self.__geo_transform[1]
            - 0.5 * self.__geo_transform[1],
            self.__x_end - self.__x_start,
        )

    def y_pts(self) -> np.ndarray:
        """
        Get the y points

        Returns:
            The y points
        """
        return np.linspace(
            self.__geo_transform[3]
            + self.__y_start * self.__geo_transform[5]
            + 0.5 * self.__geo_transform[5],
            self.__geo_transform[3]
            + self.__y_end * self.__geo_transform[5]
            - 0.5 * self.__geo_transform[5],
            self.__y_end - self.__y_start,
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
