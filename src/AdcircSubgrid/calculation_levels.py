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

import numpy as np

log = logging.getLogger(__name__)


class CalculationLevels:
    """
    Class to control computing the levels at which the subgrid calculation occurs
    """

    def __init__(
        self,
        level_distribution: str,
        n_subgrid_levels: int,
        distribution_factor: float,
        dry_pixel_depth: float,
    ) -> None:
        """
        Constructor

        Args:
            level_distribution: The distribution to use for the calculation levels
            n_subgrid_levels: The number of subgrid levels
            distribution_factor: The distribution factor used for a normal distribution
            dry_pixel_depth: The dry pixel depth
        """
        self.__level_distribution = level_distribution
        self.__n_subgrid_levels = n_subgrid_levels
        self.__distribution_factor = distribution_factor
        self.__dry_pixel_depth = dry_pixel_depth

    def get_levels(self, dem_elevations: np.ndarray) -> np.ndarray:
        """
        Generate the calculation intervals for the water surface elevation

        Args:
            dem_elevations: The DEM elevations

        Returns:
            An array of water surface elevation levels
        """
        # TODO: I think there are a lot of interesting things we could do here.
        #       it wouldn't be that hard to compute the wet levels on an overly
        #       fine grid and back out the exact % of wet the subgrid element is
        #       How much it matters is probably an unknown
        if np.isfinite(dem_elevations).sum() <= 0:
            # Return dummy data since we won't use it anyway
            return np.linspace(-1, 1, self.__n_subgrid_levels)

        if self.__level_distribution == "linear":
            return self.__generate_calculation_intervals_linear(dem_elevations)
        elif self.__level_distribution == "histogram":
            return self.__generate_calculation_intervals_histogram(dem_elevations)
        else:
            msg = "Invalid distribution type"
            raise ValueError(msg)

    def __generate_calculation_intervals_histogram(
        self, dem_elevations: np.ndarray
    ) -> np.ndarray:
        """
        Generate the calculation intervals for the water surface elevation using a histogram distribution

        Args:
            dem_elevations: The DEM elevations

        Returns:
            An array of water surface elevation levels
        """
        try:
            dem_elevations_flat = dem_elevations.flatten()
            dem_elevations_flat = dem_elevations_flat[~np.isnan(dem_elevations_flat)]

            min_dem = np.nanmin(dem_elevations_flat) - 2 * self.__dry_pixel_depth
            max_dem = np.nanmax(dem_elevations_flat) + 2 * self.__dry_pixel_depth

            levels = np.histogram(
                dem_elevations_flat, bins=self.__n_subgrid_levels - 2
            )[1][:-1]

            if levels[0] > min_dem:
                levels = np.insert(levels, 0, min_dem)
            else:
                levels_int = levels[0] + (levels[0] - levels[1])
                levels = np.insert(levels, 1, levels_int)

            if levels[-1] < max_dem:
                levels = np.append(levels, max_dem)
            else:
                levels_int = levels[-1] + (levels[-1] - levels[-2])
                levels = np.append(levels, levels_int)

            return levels
        except ValueError:
            return self.__generate_calculation_intervals_linear(dem_elevations)

    def __generate_calculation_intervals_linear(
        self, dem_elevations: np.ndarray
    ) -> np.ndarray:
        """
        Generate the calculation intervals for the water surface elevation using a linear distribution

        Args:
            dem_elevations: The DEM elevations

        Returns:
            An array of water surface elevation levels
        """
        q1 = np.nanpercentile(dem_elevations, 25)
        q3 = np.nanpercentile(dem_elevations, 75)
        iqr = q3 - q1

        start_elev = q1 - 1.5 * iqr
        end_elev = q3 + 1.5 * iqr

        return np.linspace(
            start_elev,
            end_elev,
            self.__n_subgrid_levels,
        )
