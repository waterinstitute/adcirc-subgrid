import numpy as np


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

        if self.__level_distribution == "linear":
            return self.__generate_calculation_intervals_linear(dem_elevations)
        elif self.__level_distribution == "normal":
            return self.__generate_calculation_intervals_normal(dem_elevations)
        return None

    def __generate_calculation_intervals_normal(
        self, dem_elevations: np.ndarray
    ) -> np.ndarray:
        """
        Generate the calculation intervals for the water surface elevation using a normal distribution

        Args:
            dem_elevations: The DEM elevations

        Returns:
            An array of water surface elevation levels
        """
        from scipy import stats

        dz_dry = self.__dry_pixel_depth * 2

        start_elev = np.nanmin(dem_elevations) - dz_dry
        end_elev = np.nanmax(dem_elevations) + dz_dry
        std_dev = np.nanstd(dem_elevations)

        dist = stats.norm(
            loc=(start_elev + end_elev) / 2,
            scale=std_dev / self.__distribution_factor,
        )

        calc_levels = dist.ppf(np.linspace(0.01, 0.99, self.__n_subgrid_levels))

        calc_levels[0] = start_elev if start_elev < calc_levels[0] else calc_levels[0]
        calc_levels[-1] = end_elev if end_elev > calc_levels[-1] else calc_levels[-1]

        return calc_levels

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
