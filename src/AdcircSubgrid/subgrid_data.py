import numpy as np


class SubgridData:
    """
    Class to store the output of the subgrid calculations
    """

    def __init__(
        self,
        node_count: int,
        sg_count: int,
        phi_count: int,
        interpolation_method: str = "linear",
    ) -> None:
        """
        Initialize the SubgridOutput class

        Args:
            node_count: Number of nodes in the subgrid
            sg_count: Number of subgrid levels
            phi_count: Number of phi values to store
            interpolation_method: The interpolation method to use for the output
                                  on phi levels (linear or cubic) (default: linear)
        """
        self.__node_count = node_count
        self.__sg_count = sg_count
        self.__phi_count = phi_count
        self.__interpolation_method = interpolation_method

        # Allow the user to select either linear or cubic interpolation
        if self.__interpolation_method not in ["linear", "cubic"]:
            msg = f"Invalid interpolation method: {self.__interpolation_method}"
            raise ValueError(msg)

        self.__phi_set = np.linspace(0.0, 1.0, self.__phi_count)
        self.__wet_water_depth = np.zeros((self.__node_count, self.__phi_count))
        self.__wet_total_depth = np.zeros((self.__node_count, self.__phi_count))
        self.__c_f = np.zeros((self.__node_count, self.__phi_count))
        self.__c_bf = np.zeros((self.__node_count, self.__phi_count))
        self.__c_adv = np.zeros((self.__node_count, self.__phi_count))
        self.__vertex_list = np.zeros(self.__node_count, dtype=int)
        self.__vertex_flag = np.zeros(self.__node_count, dtype=int)

    def node_count(self) -> int:
        """
        Return the number of nodes in the subgrid

        Returns:
            The number of nodes in the subgrid
        """
        return self.__node_count

    def sg_count(self) -> int:
        """
        Return the number of subgrid levels

        Returns:
            The number of subgrid levels
        """
        return self.__sg_count

    def phi_count(self) -> int:
        """
        Return the number of phi values

        Returns:
            The number of phi values
        """
        return self.__phi_count

    def phi(self) -> np.ndarray:
        """
        Return the phi values

        Returns:
            The phi values
        """
        return self.__phi_set

    def wet_water_depth(self) -> np.ndarray:
        """
        Return the wet water depth values

        Returns:
            The wet water depth values
        """
        return self.__wet_water_depth

    def wet_total_depth(self) -> np.ndarray:
        """
        Return the wet total depth values

        Returns:
            The wet total depth values
        """
        return self.__wet_total_depth

    def c_f(self) -> np.ndarray:
        """
        Return the quadratic friction values

        Returns:
            The quadratic friction values
        """
        return self.__c_f

    def c_bf(self) -> np.ndarray:
        """
        Return the friction correction values

        Returns:
            The friction correction values
        """
        return self.__c_bf

    def c_adv(self) -> np.ndarray:
        """
        Return the advection correction values

        Returns:
            The advection correction values
        """
        return self.__c_adv

    def vertex_flag(self) -> np.ndarray:
        """
        Return the vertex flag values

        Returns:
            The vertex flag values
        """
        return self.__vertex_flag

    def set_data(
        self,
        vertex_flag: np.ndarray,
        phi: np.ndarray,
        wet_water_depth: np.ndarray,
        wet_total_depth: np.ndarray,
        c_f: np.ndarray,
        c_bf: np.ndarray,
        c_adv: np.ndarray,
    ) -> None:
        """
        Set the output data for all vertices

        Args:
            vertex_flag: The vertex flag values
            phi: The phi values
            wet_water_depth: The wet water depth values
            wet_total_depth: The wet total depth values
            c_f: The quadratic friction values
            c_bf: The friction correction values
            c_adv: The advection correction values
        """
        if (
            vertex_flag.shape != (self.__node_count,)
            or phi.shape != (self.__phi_count,)
            or wet_water_depth.shape != (self.__node_count, self.__phi_count)
            or wet_total_depth.shape != (self.__node_count, self.__phi_count)
            or c_f.shape != (self.__node_count, self.__phi_count)
            or c_bf.shape != (self.__node_count, self.__phi_count)
            or c_adv.shape != (self.__node_count, self.__phi_count)
        ):
            msg = "Invalid shape for input arrays"
            raise ValueError(msg)

        self.__vertex_flag = vertex_flag
        self.__wet_water_depth = wet_water_depth
        self.__wet_total_depth = wet_total_depth
        self.__c_f = c_f
        self.__c_bf = c_bf
        self.__c_adv = c_adv

    def add_vertex(
        self,
        vertex: int,
        wet_fraction: np.ndarray,
        wet_water_depth: np.ndarray,
        wet_total_depth: np.ndarray,
        c_f: np.ndarray,
        c_bf: np.ndarray,
        c_adv: np.ndarray,
    ) -> None:
        """
        Add a vertex to the output data

        Args:
            vertex: The vertex number
            wet_fraction: The wet fraction values
            wet_water_depth: The wet water depth values
            wet_total_depth: The wet total depth values
            c_f: The quadratic friction values
            c_bf: The friction correction values
            c_adv: The advection correction values
        """
        # Check if the shape of the input arrays is correct
        if (
            wet_fraction.shape != (self.__sg_count,)
            or wet_water_depth.shape != (self.__sg_count,)
            or wet_total_depth.shape != (self.__sg_count,)
            or c_f.shape != (self.__sg_count,)
            or c_bf.shape != (self.__sg_count,)
            or c_adv.shape != (self.__sg_count,)
        ):
            msg = "Invalid shape for input arrays"
            raise ValueError(msg)

        self.__vertex_flag[vertex] = 1
        if self.__interpolation_method == "linear":
            self.__interp_linear(
                vertex, wet_fraction, wet_water_depth, wet_total_depth, c_f, c_bf, c_adv
            )
        else:
            msg = f"Invalid interpolation method: {self.__interpolation_method}"
            raise ValueError(msg)

    def __interp_linear(
        self,
        vertex: int,
        wet_fraction: np.ndarray,
        wet_water_depth: np.ndarray,
        wet_total_depth: np.ndarray,
        c_f: np.ndarray,
        c_bf: np.ndarray,
        c_adv: np.ndarray,
    ) -> None:
        """
        Interpolate the input data using linear interpolation

        Args:
            vertex: The vertex number
            wet_fraction: The wet fraction values
            wet_water_depth: The wet water depth values
            wet_total_depth: The wet total depth values
            c_f: The quadratic friction values
            c_bf: The friction correction values
            c_adv: The advection correction values
        """
        self.__wet_water_depth[vertex] = np.interp(
            self.__phi_set,
            wet_fraction,
            wet_water_depth,
        )
        self.__wet_total_depth[vertex] = np.interp(
            self.__phi_set, wet_fraction, wet_total_depth
        )
        self.__c_f[vertex] = np.interp(self.__phi_set, wet_fraction, c_f)
        self.__c_bf[vertex] = np.interp(self.__phi_set, wet_fraction, c_bf)
        self.__c_adv[vertex] = np.interp(self.__phi_set, wet_fraction, c_adv)
