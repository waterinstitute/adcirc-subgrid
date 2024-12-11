import numpy as np
from scipy.interpolate import CubicSpline


class SubgridOutput:
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
        elif self.__interpolation_method == "cubic":
            self.__interpolate_cubic(
                vertex, wet_fraction, wet_water_depth, wet_total_depth, c_f, c_bf, c_adv
            )
        else:
            msg = f"Invalid interpolation method: {self.__interpolation_method}"
            raise ValueError(msg)

    def __interpolate_cubic(
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
        Interpolate the input data using cubic splines

        Args:
            vertex: The vertex number
            wet_fraction: The wet fraction values
            wet_water_depth: The wet water depth values
            wet_total_depth: The wet total depth values
            c_f: The quadratic friction values
            c_bf: The friction correction values
            c_adv: The advection correction values
        """
        cubic_spline = CubicSpline(wet_fraction, wet_water_depth)
        self.__wet_water_depth[vertex] = cubic_spline(self.__phi_set)
        cubic_spline = CubicSpline(wet_fraction, wet_total_depth)
        self.__wet_total_depth[vertex] = cubic_spline(self.__phi_set)
        cubic_spline = CubicSpline(wet_fraction, c_f)
        self.__c_f[vertex] = cubic_spline(self.__phi_set)
        cubic_spline = CubicSpline(wet_fraction, c_bf)
        self.__c_bf[vertex] = cubic_spline(self.__phi_set)
        cubic_spline = CubicSpline(wet_fraction, c_adv)
        self.__c_adv[vertex] = cubic_spline(self.__phi_set)

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

    def write(self, output_file: str) -> None:
        """
        Write the output data to a file

        Args:
            output_file: The output file to write the data to
        """
        from netCDF4 import Dataset

        with Dataset(output_file, "w", format="NETCDF4") as dataset:
            dataset.createDimension("node", self.__node_count)
            dataset.createDimension("phi", self.__phi_count)

            binaryVertexList = dataset.createVariable(
                "binaryVertexList", "i4", ("node",), zlib=True, complevel=2
            )
            binaryVertexList.description = "Vertex subgrid residency flag"

            phi = dataset.createVariable("phi", "f4", ("phi",), zlib=True, complevel=2)
            phi.description = "Levels at which the subgrid data is stored"

            wetTotWatDepthVertex = dataset.createVariable(
                "wetTotWatDepthVertex", "f4", ("node", "phi"), zlib=True, complevel=2
            )
            wetTotWatDepthVertex.description = (
                "Mean water depth in the wet fraction of the subgrid"
            )

            gridTotWatDepthVertex = dataset.createVariable(
                "gridTotWatDepthVertex", "f4", ("node", "phi"), zlib=True, complevel=2
            )
            gridTotWatDepthVertex.description = (
                "Mean water depth for wet and non-wet areas in the subgrid"
            )

            cfVertex = dataset.createVariable(
                "cfVertex", "f4", ("node", "phi"), zlib=True, complevel=2
            )
            cfVertex.description = "Quadratic friction coefficient for subgrid"

            cbfVertex = dataset.createVariable(
                "cmfVertex", "f4", ("node", "phi"), zlib=True, complevel=2
            )
            cbfVertex.description = (
                "Quadratic friction correction coefficient for subgrid"
            )

            cadvVertex = dataset.createVariable(
                "cadvVertex", "f4", ("node", "phi"), zlib=True, complevel=2
            )
            cadvVertex.description = "Advection correction coefficient for subgrid"

            dataset.title = "ADCIRC subgrid input file"
            dataset.institution = "The Water Institute"
            dataset.source = "https://github.com/waterinstitute/adcirc-subgrid"
            dataset.history = "Created by the ADCIRC subgrid preprocessor"
            dataset.references = "https://adcirc.org/"
            dataset.comment = (
                "This file contains the output of the ADCIRC subgrid preprocessor"
            )

            binaryVertexList[:] = self.__vertex_flag
            phi[:] = self.__phi_set
            wetTotWatDepthVertex[:, :] = self.__wet_water_depth
            gridTotWatDepthVertex[:, :] = self.__wet_total_depth
            cfVertex[:, :] = self.__c_f
            cbfVertex[:, :] = self.__c_bf
            cadvVertex[:, :] = self.__c_adv
