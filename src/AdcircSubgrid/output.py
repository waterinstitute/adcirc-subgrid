import numpy as np


class SubgridOutput:
    """
    Class to store the output of the subgrid calculations
    """

    def __init__(self, node_count: int, phi_count: int):
        """
        Initialize the SubgridOutput class

        Args:
            node_count: Number of nodes in the subgrid
            phi_count: Number of phi values to store
        """
        self.__node_count = node_count
        self.__phi_count = phi_count
        self.__phi_set = np.linspace(0.0, 1.0, self.__phi_count)
        self.__wet_fraction = np.zeros((self.__node_count, self.__phi_count))
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
            c_f: The c_f values
            c_bf: The c_mf values
            c_adv: The c_adv values
        """

        # Check if the shape of the input arrays is correct
        if (
            wet_fraction.shape != (self.__phi_count,)
            or wet_water_depth.shape != (self.__phi_count,)
            or wet_total_depth.shape != (self.__phi_count,)
            or c_f.shape != (self.__phi_count,)
            or c_bf.shape != (self.__phi_count,)
            or c_adv.shape != (self.__phi_count,)
        ):
            raise ValueError("Invalid shape for input arrays")

        self.__vertex_flag[vertex] = 1
        self.__wet_fraction[vertex] = wet_fraction
        self.__wet_water_depth[vertex] = wet_water_depth
        self.__wet_total_depth[vertex] = wet_total_depth
        self.__c_f[vertex] = c_f
        self.__c_bf[vertex] = c_bf
        self.__c_adv[vertex] = c_adv

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
            dataset.comment = "This file contains the output of the ADCIRC subgrid preprocessor"

            binaryVertexList[:] = self.__vertex_flag
            phi[:] = self.__phi_set
            wetTotWatDepthVertex[:, :] = self.__wet_water_depth
            gridTotWatDepthVertex[:, :] = self.__wet_total_depth
            cfVertex[:, :] = self.__c_f
            cbfVertex[:, :] = self.__c_bf
            cadvVertex[:, :] = self.__c_adv
