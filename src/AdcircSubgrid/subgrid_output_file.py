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
import netCDF4
import numpy as np
from netCDF4 import Dataset

from .mesh import Mesh
from .subgrid_data import SubgridData


class SubgridOutputFile:
    """
    Class to store the output of the subgrid calculations
    """

    def __init__(self) -> None:
        """
        Empty constructor
        """

    @staticmethod
    def write(sg_data: SubgridData, mesh: Mesh, output_file: str) -> None:  # noqa: PLR0915
        """
        Write the output data to a file

        Args:
            sg_data: The SubgridData object to write to the file
            mesh: The Mesh object to use for the output
            output_file: The output file to write the data to
        """
        with Dataset(output_file, "w", format="NETCDF4") as dataset:
            # Write the mesh data to the netCDF file
            SubgridOutputFile.__write_mesh(mesh, dataset)

            dataset.createDimension("numPhi", sg_data.phi_count())

            binary_vertex_list = dataset.createVariable(
                "binaryVertexList", "i4", ("numNode",), zlib=True, complevel=2
            )
            binary_vertex_list.description = "Vertex subgrid residency flag"

            phi = dataset.createVariable(
                "phiSet", "f4", ("numPhi",), zlib=True, complevel=2
            )
            phi.description = "Percent wet for the subgrid element"

            # manning = dataset.createVariable(
            #     "manning_avg", "f4", ("numNode",), zlib=True, complevel=2
            # )
            manning = dataset.createVariable(
                "manningAvg", "f4", ("numNode",), zlib=True, complevel=2
            )

            manning.description = "Grid averaged manning for the vertex area"

            wet_fraction_elevation = dataset.createVariable(
                "wetFractionVertex",
                "f4",
                ("numNode", "numPhi"),
                zlib=True,
                complevel=2,
            )
            wet_fraction_elevation.description = (
                "Water surface elevation for the phi wet fraction"
            )

            wet_tot_wat_depth_vertex = dataset.createVariable(
                "wetTotWatDepthVertex",
                "f4",
                ("numNode", "numPhi"),
                zlib=True,
                complevel=2,
            )
            wet_tot_wat_depth_vertex.description = (
                "Depth sum divided by the number of wet pixels in the subgrid"
            )

            grid_tot_wat_depth_vertex = dataset.createVariable(
                "gridTotWatDepthVertex",
                "f4",
                ("numNode", "numPhi"),
                zlib=True,
                complevel=2,
            )
            grid_tot_wat_depth_vertex.description = (
                "Depth sum divided by the total number of pixels in the subgrid"
            )

            cf_vertex = dataset.createVariable(
                "cfVertex", "f4", ("numNode", "numPhi"), zlib=True, complevel=2
            )
            cf_vertex.description = "Quadratic friction coefficient for subgrid"

            cbf_vertex = dataset.createVariable(
                "cmfVertex", "f4", ("numNode", "numPhi"), zlib=True, complevel=2
            )
            cbf_vertex.description = (
                "Quadratic friction correction coefficient for subgrid"
            )

            cadv_vertex = dataset.createVariable(
                "cadvVertex", "f4", ("numNode", "numPhi"), zlib=True, complevel=2
            )
            cadv_vertex.description = "Advection correction coefficient for subgrid"

            dataset.title = "ADCIRC subgrid input file"
            dataset.institution = "The Water Institute"
            dataset.source = "https://github.com/waterinstitute/adcirc-subgrid"
            dataset.history = "Created by the ADCIRC subgrid preprocessor"
            dataset.references = "https://adcirc.org/"
            dataset.comment = (
                "This file contains the output of the ADCIRC subgrid preprocessor"
            )

            # Write the variables to netCDF and replace data where the
            # vertex is not included in the subgrid with the fill value
            binary_vertex_list[:] = sg_data.vertex_flag()

            # Make a 2D mask to fill for all nodes not in the vertex flag
            mask2d = np.repeat(
                sg_data.vertex_flag()[:, np.newaxis], sg_data.phi_count(), axis=1
            )
            mask2d = mask2d == 0

            phi[:] = sg_data.phi()

            manning[:] = sg_data.man_avg()

            wet_fraction_elevation_out = sg_data.water_level()
            wet_fraction_elevation_out[mask2d] = netCDF4.default_fillvals["f4"]
            wet_fraction_elevation[:, :] = wet_fraction_elevation_out

            wet_tot_wat_depth_vertex_out = sg_data.wet_water_depth()
            wet_tot_wat_depth_vertex_out[mask2d] = netCDF4.default_fillvals["f4"]
            wet_tot_wat_depth_vertex[:, :] = wet_tot_wat_depth_vertex_out

            grid_tot_wat_depth_vertex_out = sg_data.wet_total_depth()
            grid_tot_wat_depth_vertex_out[mask2d] = netCDF4.default_fillvals["f4"]
            grid_tot_wat_depth_vertex[:, :] = grid_tot_wat_depth_vertex_out

            cf_vertex_out = sg_data.c_f()
            cf_vertex_out[mask2d] = netCDF4.default_fillvals["f4"]
            cf_vertex[:, :] = cf_vertex_out

            cbf_vertex_out = sg_data.c_bf()
            cbf_vertex_out[mask2d] = netCDF4.default_fillvals["f4"]
            cbf_vertex[:, :] = cbf_vertex_out

            cadv_vertex_out = sg_data.c_adv()
            cadv_vertex_out[mask2d] = netCDF4.default_fillvals["f4"]
            cadv_vertex[:, :] = cadv_vertex_out

    @staticmethod
    def __write_mesh(mesh: Mesh, dataset: Dataset) -> None:
        """
        Write the mesh data to the netCDF file

        Args:
            mesh: The Mesh object to write to the file
            dataset: The netCDF dataset object to write to
        """
        dataset.createDimension("numNode", mesh.num_nodes())
        dataset.createDimension("numElem", mesh.num_elements())
        dataset.createDimension("numVert", 3)

        node_x = dataset.createVariable("x", "f8", ("numNode",), zlib=True, complevel=2)
        node_x.description = "Node x-coordinate"
        node_x.units = "degrees east"
        node_x.axis = "X"

        node_y = dataset.createVariable("y", "f8", ("numNode",), zlib=True, complevel=2)
        node_y.description = "Node y-coordinate"
        node_y.units = "degrees north"
        node_y.axis = "Y"

        node_z = dataset.createVariable(
            "depth", "f8", ("numNode",), zlib=True, complevel=2
        )
        node_z.description = "Node depth"
        node_z.units = "meters"
        node_z.axis = "Z"

        connectivity = dataset.createVariable(
            "connectivity", "i4", ("numElem", "numVert"), zlib=True, complevel=2
        )
        connectivity.description = "Element connectivity"

        # Write the mesh data to the netCDF file
        node_x[:] = mesh.nodes()[:, 0]
        node_y[:] = mesh.nodes()[:, 1]
        node_z[:] = mesh.nodes()[:, 2]
        connectivity[:, :] = mesh.elements()

    @staticmethod
    def read(filename: str) -> SubgridData:
        """
        Read the output data from a file

        Args:
            filename: The filename to read the data from

        Returns:
            The SubgridData object
        """
        with Dataset(filename, "r") as dataset:
            # Get the dimensioning information
            node_count = dataset.dimensions["numNode"].size
            phi_count = dataset.dimensions["numPhi"].size
            sg_count = phi_count

            # Note that we set the sg_count and phi_count equal since
            # there is information loss when writing the data to a file
            sg_data = SubgridData(node_count, sg_count, phi_count)

            # Read the data from the file
            vertex_flag = dataset.variables["binaryVertexList"][:].data
            phi = dataset.variables["phiSet"][:].data
            water_levels = dataset.variables["wetFractionVertex"][:].data
            wet_water_depth = dataset.variables["wetTotWatDepthVertex"][:].data
            wet_total_depth = dataset.variables["gridTotWatDepthVertex"][:].data
            c_f = dataset.variables["cfVertex"][:].data
            c_bf = dataset.variables["cmfVertex"][:].data
            c_adv = dataset.variables["cadvVertex"][:].data
            # man_avg = dataset.variables["manning_avg"][:].data
            man_avg = dataset.variables["manningAvg"][:].data

            # Set the data in the SubgridData object
            sg_data.set_data(
                vertex_flag,
                phi,
                water_levels,
                wet_water_depth,
                wet_total_depth,
                c_f,
                c_bf,
                c_adv,
                man_avg,
            )

        return sg_data
