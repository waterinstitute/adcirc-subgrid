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
from typing import Optional

import numpy as np
from matplotlib.tri import Triangulation


def plot_mesh(
    subgrid_filename: str,
    subgrid_variable: str,
    level: float,
    show: bool,
    output_filename: str,
    plot_range: Optional[list] = None,
    bbox: Optional[list] = None,
    adcirc_mesh_file: Optional[str] = None,
    colorbar: Optional[str] = "jet",
) -> None:
    """
    Plot the mesh data for a specific level in 2D using cartopy/matplotlib

    Args:
        subgrid_filename: The subgrid netCDF filename
        subgrid_variable: The subgrid variable to plot
        level: The level to plot
        show: Whether to show the plot
        output_filename: The output filename to save the plot to
        plot_range: The range of the plot
        bbox: The bounding box to plot
        adcirc_mesh_file: The ASCII ADCIRC mesh file to use for plotting
        colorbar: The colorbar to use

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    mesh_data = read_mesh(subgrid_filename, adcirc_mesh_file)
    tri = Triangulation(
        mesh_data["node_x"], mesh_data["node_y"], mesh_data["connectivity"]
    )

    # Interpolate each dataset to the given level
    plot_data = get_plotting_data(subgrid_filename, subgrid_variable)
    interpolated_data = interpolate_data(plot_data, level, subgrid_variable)

    # Mask the triangulation based on fully dry cells and cells outside the domain
    tri_masked = mask_triangulation(tri, interpolated_data["data"], plot_data["active"])

    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    plot_params = get_plot_params(subgrid_variable, plot_range)

    plot_ticks = np.linspace(
        plot_params["plot_range"][0], plot_params["plot_range"][1], 10
    )
    contour_levels = np.linspace(
        plot_params["plot_range"][0], plot_params["plot_range"][1], 100
    )

    ax.set_title(f"{plot_params['plot_title']} at {level:.2f}m")
    f1 = ax.tricontourf(
        tri_masked,
        interpolated_data["data"],
        cmap=colorbar,
        levels=contour_levels,
        extend="both",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(f1, ax=ax, orientation="vertical", ticks=plot_ticks)
    cbar.set_label(f"{plot_params['plot_title']} ({plot_params['plot_units']})")

    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    plt.tight_layout()

    if output_filename is not None:
        plt.savefig(output_filename, dpi=300)

    if show:
        plt.show()

    plt.close()


def get_plot_params(subgrid_variable: str, plot_range: Optional[list] = None) -> dict:  # noqa: PLR0912
    """
    Get the plotting parameters for the given subgrid variable or set some reasonable defaults

    Args:
        subgrid_variable: The subgrid variable to plot
        plot_range: The range of the plot

    Returns:
        A dictionary with the plot parameters
    """
    if plot_range is None:
        if subgrid_variable == "percent_wet":
            plot_range = [0.0, 1.0]
        elif (
            subgrid_variable in ("water_levels", "wet_depth")
            or subgrid_variable == "total_depth"
        ):
            plot_range = [0.0, 10.0]
        elif subgrid_variable in ("cf", "c_mf"):
            plot_range = [0.0025, 0.25]
        elif subgrid_variable == "c_adv":
            plot_range = [0.0, 1.5]
        else:
            plot_range = [0.0, 1.0]

    if subgrid_variable == "percent_wet":
        plot_title = "Wet Fraction"
        plot_units = "%"
    elif subgrid_variable == "water_levels":
        plot_title = "Water Level"
        plot_units = "m"
    elif subgrid_variable == "wet_depth":
        plot_title = "Wet Depth"
        plot_units = "m"
    elif subgrid_variable == "total_depth":
        plot_title = "Total Depth"
        plot_units = "m"
    elif subgrid_variable == "cf":
        plot_title = "Friction Coefficient"
        plot_units = ""
    elif subgrid_variable == "c_mf":
        plot_title = "Friction Correction"
        plot_units = ""
    elif subgrid_variable == "c_adv":
        plot_title = "Advection Correction"
        plot_units = ""
    else:
        plot_title = ""
        plot_units = ""

    return {
        "plot_range": plot_range,
        "plot_title": plot_title,
        "plot_units": plot_units,
    }


def interpolate_data(data: dict, water_level: float, subgrid_variable: str) -> dict:
    """
    Interpolates the phi data from a percent wet to items that are plottable for specific water levels

    Args:
        data: The raw data dictionary
        water_level: The water level to interpolate to
        subgrid_variable: The subgrid variable to interpolate

    Returns:
        The interpolated data dictionary with 2D arrays for each variable
    """
    node_count = data["water_levels"].shape[0]

    out_data = {
        # "water_levels": np.zeros(node_count),
        # "pct_wet": np.zeros(node_count),
        "data": np.zeros(node_count),
    }

    for i in range(node_count):
        wl = data["water_levels"][i, :]

        if water_level < wl[0]:
            out_data["data"][i] = np.nan
        elif water_level > wl[-1]:
            if subgrid_variable in ("total_depth", "wet_depth"):
                out_data["data"][i] = (
                    data[subgrid_variable][i, -1] + water_level - wl[-1]
                )
            elif subgrid_variable in ("cf", "c_mf"):
                out_data["data"][i] = 0.0025
            elif subgrid_variable in ("c_adv", "percent_wet"):
                out_data["data"][i] = 1.0
        else:
            if subgrid_variable == "percent_wet":
                data_interp = np.interp(water_level, wl, data["data"][:])
            else:
                data_interp = np.interp(water_level, wl, data["data"][i, :])
            out_data["data"][i] = data_interp

    return out_data


def mask_triangulation(
    tri: Triangulation, quantity: np.ndarray, in_out_list: np.ndarray
) -> Triangulation:
    """
    Mask a triangulation based on a mask array

    Args:
        tri: The matplotlib.tri.Triangulation object
        quantity: The masking array
        in_out_list: The list of in/out values

    Returns:
        The masked Triangulation object
    """
    mask = np.logical_or(np.isnan(quantity), in_out_list == 0)
    triangle_connectivity = tri.triangles
    mask_triangles = np.any(mask[triangle_connectivity], axis=1)
    return Triangulation(tri.x, tri.y, tri.triangles[~mask_triangles, :])


def read_mesh(subgrid_filename: str, adcirc_mesh_filename: str) -> dict:
    """
    Read the mesh data from the subgrid and ADCIRC mesh files

    Args:
        subgrid_filename: The subgrid netCDF filename
        adcirc_mesh_filename: The ADCIRC mesh filename

    Returns:
        A dictionary with the mesh data
    """
    if adcirc_mesh_filename is not None:
        return read_mesh_ascii(adcirc_mesh_filename)
    else:
        return read_mesh_netcdf(subgrid_filename)


def read_mesh_ascii(filename: str) -> dict:
    """
    Read the mesh data from the ADCIRC mesh file

    Args:
        filename: The ADCIRC mesh filename

    Returns:
        A dictionary with the mesh data
    """
    from .mesh import Mesh

    mesh = Mesh(filename)
    node_x = mesh.nodes()[:, 0]
    node_y = mesh.nodes()[:, 1]
    connectivity = mesh.elements()

    return {"node_x": node_x, "node_y": node_y, "connectivity": connectivity}


def read_mesh_netcdf(filename: str) -> dict:
    """
    Read the mesh data from the subgrid netCDF file

    Args:
        filename: The subgrid netCDF filename

    Returns:
        A dictionary with the mesh data
    """
    from netCDF4 import Dataset

    with Dataset(filename, "r") as dataset:
        node_x = dataset.variables["x"][:]
        node_y = dataset.variables["y"][:]
        connectivity = dataset.variables["connectivity"][:]

    return {"node_x": node_x, "node_y": node_y, "connectivity": connectivity}


def get_plotting_data(filename: str, subgrid_variable: str) -> dict:
    """
    Read the plotting data from the subgrid netCDF file

    Args:
        filename: The subgrid netCDF filename
        subgrid_variable: The subgrid variable to plot

    Returns:
        A dictionary with the plotting data
    """
    from netCDF4 import Dataset

    variable_name_map = {
        "percent_wet": "phiSet",
        "water_levels": "wetFractionVertex",
        "wet_depth": "wetTotWatDepthVertex",
        "total_depth": "gridTotWatDepthVertex",
        "cf": "cfVertex",
        "c_mf": "cmfVertex",
        "c_adv": "cadvVertex",
    }

    with Dataset(filename, "r") as dataset:
        return {
            "active": dataset.variables["binaryVertexList"][:],
            "percent_wet": dataset.variables["phiSet"][:],
            "water_levels": dataset.variables["wetFractionVertex"][:],
            "wet_depth": dataset.variables["wetTotWatDepthVertex"][:],
            "total_depth": dataset.variables["gridTotWatDepthVertex"][:],
            "cf": dataset.variables["cfVertex"][:],
            "c_mf": dataset.variables["cmfVertex"][:],
            "c_adv": dataset.variables["cadvVertex"][:],
            "data": dataset.variables[variable_name_map[subgrid_variable]][:],
        }
