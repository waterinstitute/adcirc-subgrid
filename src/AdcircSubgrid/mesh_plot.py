from typing import Optional

import numpy as np
from matplotlib.tri import Triangulation

COLORBAR = "jet"


def plot_mesh(  # noqa: PLR0915
    subgrid_filename: str,
    level: float,
    show: bool,
    output_filename: str,
    adcirc_mesh_file: Optional[str] = None,
) -> None:
    """
    Plot the mesh data for a specific level in 2D using cartopy/matplotlib

    Args:
        subgrid_filename: The subgrid netCDF filename
        level: The level to plot
        show: Whether to show the plot
        output_filename: The output filename to save the plot to
        adcirc_mesh_file: The ASCII ADCIRC mesh file to use for plotting

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    from netCDF4 import Dataset

    from .mesh import Mesh

    # If the adcirc mesh file is provided, read the data
    if adcirc_mesh_file is not None:
        mesh = Mesh(adcirc_mesh_file)
        node_x = mesh.nodes()[:, 0]
        node_y = mesh.nodes()[:, 1]
        # node_z = mesh.nodes()[:, 2]
        connectivity = mesh.elements()
    else:
        with Dataset(subgrid_filename, "r") as dataset:
            node_x = dataset.variables["x"][:]
            node_y = dataset.variables["y"][:]
            # node_z = dataset.variables["depth"][:]
            connectivity = dataset.variables["connectivity"][:]

    # Read the data from the netCDF file
    with Dataset(subgrid_filename, "r") as dataset:
        raw_data = {
            "percent_wet": dataset.variables["phiSet"][:],
            "water_levels": dataset.variables["wetFractionVertex"][:],
            "wet_depth": dataset.variables["wetTotWatDepthVertex"][:],
            "total_depth": dataset.variables["gridTotWatDepthVertex"][:],
            "cf": dataset.variables["cfVertex"][:],
            "c_mf": dataset.variables["cmfVertex"][:],
            "c_adv": dataset.variables["cadvVertex"][:],
        }

    # Create a base triangulation
    tri = Triangulation(node_x, node_y, connectivity)

    # Interpolate each dataset to the given level
    interpolated_data = interpolate_phi_data(raw_data, level)

    # Mask the triangulation based on fully dry cells
    wf_tri = mask_triangulation(tri, interpolated_data["pct_wet"])

    # Plot the data
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    wf_ticks = np.linspace(0, 1, 10)
    wf_levels = np.linspace(0, 1, 100)

    cf_ticks = np.linspace(0, 0.2, 10)
    cf_levels = np.linspace(0, 0.2, 100)

    c_mf_ticks = np.linspace(0, 0.2, 10)
    c_mf_levels = np.linspace(0, 0.2, 100)

    c_adv_ticks = np.linspace(0.8, 1.2, 10)
    c_adv_levels = np.linspace(0.8, 1.2, 100)

    ax[0, 0].set_title("Wet Fraction")
    f1 = ax[0, 0].tricontourf(
        wf_tri,
        interpolated_data["pct_wet"],
        cmap=COLORBAR,
        levels=wf_levels,
        extend="both",
    )
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_xlabel("Longitude")
    ax[0, 0].set_ylabel("Latitude")
    fig.colorbar(f1, ax=ax[0, 0], orientation="vertical", ticks=wf_ticks)

    ax[0, 1].set_title("Wet Depth")
    f2 = ax[0, 1].tricontourf(
        wf_tri,
        interpolated_data["wet_depth"],
        cmap=COLORBAR,
        levels=100,
        extend="both",
    )
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set_xlabel("Longitude")
    ax[0, 1].set_ylabel("Latitude")
    fig.colorbar(f2, ax=ax[0, 1], orientation="vertical")

    ax[1, 1].set_title("Total Depth")
    f3 = ax[1, 1].tricontourf(
        wf_tri,
        interpolated_data["total_depth"],
        cmap=COLORBAR,
        levels=100,
        extend="both",
    )
    ax[1, 1].set_aspect("equal")
    ax[1, 1].set_xlabel("Longitude")
    ax[1, 1].set_ylabel("Latitude")
    fig.colorbar(f3, ax=ax[1, 1], orientation="vertical")

    ax[1, 0].set_title("Friction Coefficient")
    f4 = ax[1, 0].tricontourf(
        wf_tri,
        interpolated_data["cf"],
        levels=cf_levels,
        cmap=COLORBAR,
        extend="both",
    )
    ax[1, 0].set_aspect("equal")
    ax[1, 0].set_xlabel("Longitude")
    ax[1, 0].set_ylabel("Latitude")
    fig.colorbar(f4, ax=ax[1, 0], orientation="vertical", ticks=cf_ticks)

    ax[2, 0].set_title("Friction Correction")
    f5 = ax[2, 0].tricontourf(
        wf_tri,
        interpolated_data["c_mf"],
        levels=c_mf_levels,
        cmap=COLORBAR,
        extend="both",
    )
    ax[2, 0].set_aspect("equal")
    ax[2, 0].set_xlabel("Longitude")
    ax[2, 0].set_ylabel("Latitude")
    fig.colorbar(f5, ax=ax[2, 0], orientation="vertical", ticks=c_mf_ticks)

    ax[2, 1].set_title("Advection Correction")
    f6 = ax[2, 1].tricontourf(
        wf_tri,
        interpolated_data["c_adv"],
        levels=c_adv_levels,
        cmap=COLORBAR,
        extend="both",
    )
    ax[2, 1].set_aspect("equal")
    ax[2, 1].set_xlabel("Longitude")
    ax[2, 1].set_ylabel("Latitude")
    fig.colorbar(f6, ax=ax[2, 1], orientation="vertical", ticks=c_adv_ticks)

    plt.tight_layout()

    if show:
        plt.show()

    if output_filename is not None:
        plt.savefig(output_filename, dpi=300)


def interpolate_phi_data(data: dict, water_level: float) -> dict:
    """
    Interpolates the phi data from a percent wet to items that are plottable for specific water levels

    Args:
        data: The raw data dictionary
        water_level: The water level to interpolate to

    Returns:
        The interpolated data dictionary with 2D arrays for each variable
    """
    percent_levels = data["percent_wet"]
    node_count = data["water_levels"].shape[0]

    out_data = {
        "water_levels": np.zeros(node_count),
        "pct_wet": np.zeros(node_count),
        "wet_depth": np.zeros(node_count),
        "total_depth": np.zeros(node_count),
        "cf": np.zeros(node_count),
        "c_mf": np.zeros(node_count),
        "c_adv": np.zeros(node_count),
    }

    for i in range(node_count):
        wl = data["water_levels"][i, :]

        if water_level < wl[0]:
            out_data["water_levels"][i] = np.nan
            out_data["wet_depth"][i] = np.nan
            out_data["total_depth"][i] = np.nan
            out_data["cf"][i] = np.nan
            out_data["c_mf"][i] = np.nan
            out_data["c_adv"][i] = np.nan
            out_data["pct_wet"][i] = np.nan
            continue

        pct_interp = np.interp(water_level, wl, percent_levels)
        wet_depth_interp = np.interp(water_level, wl, data["wet_depth"][i, :])
        total_depth_interp = np.interp(water_level, wl, data["total_depth"][i, :])
        cf_interp = np.interp(water_level, wl, data["cf"][i, :])
        c_mf_interp = np.interp(water_level, wl, data["c_mf"][i, :])
        c_adv_interp = np.interp(water_level, wl, data["c_adv"][i, :])

        out_data["water_levels"][i] = water_level
        out_data["pct_wet"][i] = pct_interp
        out_data["wet_depth"][i] = wet_depth_interp
        out_data["total_depth"][i] = total_depth_interp
        out_data["cf"][i] = cf_interp
        out_data["c_mf"][i] = c_mf_interp
        out_data["c_adv"][i] = c_adv_interp

    return out_data


def mask_triangulation(tri: Triangulation, quantity: np.ndarray) -> Triangulation:
    """
    Mask a triangulation based on a mask array

    Args:
        tri: The matplotlib.tri.Triangulation object
        quantity: The masking array

    Returns:
        The masked Triangulation object
    """
    mask = np.isnan(quantity)
    triangle_connectivity = tri.triangles
    mask_triangles = np.any(mask[triangle_connectivity], axis=1)
    return Triangulation(tri.x, tri.y, tri.triangles[~mask_triangles, :])
