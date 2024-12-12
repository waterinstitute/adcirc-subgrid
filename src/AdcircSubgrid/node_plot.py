import logging

import numpy as np

logger = logging.getLogger(__name__)


def node_plot(
    filename: str,
    node: int,
    basis: str,
    show: bool,
    save_filename: str,
    index_type: int = 1,
) -> None:
    """
    Plot the subgrid variables for a node

    Args:
        filename: The name of the subgrid netCDF file
        node: The node number to plot
        basis: The basis for plotting. Either wse or phi
        show: Whether to show the plot
        save_filename: The name of the file to save the plot to
        index_type: The type of indexing to use (one-based or zero-based)
    """
    import matplotlib.pyplot as plt

    from .subgrid_output_file import SubgridOutputFile

    subgrid_data = SubgridOutputFile.read(filename)
    node_index = __get_node_index(index_type, node)
    vertex_data = subgrid_data.get_vertex(node_index)

    if not vertex_data["resident"]:
        logger.error(f"Node '{node_index}' is not resident in the subgrid")
        return

    x_basis, x_basis_label, plot_marker = __get_x_axis(basis, vertex_data)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(x_basis, vertex_data["wet_water_depth"], marker=plot_marker, label="Wet Water Depth")
    ax[0, 0].plot(x_basis, vertex_data["wet_total_depth"], marker=plot_marker, label="Total Water Depth")
    ax[0, 0].set_xlim(x_basis[0], x_basis[-1])
    ax[0, 0].set_xlabel(x_basis_label)
    ax[0, 0].set_ylabel("Wet Water Depth (m)")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].plot(x_basis, vertex_data["c_f"], marker=plot_marker)
    ax[0, 1].set_xlim(x_basis[0], x_basis[-1])
    ax[0, 1].set_xlabel(x_basis_label)
    ax[0, 1].set_ylabel("C_f")
    ax[0, 1].grid()

    ax[1, 0].plot(x_basis, vertex_data["c_adv"], marker=plot_marker)
    ax[1, 0].set_xlim(x_basis[0], x_basis[-1])
    ax[1, 0].set_xlabel(x_basis_label)
    ax[1, 0].set_ylabel("C_adv")
    ax[1, 0].grid()

    ax[1, 1].plot(x_basis, vertex_data["c_bf"], marker=plot_marker)
    ax[1, 1].set_xlim(x_basis[0], x_basis[-1])
    ax[1, 1].set_xlabel(x_basis_label)
    ax[1, 1].set_ylabel("C_bf")
    ax[1, 1].grid()

    fig.suptitle(f"Node {node_index} Subgrid Variables")

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)

    if show:
        plt.show()


def __get_x_axis(basis: str, vertex_data: dict) -> tuple[np.ndarray, str, str]:
    """
    Get the x-axis data for the plot based on the selected basis

    Args:
        basis: The basis for plotting. Either wse or phi
        vertex_data: The vertex data

    Returns:
        The x-axis data, the x-axis label, and the plot marker
    """
    if basis == "phi":
        x_basis = vertex_data["phi"]
        x_basis_label = "Phi"
    elif basis == "wse":
        x_basis = vertex_data["water_level"]
        x_basis_label = "Water Level (m)"
    else:
        msg = "Basis must be either 'wse' or 'phi'"
        raise ValueError(msg)

    # If we have more than 100 phi levels, we won't use a marker
    if len(x_basis) > 100:
        plot_marker = None
    else:
        plot_marker = "o"

    return x_basis, x_basis_label, plot_marker


def __get_node_index(index_type: int, node: int) -> int:
    """
    Get the node index based on the index type

    Args:
        index_type: The type of indexing to use (0 or 1)
        node: The node number

    Returns:
        The node index
    """
    # Note that ADCIRC uses 1-based indexing for nodes, so we need to
    # subtract 1 to get the correct index
    if index_type == 1:
        node_index = node - 1
    elif index_type == 0:
        node_index = node
    else:
        msg = "Index type must be either 0 (zero-based) or 1 (one-based)"
        raise ValueError(msg)
    return node_index
