import logging

logger = logging.getLogger(__name__)


def node_inspector(
    filename: str, node: int, basis: str, show: bool, save: str, index_type: int = 1
) -> None:
    """
    Inspect the subgrid variables for a node

    Args:
        filename: The name of the subgrid netCDF file
        node: The node number to inspect
        basis: The basis for plotting. Either wse or phi
        show: Whether to show the plot
        save: The name of the file to save the plot to
        index_type: The type of indexing to use (one-based or zero-based)
    """
    import matplotlib.pyplot as plt

    from .subgrid_output_file import SubgridOutputFile

    subgrid_data = SubgridOutputFile.read(filename)

    # Note that ADCIRC uses 1-based indexing for nodes, so we need to
    # subtract 1 to get the correct index
    if index_type == 1:
        node_index = node - 1
    elif index_type == 0:
        node_index = node
    else:
        logger.error(f"Invalid index type: {index_type}")
        return

    vertex_data = subgrid_data.get_vertex(node_index)

    if not vertex_data["resident"]:
        logger.error(f"Node '{node_index}' is not resident in the subgrid")
        return

    if basis == "phi":
        x_basis = vertex_data["phi"]
        x_basis_label = "Phi"
    elif basis == "wse":
        x_basis = vertex_data["water_level"]
        x_basis_label = "Water Level (m)"
    else:
        logger.error(f"Invalid basis: {basis}")
        return

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(x_basis, vertex_data["wet_water_depth"], marker="o")
    ax[0, 0].plot(x_basis, vertex_data["wet_total_depth"], marker="o")
    ax[0, 0].set_xlim(x_basis[0], x_basis[-1])
    ax[0, 0].set_xlabel(x_basis_label)
    ax[0, 0].set_ylabel("Wet Water Depth (m)")
    ax[0, 0].grid()

    ax[0, 1].plot(x_basis, vertex_data["c_f"], marker="o")
    ax[0, 1].set_xlim(x_basis[0], x_basis[-1])
    ax[0, 1].set_xlabel(x_basis_label)
    ax[0, 1].set_ylabel("C_f")
    ax[0, 1].grid()

    ax[1, 0].plot(x_basis, vertex_data["c_adv"], marker="o")
    ax[1, 0].set_xlim(x_basis[0], x_basis[-1])
    ax[1, 0].set_xlabel(x_basis_label)
    ax[1, 0].set_ylabel("C_adv")
    ax[1, 0].grid()

    ax[1, 1].plot(x_basis, vertex_data["c_bf"], marker="o")
    ax[1, 1].set_xlim(x_basis[0], x_basis[-1])
    ax[1, 1].set_xlabel(x_basis_label)
    ax[1, 1].set_ylabel("C_bf")
    ax[1, 1].grid()

    fig.suptitle(f"Node {node_index} Subgrid Variables")

    plt.tight_layout()
    plt.show()
