import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] :: %(levelname)s :: %(name)s :: %(message)s",
)

logger = logging.getLogger(__name__)


def run_preprocessor(args: argparse.Namespace) -> None:
    """
    Run the subgrid preprocessor
    Args:
        args: An argparse.Namespace object
    """
    from .input_file import InputFile
    from .preprocessor import SubgridPreprocessor

    logger.info(f"Running preprocessor with config file {args.config}")

    preprocessor = SubgridPreprocessor(InputFile(args.config), args.window_memory)
    preprocessor.process()
    preprocessor.write()


def run_mesh_plot(args: argparse.Namespace) -> None:
    """
    Run the subgrid postprocessor
    Args:
        args: An argparse.Namespace object
    """
    from .mesh_plot import plot_mesh

    plot_mesh(
        args.filename,
        args.water_level,
        args.show,
        args.output_filename,
        args.mesh_file,
    )


def initialize_preprocessor_parser(subparsers) -> None:  # noqa: ANN001
    """
    Initialize the preprocessor parser

    Args:
        subparsers: The subparsers object
    """
    prep_parser = subparsers.add_parser(
        "prep", help="Process data into subgrid variables"
    )
    prep_parser.add_argument("config", help="Path to the yaml configuration file")
    prep_parser.add_argument(
        "--window-memory",
        help="Raster window memory limit (approx) in mb (default=64)",
        type=int,
        default=64,
        required=False,
    )
    prep_parser.set_defaults(func=run_preprocessor)


def initialize_node_plot_parser(subparsers) -> None:  # noqa: ANN001
    """
    Initialize the node-plot parser

    Args:
        subparsers: The subparsers object
    """
    node_plot_parser = subparsers.add_parser(
        "plot-node", help="Plot subgrid variables for a node"
    )
    node_plot_parser.add_argument(
        "--filename", help="Name of the subgrid netCDF file", type=str, required=True
    )
    node_plot_parser.add_argument("--show", help="Show the plot", action="store_true")
    node_plot_parser.add_argument(
        "--save", help="Save the plot to a file", type=str, default=None
    )
    node_plot_parser.add_argument(
        "--node", help="Node number to plot", type=int, default=None, required=True
    )
    node_plot_parser.add_argument(
        "--basis", help="Basis for plotting (wse or phi) (default=phi)", type=str
    )
    node_plot_parser.add_argument(
        "--index-base",
        help="Index base (0 [ie. netCDF] or 1 [ie. ADCIRC]) (default=0)",
        type=int,
        default=0,
    )
    node_plot_parser.set_defaults(func=run_node_plot)


def run_node_plot(args: argparse.Namespace) -> None:
    """
    Run the subgrid node plotter

    Args:
        args: An argparse.Namespace object
    """
    from .node_plot import node_plot

    if not args.show and not args.save:
        msg = "Either --show or --save must be specified"
        raise ValueError(msg)

    node_plot(
        args.filename, args.node, args.basis, args.show, args.save, args.index_base
    )


def initialize_mesh_plot_parser(subparsers) -> None:  # noqa: ANN001
    """
    Initialize the mesh plot parser

    Args:
        subparsers: The subparsers object
    """
    mesh_plot_parser = subparsers.add_parser("plot-mesh", help="Mesh plot help")
    mesh_plot_parser.add_argument(
        "filename", help="Name of the subgrid netCDF file", type=str
    )
    mesh_plot_parser.add_argument(
        "--water-level", help="Level to plot", type=float, default=0.0
    )
    mesh_plot_parser.add_argument(
        "--mesh-file",
        help="ADCIRC mesh file for plotting if not in the subgrid file",
        type=str,
        default=None,
    )
    mesh_plot_parser.add_argument("--show", help="Show the plot", action="store_true")
    mesh_plot_parser.add_argument(
        "--output-filename",
        help="Output filename for the plot",
        type=str,
        default=None,
    )
    mesh_plot_parser.set_defaults(func=run_mesh_plot)


def cli_main() -> None:
    """
    Entry point for the subgrid CLI
    """
    import argparse

    parser = argparse.ArgumentParser(description="Adcirc Subgrid Processor")
    subparsers = parser.add_subparsers(help="Sub-command help")
    initialize_preprocessor_parser(subparsers)
    initialize_node_plot_parser(subparsers)
    initialize_mesh_plot_parser(subparsers)
    args = parser.parse_args()
    args.func(args)
