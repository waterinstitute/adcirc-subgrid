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
import argparse
import logging

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
        args.variable,
        args.water_level,
        args.show,
        args.output_filename,
        args.range,
        args.bbox,
        args.mesh_file,
        args.colorbar,
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
    mesh_plot_parser.add_argument(
        "--variable",
        help="Variable to plot",
        type=str,
        default="percent_wet",
        choices=[
            "percent_wet",
            "wet_depth",
            "total_depth",
            "cf",
            "c_mf",
            "c_adv",
            "n_avg",
        ],
    )
    mesh_plot_parser.add_argument(
        "--bbox",
        help="Bounding box for the plot (minx, miny, maxx, maxy)",
        type=float,
        nargs=4,
        default=None,
    )
    mesh_plot_parser.add_argument(
        "--range",
        help="Range for the plot (min, max)",
        type=float,
        nargs=2,
        default=None,
    )
    mesh_plot_parser.add_argument(
        "--colorbar", help="Name of the colorbar", type=str, default="jet"
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
    parser.add_argument("--verbose", help="Use verbose logging", action="store_true")
    subparsers = parser.add_subparsers(help="Sub-command help")
    initialize_preprocessor_parser(subparsers)
    initialize_node_plot_parser(subparsers)
    initialize_mesh_plot_parser(subparsers)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s] :: %(levelname)s :: %(name)s :: %(message)s",
        )

        # In debug, numba and rasterio are a bit much
        logging.getLogger("numba.core").setLevel(logging.INFO)
        logging.getLogger("rasterio").setLevel(logging.INFO)

    else:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] :: %(levelname)s :: %(name)s :: %(message)s",
        )

    args.func(args)
