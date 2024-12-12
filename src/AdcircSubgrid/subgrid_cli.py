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

    preprocessor = SubgridPreprocessor(InputFile(args.config))
    preprocessor.process()
    preprocessor.write()


def run_postprocessor(args: argparse.Namespace) -> None:
    """
    Run the subgrid postprocessor
    Args:
        args: An argparse.Namespace object
    """
    msg = "Postprocessor not implemented"
    raise NotImplementedError(msg)


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
    prep_parser.set_defaults(func=run_preprocessor)


def initialize_inspection_parser(subparsers) -> None:  # noqa: ANN001
    """
    Initialize the inspection parser

    Args:
        subparsers: The subparsers object
    """
    insp_parser = subparsers.add_parser(
        "inspect", help="Inspect subgrid variables for a node"
    )
    insp_parser.add_argument(
        "--filename", help="Name of the subgrid netCDF file", type=str, required=True
    )
    insp_parser.add_argument("--show", help="Show the plot", action="store_true")
    insp_parser.add_argument(
        "--save", help="Save the plot to a file", type=str, default=None
    )
    insp_parser.add_argument(
        "--node", help="Node number to inspect", type=int, default=None, required=True
    )
    insp_parser.add_argument(
        "--basis", help="Basis for plotting (wse or phi) (default=phi)", type=str
    )
    insp_parser.add_argument(
        "--index-base",
        help="Index base (0 [ie. netCDF] or 1 [ie. ADCIRC]) (default=0)",
        type=int,
        default=0,
    )
    insp_parser.set_defaults(func=run_inspection)


def run_inspection(args: argparse.Namespace) -> None:
    """
    Run the subgrid inspection

    Args:
        args: An argparse.Namespace object
    """
    from .node_inspector import node_inspector

    node_inspector(
        args.filename, args.node, args.basis, args.show, args.save, args.index_base
    )


def initialize_postprocessor_parser(subparsers) -> None:  # noqa: ANN001
    """
    Initialize the postprocessor parser

    Args:
        subparsers: The subparsers object
    """
    post_parser = subparsers.add_parser("post", help="Postprocessor help")
    post_parser.set_defaults(func=run_postprocessor)


def cli_main() -> None:
    """
    Entry point for the subgrid CLI
    """
    import argparse

    parser = argparse.ArgumentParser(description="Adcirc Subgrid Processor")
    subparsers = parser.add_subparsers(help="Sub-command help")
    initialize_preprocessor_parser(subparsers)
    initialize_inspection_parser(subparsers)
    initialize_postprocessor_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
