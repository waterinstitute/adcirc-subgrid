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
    prep_parser = subparsers.add_parser("prep", help="Preprocessor help")
    prep_parser.add_argument("config", help="Path to the yaml configuration file")
    prep_parser.set_defaults(func=run_preprocessor)


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
    initialize_postprocessor_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
