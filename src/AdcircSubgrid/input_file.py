import os

from schema import And, Optional, Schema, Use

SUBGRID_SCHEMA = Schema(
    {
        "output_filename": Use(str),  # Output filename (.nc)
        "adcirc_mesh": Use(str),  # Adcirc mesh filename (Note: Currently ascii only)
        "manning_lookup": Use(
            str
        ),  # Manning lookup table filename or 'ccap' to use the default CCAP lookup table
        Optional("n_subgrid_levels", default=11): Use(
            int
        ),  # Number of subgrid levels (default: 11)
        Optional("n_phi_levels", default=11): Use(
            int
        ),  # Number of phi levels (default: 11)
        "dem": And(
            Use(str), os.path.exists
        ),  # Digital elevation model filename. (GDAL-readable formats are supported)
        "land_cover": Use(
            str
        ),  # Land cover filename (GDAL-readable formats are supported)
        Optional("progress_bar_increment", default=10): Use(
            int
        ),  # Increment for the progress bar (default: 10)
    }
)


class InputFile:
    """
    InputFile class to read and store the input file for the subgrid preprocessor.
    """

    def __init__(self, input_file: str) -> None:
        """
        Initialize the InputFile class using a YAML input file

        Args:
            input_file: File path to the input file
        """
        self.__input_file = input_file
        self.__data = InputFile.__read_yaml_input(self.__input_file)

    @staticmethod
    def __read_yaml_input(input_file: str) -> dict:
        """
        Read a YAML input file and validate the schema

        Args:
            input_file: File path to the input file

        Returns: The data from the input file as a dictionary
        """
        import yaml

        with open(input_file) as file:
            return SUBGRID_SCHEMA.validate(yaml.safe_load(file))

    def data(self) -> dict:
        """
        Get the data from the input file

        Returns:
            The data from the input file as a dictionary
        """
        return self.__data
