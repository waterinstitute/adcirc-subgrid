from schema import Schema, Use

SUBGRID_SCHEMA = Schema(
    {
        "output_filename": Use(str),
        "adcirc_mesh": Use(str),
        "manning_lookup": Use(str),
        "min_elevation": Use(float),
        "max_elevation": Use(float),
        "elevation_step_size": Use(float),
        "dem": Use(str),
        "land_cover": Use(str),
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
