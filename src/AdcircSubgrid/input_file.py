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
import os

from schema import And, Optional, Schema, Use

SUBGRID_SCHEMA = Schema(
    {
        "input": {
            "adcirc_mesh": And(Use(str), os.path.exists),
            Optional("nodal_attributes", default=None): Use(str),
            "manning_lookup": Use(str),
            "dem": And(Use(str), os.path.exists),
            "land_cover": And(Use(str), os.path.exists),
        },
        "output": {
            "filename": Use(str),
            Optional("progress_bar_increment", default=10): Use(int),
        },
        "options": {
            Optional("n_subgrid_levels", default=11): Use(int),
            Optional("n_phi_levels", default=11): Use(int),
            Optional("subgrid_level_distribution", default="linear"): And(
                Use(str), lambda x: x in ["linear", "histogram"]
            ),
            Optional("distribution_factor", default=1.0): And(
                Use(float), lambda x: x > 0.0
            ),
            Optional("existing_subgrid", default=None): And(Use(str), os.path.exists),
            Optional("nodal_attribute_manning_override", default=False): Use(bool),
            Optional("nodal_attribute_depth_min", default=2.0): Use(float),
            Optional("nodal_attribute_manning_max", default=0.025): Use(float),
        },
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
