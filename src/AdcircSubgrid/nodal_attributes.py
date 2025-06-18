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
from typing import Union

import numpy as np


class NodalAttribute:
    """
    Class to handle a single ADCIRC nodal attribute
    """

    def __init__(
        self,
        name: str,
        units: str,
        default_value: Union[float, list[float]],
        values: np.array,
    ) -> None:
        """
        Initialize the NodalAttribute class

        Args:
            name: Name of the nodal attribute
            units: Units of the nodal attribute
            default_value: Default value of the nodal attribute
            values: Nodal attribute values as a numpy array
        """
        # Validate that the number of default values matches the number of columns in values
        if isinstance(default_value, list):
            if len(default_value) != values.shape[1]:
                msg = "Number of default values must match number of columns in values"
                raise ValueError(msg)
        elif isinstance(default_value, float) and values.shape[1] != 1:
            msg = "Number of default values must match number of columns in values"
            raise ValueError(msg)

        self.__name = name
        self.__units = units
        self.__default_value = default_value
        self.__values = values

    def name(self) -> str:
        """
        Get the name of the nodal attribute

        Returns: The name of the nodal attribute
        """
        return self.__name

    def units(self) -> str:
        """
        Get the units of the nodal attribute

        Returns: The units of the nodal attribute
        """
        return self.__units

    def default_value(self) -> Union[float, list[float]]:
        """
        Get the default value of the nodal attribute

        Returns: The default value of the nodal attribute
        """
        return self.__default_value

    def values(self) -> np.array:
        """
        Get the nodal attribute values

        Returns: The nodal attribute values as a numpy array
        """
        return self.__values


class NodalAttributes:
    """
    Class to handle ADCIRC nodal attributes files
    """

    def __init__(self, filename: str) -> None:
        """
        Initialize the NodalAttributes class using a ADCIRC nodal attributes file

        Args:
            filename: File path to the ADCIRC nodal attributes file
        """
        self.__filename = filename
        self.__attributes: dict[str, NodalAttribute] = {}
        self.__data = NodalAttributes.__read_nodal_attributes(self.__filename)

    def attribute(self, name: str) -> NodalAttribute:
        """
        Get a nodal attribute by name

        Args:
            name: Name of the nodal attribute

        Returns: The nodal attribute
        """
        return self.__data[name]

    def attribute_names(self) -> list[str]:
        """
        Get a list of all nodal attribute names

        Returns: A list of all nodal attribute names
        """
        return list(self.__data.keys())

    @staticmethod
    def __read_nodal_attributes(filename: str) -> dict:
        """
        Read a ADCIRC nodal attributes file

        Args:
            filename: File path to the ADCIRC nodal attributes file

        Returns: The data from the ADCIRC nodal attributes file as a dictionary
        """
        data = {}
        with open(filename) as file:
            _ = file.readline()  # Header line
            n_nodes = int(file.readline().strip())
            n_attributes = int(file.readline().strip())

            metadata = {}

            for _ in range(n_attributes):
                name = file.readline().strip()
                unit = file.readline().strip()
                n_value = int(file.readline().strip())

                if n_value == 1:
                    default_value = float(file.readline().strip())
                else:
                    default_value = [
                        float(val) for val in file.readline().strip().split()
                    ]

                metadata[name] = {
                    "units": unit,
                    "n_values": n_value,
                    "default_values": default_value,
                }

            # If we didn't find a manning attribute, raise an error since
            # we haven't implemented other ways to deal with this yet
            if "mannings_n_at_sea_floor" not in metadata:
                msg = "No manning attribute found in nodal attributes file"
                raise ValueError(msg)

            for _ in range(n_attributes):
                this_name = file.readline().strip()
                n_non_default = int(file.readline().strip())
                values = np.zeros((n_nodes, metadata[this_name]["n_values"]))

                # Set all values to the default value initially
                values[:, :] = metadata[this_name]["default_values"]

                for _ in range(n_non_default):
                    line = file.readline().strip().split()
                    node_id = int(line[0]) - 1
                    if metadata[this_name]["n_values"] == 1:
                        v = float(line[1])
                    else:
                        v = [float(val) for val in line[1:]]
                    values[node_id, :] = np.array(v)

                data[this_name] = NodalAttribute(
                    name=this_name,
                    units=metadata[this_name]["units"],
                    default_value=metadata[this_name]["default_values"],
                    values=values,
                )

        # This may change later, but for now we only need "mannings_n_at_sea_floor"
        # Drop any other attributes we may have read to save memory. The above code
        # still reads all attributes so if we need to add more later we can.
        return {"mannings_n_at_sea_floor": data["mannings_n_at_sea_floor"]}
