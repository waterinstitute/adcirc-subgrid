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
import numpy as np


class LookupTable:
    """
    LookupTable class to read and store the Manning lookup table. The lookup
    table is read from a CSV file with the format:

    Class, Manning's n
    Class, Manning's n
    Class, Manning's n

    The data is stored in a numpy array which is sized to the maximum class value
    so we can use direct indexing to look up the Manning's n value for a given class
    rather than a hash table.
    """

    @staticmethod
    def ccap_lookup() -> np.ndarray:
        """
        Generates the default ccap lookup using:

          2    0.120     :High Intensity Developed
          3    0.100     :Medium Intensity Developed
          4    0.070     :Low Intesnity Developed
          5    0.035     :Developed Open Space
          6    0.100     :Cultivated Land
          7    0.055     :Pasture/Hay
          8    0.035     :Grassland
          9    0.160     :Deciduous Forest
         10    0.180     :Evergreen Forest
         11    0.170     :Mixed Forest
         12    0.080     :Scrub/Shrub
         13    0.150     :Palustrine Forested Wetland
         14    0.075     :Palustrine Scrub/Shrub Wetland
         15    0.070     :Palustrine Emergent Wetland
         16    0.150     :Estuarine Forested Wetland
         17    0.070     :Estuarine Scrub/Schrub Wetland
         18    0.050     :Estuarine Emergent Wetland
         19    0.030     :Unconsolidated Shore
         20    0.030     :Bare Land
         21    0.025     :Open Water
         22    0.035     :Palustrine Aquatic Bed
         23    0.030     :Estuarine Aquatic Bed

        Returns:
            Default CCAP array
        """
        default_ccap_values = np.array(
            [
                np.nan,
                np.nan,
                0.120,
                0.100,
                0.070,
                0.035,
                0.100,
                0.055,
                0.035,
                0.160,
                0.180,
                0.170,
                0.080,
                0.150,
                0.075,
                0.070,
                0.150,
                0.070,
                0.050,
                0.030,
                0.030,
                0.025,
                0.035,
                0.030,
            ]
        )

        out_array = np.full(256, np.nan)
        out_array[0:24] = default_ccap_values

        return out_array

    def __init__(self, lookup_table: str) -> None:
        """
        Initialize the Manning lookup table

        Args:
            lookup_table: File path to the Manning lookup table
        """
        import os

        if lookup_table == "ccap":
            self.__lookup_table = LookupTable.ccap_lookup()
        else:
            if not os.path.exists(lookup_table):
                msg = f"Lookup table file {lookup_table} does not exist"
                raise ValueError(msg)
            self.__lookup_table = LookupTable.read_lookup_table(lookup_table)

    @staticmethod
    def read_lookup_table(lookup_filename: str) -> np.ndarray:
        """
        Read a Manning lookup file with the format:

        Class, Manning's n
        Class, Manning's n
        Class, Manning's n

        Args:
            lookup_filename: File path to the Manning lookup table

        Returns:
            Lookup table as a dictionary
        """
        lookup_table = np.loadtxt(lookup_filename, delimiter=",", skiprows=0)
        max_value = int(np.max(lookup_table[:, 0]))

        if len(lookup_table) != len(np.unique(lookup_table[:, 0])):
            msg = "Duplicate classes found in Manning lookup table"
            raise ValueError(msg)

        arr = np.full(max_value + 1, np.nan)
        for row in lookup_table:
            arr[int(row[0])] = row[1]

        # Create an array of size 256 and fill it with NaN
        out_array = np.full(256, np.nan)

        # Fill the rest with the values from the lookup table
        out_array[0 : arr.size] = arr

        return out_array

    def lookup_table(self) -> np.ndarray:
        """
        Get the lookup table

        Returns:
            The lookup table as a numpy array
        """
        return self.__lookup_table
