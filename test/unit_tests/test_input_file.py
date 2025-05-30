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

import yaml

from AdcircSubgrid.input_file import InputFile


def test_input_file() -> None:
    """
    Test the InputFile class to ensure that it reads the input file correctly
    """
    input_file_data = {
        "input": {
            "adcirc_mesh": "fort.14",
            "manning_lookup": "ccap",
            "dem": "All_Regions_v20240403_6m_m_4326_5_4.tif",
            "land_cover": "conus_2016_ccap_landcover_20200311.tif",
        },
        "output": {
            "filename": "subgrid.nc",
            "progress_bar_increment": 10,
        },
        "options": {
            "n_subgrid_levels": 20,
            "n_phi_levels": 11,
            "subgrid_level_distribution": "linear",
            "distribution_factor": 1.0,
        },
    }

    # Touch the dem and land_cover files so that they exist. Set them
    # so that they are deleted when the context manager exits.
    dem_filename = input_file_data["input"]["dem"]
    land_cover_filename = input_file_data["input"]["land_cover"]

    with (
        open(dem_filename, "w"),
        open(land_cover_filename, "w"),
        open("input_file.yaml", "w") as file,
    ):
        try:
            yaml.safe_dump(input_file_data, file)
            file.flush()

            input_file = InputFile(file.name)

            assert input_file.data()["input"]["adcirc_mesh"] == "fort.14"
            assert input_file.data()["output"]["filename"] == "subgrid.nc"
            assert input_file.data()["input"]["manning_lookup"] == "ccap"
            assert input_file.data()["options"]["n_subgrid_levels"] == 20
            assert (
                input_file.data()["input"]["dem"]
                == "All_Regions_v20240403_6m_m_4326_5_4.tif"
            )
            assert (
                input_file.data()["input"]["land_cover"]
                == "conus_2016_ccap_landcover_20200311.tif"
            )
        finally:
            os.remove(dem_filename)
            os.remove(land_cover_filename)
            os.remove("input_file.yaml")
