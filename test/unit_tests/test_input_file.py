import tempfile

import yaml
from AdcircSubgrid.input_file import InputFile


def test_input_file() -> None:
    """
    Test the InputFile class to ensure that it reads the input file correctly
    """
    input_file_data = {
        "adcirc_mesh": "fort.14",
        "output_filename": "subgrid.nc",
        "manning_lookup": "ccap",
        "n_subgrid_levels": 20,
        "dem": "All_Regions_v20240403_6m_m_4326_5_4.tif",
        "land_cover": "conus_2016_ccap_landcover_20200311.tif",
    }

    with tempfile.NamedTemporaryFile("w", delete=True) as file:
        yaml.safe_dump(input_file_data, file)
        file.flush()

        input_file = InputFile(file.name)

        assert input_file.data()["adcirc_mesh"] == "fort.14"
        assert input_file.data()["output_filename"] == "subgrid.nc"
        assert input_file.data()["manning_lookup"] == "ccap"
        assert input_file.data()["n_subgrid_levels"] == 20
        assert input_file.data()["dem"] == "All_Regions_v20240403_6m_m_4326_5_4.tif"
        assert (
            input_file.data()["land_cover"] == "conus_2016_ccap_landcover_20200311.tif"
        )
