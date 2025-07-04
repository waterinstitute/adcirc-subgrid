# ADCIRC Subgrid Preprocessor
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![Testing](https://github.com/waterinstitute/adcirc-subgrid/actions/workflows/pytest.yaml/badge.svg)](https://github.com/waterinstitute/adcirc-subgrid/actions/workflows/pytest.yaml)
![black](https://img.shields.io/badge/code%20style-black-000000.svg)

The ADCIRC Subgrid Preprocessor is a Python package which generates the subgrid input file for ADCIRC. ADCIRC's subgrid
input file allows for the specification of subgrid correction terms, which account for the effects of unresolved
bathymetric, topographic, and frictional features on the flow field. The result is a more accurate result without the
need for a high-resolution mesh.

This work is an adaptation of the original ADCIRC subgrid code [here](https://github.com/ccht-ncsu/subgridADCIRCUtility.git).

## Installation

> [!IMPORTANT]
> Once this package is stable, it will be distributed via normal Python package channels. Until then, it can only
> be installed manually using the instructions below.

The ADCIRC Subgrid Preprocessor can be installed via pip from the root directory of the repository:
```bash
pip install .
```
Note that the package requires GDAL to be installed on your system. It will likely be much easier to create a conda environment with the necessary dependencies:
```bash
conda create -n adcirc-subgrid -c conda-forge python=3 gdal geopandas pandas netcdf4 pyyaml numba scipy schema numpy shapely xarray pyproj matplotlib rasterio rioxarray tdqm
```

The conda solver, even with `libmamba` can sometimes take a while. For our CI environment, we use `conda-lock` to create a
static environment file that can be used to quickly create the environment. You can do this by running:
```bash
conda install -n adcirc-subgrid --file ${repository_root}/requirements/adcirc-subgrid-conda-{os}-{arch}.yaml
```

## Usage

The package has multiple command line options which can be used to generate the subgrid file and examine the output.

When creating the subgrid input file, the user must provide an ADCIRC mesh file, a GDAL-compatible DEM file, a land use
file, and the subgrid input file. Importantly, we currently recommend that all input files are in the WGS84 projection.

The format of the yaml input file is as follows:

```yaml
input:
  adcirc_mesh: fort.14
  manning_lookup: ccap # Either a lookup file or 'ccap' to use the default table
  dem: All_Regions_v20240403_6m_m_4326_4_5.tif
  land_cover: conus_2016_ccap_landcover_20200311_4326.tif

output:
  filename: subgrid.nc
  progress_bar_increment: 5

options:
  # Control for the number of subgrid levels for calculation and output
  n_subgrid_levels: 50 # Controls the number of levels the calculation is performed on
  n_phi_levels: 50 # Controls the number of phi levels between 0 and 1 where output is written

  # Control for the way the subgrid water levels are distributed
  subgrid_level_distribution: histogram # Either 'histogram' or 'linear'
```
The code can then be run using the following command:
```bash
adcirc-subgrid prep input.yaml
```
In order to create a subgrid input file for multiple areas you will have to run the prep code multiple times with a new
yaml file, existing subgrid lookup table, and new DEM and landcover datasets. The code will read in the existing table
and add data for areas covered by the new datasets. If the new datasets overlap with existing subgrid areas, the existing
areas will NOT be overwritten. Therefore, it is important to prioritize your datasets, with the best data being used first
to build the subgrid lookup.

Note: There is an additional line in the options section of the yaml that needs to be added to tell the code to read in an
existing lookup table. The yaml format should be as follows:

```yaml
input:
  adcirc_mesh: fort.14
  manning_lookup: ccap # Either a lookup file or 'ccap' to use the default table
  dem: All_Regions_v20240403_6m_m_4326_4_5.tif
  land_cover: conus_2016_ccap_landcover_20200311_4326.tif

output:
  filename: subgrid_updated.nc
  progress_bar_increment: 5

options:
  # Control for the number of subgrid levels for calculation and output
  n_subgrid_levels: 50 # Controls the number of levels the calculation is performed on
  n_phi_levels: 50 # Controls the number of phi levels between 0 and 1 where output is written
  existing_subgrid: ./subgrid.nc # allows code to add to existing subgrid table

  # Control for the way the subgrid water levels are distributed
  subgrid_level_distribution: histogram # Either 'histogram' or 'linear'
```
Run the ```prep``` again as you did before, but with new yaml file.
