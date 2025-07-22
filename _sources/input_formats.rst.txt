Input File Formats
==================

The ADCIRC Subgrid Preprocessor requires several input files to generate subgrid corrections. This section provides detailed specifications for each required input format and configuration options.

Configuration File Format
--------------------------

The primary configuration is specified using YAML format. The configuration file has three main sections: ``input``, ``output``, and ``options``.

Basic Configuration Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   input:
     adcirc_mesh: fort.14
     manning_lookup: ccap
     dem: elevation_data.tif
     land_cover: landcover_data.tif

   output:
     filename: subgrid.nc
     progress_bar_increment: 5

   options:
     n_subgrid_levels: 50
     n_phi_levels: 50
     subgrid_level_distribution: histogram

Input Section
~~~~~~~~~~~~~

**adcirc_mesh** (required)
  Path to the ADCIRC mesh file (typically ``fort.14``). This file defines the computational grid and initial bathymetry.

  * **Format**: ADCIRC ASCII mesh format
  * **Example**: ``./fort.14``

**manning_lookup** (required)
  Manning's roughness coefficient lookup specification. Can be either a path to a custom lookup file or ``ccap`` for the built-in CCAP lookup table.

  * **Options**:

    * ``ccap``: Use built-in C-CAP (Coastal Change Analysis Program) lookup table
    * ``/path/to/lookup.csv``: Path to custom CSV lookup file

  * **Custom Format**: CSV file with columns: ``Class``, ``Manning's n``
  * **Example**: ``ccap`` or ``./custom_manning.csv``

**dem** (required)
  Path to the Digital Elevation Model file. Must be in a GDAL-compatible raster format.

  * **Supported Formats**: GeoTIFF (``.tif``, ``.tiff``), NetCDF (``.nc``), HDF (``.hdf``)
  * **Coordinate System**: Should match other datasets (WGS84 recommended)
  * **Example**: ``./bathymetry_dem.tif``

**land_cover** (required)
  Path to the land cover classification raster. Used with Manning's lookup table to assign friction coefficients.

  * **Supported Formats**: Same as DEM
  * **Values**: Integer class codes corresponding to lookup table entries
  * **Example**: ``./landcover_ccap.tif``

Output Section
~~~~~~~~~~~~~~

**filename** (required)
  Output filename for the generated subgrid NetCDF file.

  * **Format**: NetCDF4 format (``.nc`` extension recommended)
  * **Example**: ``subgrid_corrections.nc``

**progress_bar_increment** (optional, default: 10)
  Frequency of progress updates during processing (in percentage points).

  * **Range**: 1-100
  * **Example**: ``5`` (update every 5% completion)

Options Section
~~~~~~~~~~~~~~~

**n_subgrid_levels** (optional, default: 11)
  Number of water level increments used for subgrid calculations. More levels provide higher accuracy but increase computation time.

  * **Recommended Range**: 25-100
  * **Performance Impact**: Linear increase in computation time
  * **Example**: ``50``

**n_phi_levels** (optional, default: 11)
  Number of phi (φ) levels between 0 and 1 written to the output file. This controls the resolution of the φ vs. water level relationship.

  * **Recommended Range**: 25-100
  * **Storage Impact**: Affects output file size
  * **Example**: ``50``

**subgrid_level_distribution** (optional, default: "linear")
  Method for distributing water levels used in subgrid calculations.

  * **Options**:

    * ``linear``: Evenly spaced levels
    * ``histogram``: Distribution based on elevation histogram (recommended)

  * **Example**: ``histogram``

**distribution_factor** (optional, default: 1.0)
  Scaling factor for water level distribution range.

  * **Range**: > 0.0
  * **Typical Values**: 0.5 - 2.0
  * **Example**: ``1.5``

**existing_subgrid** (optional, default: None)
  Path to existing subgrid file for incremental updates. Allows building composite subgrid tables from multiple datasets.

  * **Format**: NetCDF4 subgrid file from previous run
  * **Behavior**: New data supplements existing; no overwriting of existing areas
  * **Example**: ``./previous_subgrid.nc``

Advanced Configuration Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   input:
     adcirc_mesh: /data/meshes/gulf_coast.14
     manning_lookup: /data/lookups/custom_roughness.csv
     dem: /data/elevation/lidar_2023_6m.tif
     land_cover: /data/landcover/ccap_2021_30m.tif

   output:
     filename: gulf_coast_subgrid_v2.nc
     progress_bar_increment: 2

   options:
     n_subgrid_levels: 75
     n_phi_levels: 75
     subgrid_level_distribution: histogram
     distribution_factor: 1.25
     existing_subgrid: gulf_coast_subgrid_v1.nc

ADCIRC Mesh File (fort.14)
---------------------------

The ADCIRC mesh file defines the computational grid and provides initial bathymetry values.

Format Specification
~~~~~~~~~~~~~~~~~~~~~

The fort.14 file uses ADCIRC's standard ASCII format:

.. code-block:: text

   ADCIRC Mesh Title
   NE NN        ! Number of elements, number of nodes
   1  x1  y1  z1    ! Node ID, x-coordinate, y-coordinate, depth
   2  x2  y2  z2
   ...
   NN  xNN yNN zNN
   1  3  n1  n2  n3  ! Element ID, nodes per element, node connectivity
   2  3  n4  n5  n6
   ...

**Key Requirements:**

* **Coordinate System**: Should match DEM and land cover data
* **Bathymetry Values**: Positive depths below datum
* **Node Numbering**: Must be consecutive starting from 1
* **Element Connectivity**: Counter-clockwise node ordering

Digital Elevation Model (DEM)
------------------------------

High-resolution elevation data is critical for effective subgrid corrections.

Data Requirements
~~~~~~~~~~~~~~~~~

**Resolution**:
  DEM resolution should be significantly finer than mesh resolution. Typical ratios:

  * **Coastal applications**: DEM 5-10x finer than average mesh spacing
  * **Urban flooding**: DEM 10-20x finer than mesh spacing
  * **Large domains**: DEM 3-5x finer than mesh spacing

**Coordinate System**:
  * WGS84 (EPSG:4326) recommended for consistency
  * Must match ADCIRC mesh coordinate system
  * Ensure proper datum alignment between datasets

**Coverage**:
  * Must completely cover the computational domain
  * Extended coverage beyond mesh boundaries recommended
  * No-data values should be properly defined

**Elevation Convention**:
  * Positive values above datum
  * Negative values below datum
  * Consistent with ADCIRC bathymetry convention

Format Examples
~~~~~~~~~~~~~~~

**GeoTIFF (recommended)**:

.. code-block:: bash

   # Check DEM properties
   gdalinfo elevation_data.tif

   # Verify coordinate system
   gdalsrsinfo elevation_data.tif

**NetCDF**:

.. code-block:: bash

   # Check NetCDF structure
   ncdump -h elevation_data.nc

Land Cover Data
---------------

Land cover classification provides the basis for Manning's roughness assignment.

Classification Systems
~~~~~~~~~~~~~~~~~~~~~~

**CCAP (Coastal Change Analysis Program)**:
  Standard classification with built-in lookup table based on research validation:

  * **Class 2**: High Intensity Developed (n=0.120)
  * **Class 3**: Medium Intensity Developed (n=0.100)
  * **Class 4**: Low Intensity Developed (n=0.070)
  * **Class 5**: Developed Open Space (n=0.035)
  * **Class 6**: Cultivated Land (n=0.100)
  * **Class 7**: Pasture/Hay (n=0.055)
  * **Class 8**: Grassland (n=0.035)
  * **Class 9**: Deciduous Forest (n=0.160)
  * **Class 10**: Evergreen Forest (n=0.180)
  * **Class 11**: Mixed Forest (n=0.170)
  * **Class 12**: Scrub/Shrub (n=0.080)
  * **Class 13**: Palustrine Forested Wetland (n=0.150)
  * **Class 14**: Palustrine Scrub/Shrub Wetland (n=0.075)
  * **Class 15**: Palustrine Emergent Wetland (n=0.070)
  * **Class 16**: Estuarine Forested Wetland (n=0.150)
  * **Class 17**: Estuarine Scrub/Shrub Wetland (n=0.075)
  * **Class 18**: Estuarine Emergent Wetland (n=0.070)
  * **Class 19**: Unconsolidated Shore (n=0.030)
  * **Class 20**: Bare Land (n=0.035)
  * **Class 21**: Open Water (n=0.025)
  * **Class 22**: Palustrine Aquatic Bed (n=0.025)
  * **Class 23**: Estuarine Aquatic Bed (n=0.025)

**Research-Validated Manning's Coefficients:**

Based on recent research (Woodruff et al., 2025), the following values have been validated for coastal applications:

  * **Open Water**: n=0.025
  * **Palustrine Emergent Wetland**: n=0.07
  * **Estuarine Emergent Wetland**: n=0.05
  * **Cultivated Land**: n=0.1

Custom Manning's Lookup
~~~~~~~~~~~~~~~~~~~~~~~~

Create a CSV file with custom roughness values:

.. code-block:: text

   Class,Manning's n
   1,0.025
   2,0.040
   3,0.080
   4,0.120
   5,0.030

**Requirements**:
  * CSV format with header row
  * ``Class`` column: Integer class codes
  * ``Manning's n`` column: Roughness coefficients
  * All classes present in land cover data must be included

Data Quality Considerations
---------------------------

Coordinate System Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure all datasets use the same coordinate reference system:

.. code-block:: bash

   # Check coordinate systems
   gdalsrsinfo -e adcirc_mesh.14  # For mesh files, check documentation
   gdalsrsinfo elevation.tif
   gdalsrsinfo landcover.tif

   # Reproject if needed
   gdalwarp -t_srs EPSG:4326 input.tif output_reprojected.tif

Resolution Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~

Verify appropriate resolution relationships:

.. code-block:: python

   import rasterio

   # Check DEM resolution
   with rasterio.open('dem.tif') as src:
       print(f"DEM resolution: {src.res}")

   # Compare with typical mesh spacing
   # Mesh spacing should be 3-20x larger than DEM pixel size

Data Coverage Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure complete domain coverage:

.. code-block:: python

   import rasterio
   from rasterio.mask import mask
   import geopandas as gpd

   # Check if DEM covers mesh domain
   with rasterio.open('dem.tif') as dem:
       dem_bounds = dem.bounds

   # Compare with mesh bounds
   print(f"DEM bounds: {dem_bounds}")
   # Verify mesh extent falls within DEM coverage

File Naming Conventions
-----------------------

Recommended naming patterns for organization:

.. code-block:: text

   # Configuration files
   config_[region]_[resolution]_[version].yaml

   # Output files
   subgrid_[region]_[resolution]_[date].nc

   # Input data
   dem_[region]_[source]_[resolution]_[date].tif
   landcover_[region]_[source]_[year].tif

   # Examples
   config_galveston_6m_v1.yaml
   subgrid_galveston_6m_20240315.nc
   dem_galveston_lidar_6m_20240101.tif
   landcover_galveston_ccap_2021.tif

This comprehensive input format guide ensures proper data preparation for successful subgrid generation.
