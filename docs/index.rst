ADCIRC Subgrid Preprocessor Documentation
==========================================

The ADCIRC Subgrid Preprocessor is a Python package that generates subgrid input files for ADCIRC hydrodynamic models. Subgrid correction terms account for the effects of unresolved bathymetric, topographic, and frictional features on the flow field, enabling more accurate modeling results without requiring high-resolution computational meshes.

This documentation provides comprehensive guidance for ADCIRC users who need to generate subgrid files for their modeling applications.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   input_formats
   usage
   examples
   scientific_background
   visualization
   troubleshooting
   api_reference

Key Features
============

* **Automated Subgrid Generation**: Process ADCIRC mesh files, DEM data, and land cover information to generate subgrid correction files
* **Multi-Dataset Support**: Build subgrid tables incrementally from multiple datasets with automatic prioritization
* **Flexible Configuration**: YAML-based configuration system with extensive customization options
* **Built-in Visualization**: Integrated plotting capabilities for quality control and result analysis
* **Scientific Accuracy**: Based on established subgrid methodologies adapted from NCSU's original implementation

Quick Start
===========

1. **Install the package**:

   .. code-block:: bash

      conda create -n adcirc-subgrid -c conda-forge python=3 gdal geopandas pandas netcdf4 pyyaml numba scipy
      conda activate adcirc-subgrid
      pip install .

2. **Create a configuration file** (``input.yaml``):

   .. code-block:: yaml

      input:
        adcirc_mesh: fort.14
        manning_lookup: ccap
        dem: elevation_data.tif
        land_cover: landcover_data.tif

      output:
        filename: subgrid.nc

      options:
        n_subgrid_levels: 50
        n_phi_levels: 50
        subgrid_level_distribution: histogram

3. **Generate the subgrid file**:

   .. code-block:: bash

      adcirc-subgrid prep input.yaml

Support and Development
=======================

* **Source Code**: https://github.com/waterinstitute/adcirc-subgrid
* **Issue Tracker**: https://github.com/waterinstitute/adcirc-subgrid/issues
* **License**: Apache License 2.0

This work is an adaptation of the original ADCIRC subgrid code developed at NC State University.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
