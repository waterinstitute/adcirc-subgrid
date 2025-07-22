Installation
============

System Requirements
-------------------

The ADCIRC Subgrid Preprocessor requires:

* **Python**: Version 3.9 or higher
* **Operating System**: Linux, macOS, or Windows
* **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
* **Storage**: Sufficient space for input datasets and output files

Key Dependencies
----------------

The package relies on several scientific Python libraries:

* **GDAL**: Geospatial Data Abstraction Library for raster/vector operations
* **NumPy/SciPy**: Numerical computing libraries
* **Pandas**: Data manipulation and analysis
* **NetCDF4**: Network Common Data Format support
* **Numba**: Just-in-time compilation for performance
* **GeoPandas**: Geospatial data operations
* **Rasterio/RioxArray**: Raster data I/O and processing

Installation Methods
--------------------

Method 1: Conda Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended installation method uses conda to manage dependencies, particularly for GDAL which can be challenging to install via pip:

**Step 1: Create conda environment**

.. code-block:: bash

   conda create -n adcirc-subgrid -c conda-forge python=3 gdal geopandas pandas netcdf4 pyyaml numba scipy schema numpy shapely xarray pyproj matplotlib rasterio rioxarray tqdm

**Step 2: Activate environment**

.. code-block:: bash

   conda activate adcirc-subgrid

**Step 3: Install the package**

.. code-block:: bash

   cd /path/to/adcirc-subgrid
   pip install .

Method 2: Using Conda-Lock Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reproducible installations, pre-generated conda-lock files are provided:

**Linux (x86_64):**

.. code-block:: bash

   conda create -n adcirc-subgrid
   conda activate adcirc-subgrid
   conda install --file requirements/adcirc-subgrid-conda-linux-64.yaml

**macOS (Intel):**

.. code-block:: bash

   conda create -n adcirc-subgrid
   conda activate adcirc-subgrid
   conda install --file requirements/adcirc-subgrid-conda-osx-64.yaml

**macOS (Apple Silicon):**

.. code-block:: bash

   conda create -n adcirc-subgrid
   conda activate adcirc-subgrid
   conda install --file requirements/adcirc-subgrid-conda-osx-arm64.yaml

**Windows:**

.. code-block:: bash

   conda create -n adcirc-subgrid
   conda activate adcirc-subgrid
   conda install --file requirements/adcirc-subgrid-conda-win-64.yaml

Method 3: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development or contributing to the project:

.. code-block:: bash

   git clone https://github.com/waterinstitute/adcirc-subgrid.git
   cd adcirc-subgrid
   conda env create -f requirements/environment.yaml
   conda activate adcirc-subgrid
   pip install -e .

Verification
------------

To verify your installation:

**Check package installation:**

.. code-block:: bash

   python -c "import AdcircSubgrid; print('Installation successful')"

**Check command-line interface:**

.. code-block:: bash

   adcirc-subgrid --help

You should see the help message with available subcommands (``prep``, ``plot-mesh``, ``plot-node``).

**Test with sample data:**

.. code-block:: bash

   cd examples/GBAY
   # Download data if using git-lfs
   git lfs fetch
   git lfs pull
   git lfs checkout
   gunzip fort.14.gz

   # Run test
   adcirc-subgrid prep input.yaml

Common Installation Issues
--------------------------

GDAL Installation Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: GDAL fails to install or import

**Solutions**:
- Use conda instead of pip for GDAL installation
- Ensure conda-forge channel is used: ``conda install -c conda-forge gdal``
- On Ubuntu/Debian: ``sudo apt-get install gdal-bin libgdal-dev``
- On CentOS/RHEL: ``sudo yum install gdal gdal-devel``

Memory Issues During Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Installation fails due to memory constraints

**Solutions**:
- Increase virtual memory/swap space
- Use ``pip install --no-cache-dir``
- Install dependencies individually rather than all at once

Permission Errors
~~~~~~~~~~~~~~~~~

**Problem**: Permission denied during installation

**Solutions**:
- Use virtual/conda environments instead of system Python
- Install with ``--user`` flag: ``pip install --user .``
- Ensure proper write permissions in installation directory

Dependency Conflicts
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Conflicting package versions

**Solutions**:
- Create fresh conda environment
- Use conda-lock files for reproducible environments
- Check for conflicting conda channels

Environment Variables
---------------------

Some installations may require environment variable configuration:

**GDAL Data Path** (if needed):

.. code-block:: bash

   export GDAL_DATA=/path/to/gdal/data

**Conda Environment Activation Script** (optional):

.. code-block:: bash

   # Add to ~/.bashrc or ~/.zshrc
   alias activate-subgrid='conda activate adcirc-subgrid'

Platform-Specific Notes
-----------------------

Linux
~~~~~

* Most distributions work well with conda installation
* Some HPC systems may require module loading: ``module load gdal python``
* Ensure adequate shared memory for large datasets: ``ulimit -s unlimited``

macOS
~~~~~

* Apple Silicon (M1/M2) users should use ``osx-arm64`` conda packages
* Homebrew GDAL may conflict with conda GDAL
* Use ``arch -arm64 conda`` on Apple Silicon if needed

Windows
~~~~~~~

* Windows Subsystem for Linux (WSL) is recommended for complex workflows
* PowerShell or Command Prompt both work for basic operations
* Long path support may be needed for deep directory structures

Performance Optimization
------------------------

For optimal performance:

**Memory Settings:**

.. code-block:: bash

   # Increase available memory for raster processing
   export GDAL_CACHEMAX=1024  # MB

**Numba Compilation:**

.. code-block:: bash

   # Disable threading if experiencing issues
   export NUMBA_DISABLE_TBB=1

**Parallel Processing:**

.. code-block:: bash

   # Set number of threads for numpy operations
   export OMP_NUM_THREADS=4

Next Steps
----------

After successful installation:

1. Review the :doc:`input_formats` documentation
2. Try the :doc:`examples` with provided sample data
3. Configure your own :doc:`usage` workflow
4. Set up :doc:`visualization` for quality control

Uninstallation
--------------

To remove the package:

.. code-block:: bash

   # Remove package only
   pip uninstall AdcircSubgrid

   # Remove entire environment
   conda deactivate
   conda env remove -n adcirc-subgrid
