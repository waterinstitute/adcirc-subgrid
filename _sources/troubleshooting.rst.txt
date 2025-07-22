Troubleshooting Guide
=====================

This guide provides solutions to common issues encountered when using the ADCIRC Subgrid Preprocessor.

Installation Issues
-------------------

GDAL Installation Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: GDAL fails to install or import with errors like:

.. code-block:: text

   ImportError: No module named 'osgeo'
   ERROR: Could not install packages due to an OSError

**Solutions**:

1. **Use Conda for GDAL Installation**:

   .. code-block:: bash

      # Remove existing installation
      pip uninstall gdal

      # Install via conda
      conda install -c conda-forge gdal

2. **System-Level GDAL Installation** (Linux):

   .. code-block:: bash

      # Ubuntu/Debian
      sudo apt-get update
      sudo apt-get install gdal-bin libgdal-dev
      export GDAL_CONFIG=/usr/bin/gdal-config
      pip install gdal==$(gdal-config --version)

      # CentOS/RHEL
      sudo yum install gdal gdal-devel
      pip install gdal

3. **macOS Specific**:

   .. code-block:: bash

      # Using Homebrew (may conflict with conda)
      brew install gdal

      # Set environment variables
      export GDAL_CONFIG=/usr/local/bin/gdal-config
      pip install gdal

**Verification**:

.. code-block:: bash

   python -c "from osgeo import gdal; print('GDAL version:', gdal.__version__)"

Memory Errors During Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Installation fails with memory-related errors:

.. code-block:: text

   MemoryError: Unable to allocate array
   ERROR: Failed building wheel for package

**Solutions**:

1. **Increase Virtual Memory**:

   .. code-block:: bash

      # Linux
      sudo fallocate -l 4G /swapfile
      sudo chmod 600 /swapfile
      sudo mkswap /swapfile
      sudo swapon /swapfile

2. **Use No-Cache Installation**:

   .. code-block:: bash

      pip install --no-cache-dir .

3. **Install Dependencies Individually**:

   .. code-block:: bash

      pip install numpy
      pip install scipy
      pip install pandas
      # Continue with other packages...

Dependency Conflicts
~~~~~~~~~~~~~~~~~~~~

**Problem**: Conflicting package versions:

.. code-block:: text

   ERROR: pip's dependency resolver does not currently take into account
   all the packages that are installed

**Solutions**:

1. **Create Fresh Environment**:

   .. code-block:: bash

      conda deactivate
      conda env remove -n adcirc-subgrid
      conda create -n adcirc-subgrid python=3.9

2. **Use Conda-Lock Files**:

   .. code-block:: bash

      conda install --file requirements/adcirc-subgrid-conda-linux-64.yaml

3. **Pin Specific Versions**:

   .. code-block:: bash

      pip install "numpy>=1.20,<1.25" "pandas>=1.3,<2.0"

Runtime Errors
--------------

Configuration File Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: YAML parsing errors:

.. code-block:: text

   yaml.scanner.ScannerError: while scanning a simple key
   could not find expected ':'

**Solutions**:

1. **Check YAML Syntax**:

   .. code-block:: bash

      # Use online YAML validator or
      python -c "import yaml; yaml.safe_load(open('config.yaml'))"

2. **Common YAML Issues**:

   .. code-block:: yaml

      # Incorrect (missing space after colon)
      input:
        adcirc_mesh:./fort.14

      # Correct
      input:
        adcirc_mesh: ./fort.14

      # Incorrect (inconsistent indentation)
      options:
       n_subgrid_levels: 50
         n_phi_levels: 50

      # Correct
      options:
        n_subgrid_levels: 50
        n_phi_levels: 50

File Path Issues
~~~~~~~~~~~~~~~~~

**Problem**: File not found errors:

.. code-block:: text

   FileNotFoundError: No such file or directory: './fort.14'

**Solutions**:

1. **Use Absolute Paths**:

   .. code-block:: yaml

      input:
        adcirc_mesh: /full/path/to/fort.14
        dem: /full/path/to/elevation.tif

2. **Verify File Existence**:

   .. code-block:: bash

      ls -la fort.14
      file fort.14  # Check file type

3. **Check Current Directory**:

   .. code-block:: bash

      pwd  # Print working directory
      ls   # List files in current directory

Memory and Performance Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Out of memory errors during processing:

.. code-block:: text

   MemoryError: Unable to allocate 2.3 GiB for an array with shape...

**Solutions**:

1. **Reduce Window Memory**:

   .. code-block:: bash

      adcirc-subgrid prep config.yaml --window-memory 32

2. **Process Smaller Regions**:

   .. code-block:: bash

      # Use GDAL to clip input data
      gdal_translate -projwin xmin ymin xmax ymax input.tif subset.tif

3. **System Memory Settings**:

   .. code-block:: bash

      # Linux: Check available memory
      free -h

      # Increase swap space if needed
      sudo swapon --show

4. **Environment Variables**:

   .. code-block:: bash

      export GDAL_CACHEMAX=256  # MB
      export OMP_NUM_THREADS=2  # Reduce parallelism

Coordinate System Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Coordinate system mismatches:

.. code-block:: text

   ValueError: Incompatible coordinate reference systems

**Solutions**:

1. **Check Coordinate Systems**:

   .. code-block:: bash

      gdalsrsinfo elevation.tif
      gdalsrsinfo landcover.tif

2. **Reproject Data**:

   .. code-block:: bash

      # Reproject to WGS84
      gdalwarp -t_srs EPSG:4326 input.tif output_wgs84.tif

3. **Verify ADCIRC Mesh CRS**:

   Check ADCIRC documentation or mesh metadata for coordinate system information.

Data Quality Issues
--------------------

Invalid φ Values
~~~~~~~~~~~~~~~~~

**Problem**: φ values outside the valid range [0,1]:

.. code-block:: text

   Warning: Found phi values > 1.0 or < 0.0

**Diagnosis**:

.. code-block:: python

   import xarray as xr
   import numpy as np

   ds = xr.open_dataset('subgrid.nc')
   phi = ds['phi']

   # Check for invalid values
   invalid_high = (phi > 1.0).any()
   invalid_low = (phi < 0.0).any()

   print(f"Values > 1.0: {invalid_high.sum().item()}")
   print(f"Values < 0.0: {invalid_low.sum().item()}")

**Solutions**:

1. **Check DEM Quality**:

   .. code-block:: bash

      gdalinfo -stats elevation.tif
      # Look for unusual min/max values

2. **Verify Datum Consistency**:

   Ensure DEM and mesh use consistent vertical datums.

3. **Adjust Processing Parameters**:

   .. code-block:: yaml

      options:
        distribution_factor: 0.8  # Reduce range
        n_subgrid_levels: 30      # Fewer levels

Non-Monotonic φ Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: φ decreases with increasing water level:

.. code-block:: text

   Warning: Non-monotonic phi relationship detected

**Diagnosis**:

.. code-block:: python

   # Check monotonicity
   phi_diff = ds['phi'].diff(dim='phi_level')
   non_monotonic = (phi_diff < 0).any(dim='phi_level')
   problem_nodes = non_monotonic.sum().item()

**Solutions**:

1. **Increase Processing Levels**:

   .. code-block:: yaml

      options:
        n_subgrid_levels: 75  # More levels for smoother relationships

2. **Use Histogram Distribution**:

   .. code-block:: yaml

      options:
        subgrid_level_distribution: histogram  # More adaptive

3. **Clean Input DEM**:

   .. code-block:: bash

      # Remove spikes and fill holes
      gdal_fillnodata.py elevation.tif elevation_cleaned.tif

Visualization Problems
-----------------------

Matplotlib Display Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Plots don't display or show blank windows:

.. code-block:: text

   UserWarning: Matplotlib is currently using agg, which is a non-GUI backend

**Solutions**:

1. **Set Backend**:

   .. code-block:: python

      import matplotlib
      matplotlib.use('TkAgg')  # or 'Qt5Agg'
      import matplotlib.pyplot as plt

2. **Install GUI Backend**:

   .. code-block:: bash

      # For Tkinter backend
      sudo apt-get install python3-tk

      # For Qt backend
      pip install PyQt5

3. **Use X11 Forwarding** (SSH):

   .. code-block:: bash

      ssh -X username@hostname

Large Dataset Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Visualization crashes with large datasets:

.. code-block:: text

   MemoryError during plotting

**Solutions**:

1. **Plot Subsets**:

   .. code-block:: bash

      adcirc-subgrid plot-mesh subgrid.nc --variable percent_wet \
          --bbox -95.0 29.0 -94.0 30.0

2. **Reduce Point Density**:

   .. code-block:: python

      # In Python visualization scripts
      stride = 10  # Plot every 10th point
      plt.scatter(x[::stride], y[::stride], c=phi[::stride])

3. **Use Vector Graphics**:

   .. code-block:: bash

      adcirc-subgrid plot-mesh subgrid.nc --output-filename plot.svg

Performance Optimization
------------------------

Slow Processing
~~~~~~~~~~~~~~~~

**Problem**: Preprocessing takes excessively long:

**Diagnosis**:

1. **Profile Processing Steps**:

   .. code-block:: bash

      adcirc-subgrid --verbose prep config.yaml 2>&1 | tee processing.log

2. **Check Resource Usage**:

   .. code-block:: bash

      # Monitor during processing
      htop  # or top
      iostat -x 1  # I/O monitoring

**Solutions**:

1. **Optimize Configuration**:

   .. code-block:: yaml

      options:
        n_subgrid_levels: 30    # Reduce from 50+
        subgrid_level_distribution: histogram  # More efficient

2. **Increase Memory Allocation**:

   .. code-block:: bash

      adcirc-subgrid prep config.yaml --window-memory 128

3. **Use Faster Storage**:

   Move input/output files to SSD storage for better I/O performance.

4. **Parallel Processing**:

   .. code-block:: bash

      export OMP_NUM_THREADS=4  # Use available cores

Large File Handling
~~~~~~~~~~~~~~~~~~~

**Problem**: Processing very large DEM files:

**Solutions**:

1. **Use Pyramids/Overviews**:

   .. code-block:: bash

      gdaladdo -r average elevation.tif 2 4 8 16

2. **Tile Large Files**:

   .. code-block:: bash

      gdal_retile.py -ps 2048 2048 -targetDir tiles/ elevation.tif

3. **Use Cloud-Optimized GeoTIFF**:

   .. code-block:: bash

      gdal_translate -of COG -co COMPRESS=LZW input.tif output_cog.tif

Integration Issues
-------------------

ADCIRC Compatibility
~~~~~~~~~~~~~~~~~~~~

**Problem**: ADCIRC doesn't read subgrid file:

.. code-block:: text

   ERROR in READ_SFLUX_DATA: NetCDF file not found or corrupted

**Solutions**:

1. **Verify File Format**:

   .. code-block:: bash

      ncdump -h subgrid.nc | head -20

2. **Check File Permissions**:

   .. code-block:: bash

      ls -la subgrid.nc
      chmod 644 subgrid.nc

3. **Verify Node Count Match**:

   .. code-block:: bash

      # Compare mesh and subgrid node counts
      mesh_nodes=$(head -2 fort.14 | tail -1 | awk '{print $2}')
      subgrid_nodes=$(ncdump -h subgrid.nc | grep "node =" | cut -d= -f2 | cut -d' ' -f2)
      echo "Mesh: $mesh_nodes, Subgrid: $subgrid_nodes"

ADCIRC fort.15 Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ADCIRC run fails with subgrid enabled:

**Solutions**:

1. **Check NWP Setting**:

   .. code-block:: fortran

      NWP = 1  ! Must be 1 to enable subgrid

2. **Verify File Path**:

   .. code-block:: fortran

      ! Use relative or absolute path
      NCFILE = './subgrid.nc'
      ! or
      NCFILE = '/full/path/to/subgrid.nc'

3. **Adjust Timestep**:

   .. code-block:: fortran

      DTDP = 0.5  ! May need smaller timestep for stability

Getting Help
-------------

Log Information
~~~~~~~~~~~~~~~~

When reporting issues, include:

1. **Verbose Log Output**:

   .. code-block:: bash

      adcirc-subgrid --verbose prep config.yaml > processing.log 2>&1

2. **System Information**:

   .. code-block:: bash

      python --version
      conda list | grep -E "(gdal|numpy|scipy|pandas|netcdf4)"
      uname -a  # Linux/macOS
      # or
      systeminfo  # Windows

3. **Configuration File**:

   Include your YAML configuration (remove sensitive paths if needed).

4. **Error Messages**:

   Copy complete error messages, including stack traces.

Community Resources
~~~~~~~~~~~~~~~~~~~~

* **GitHub Issues**: https://github.com/waterinstitute/adcirc-subgrid/issues
* **ADCIRC Forums**: For ADCIRC-specific integration questions
* **Documentation**: This documentation for reference

When submitting issues, provide:
- Minimal reproducible example
- Complete error messages
- System and environment information
- Steps already attempted

This troubleshooting guide should resolve most common issues encountered during subgrid preprocessing workflows.
