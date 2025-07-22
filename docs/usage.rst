Usage Guide
===========

This guide covers the practical aspects of using the ADCIRC Subgrid Preprocessor, from basic workflows to advanced techniques.

Command Line Interface
----------------------

The subgrid preprocessor provides a command-line interface with several subcommands:

Basic Command Structure
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   adcirc-subgrid [global-options] <subcommand> [subcommand-options]

Global Options
~~~~~~~~~~~~~~

**--verbose**
  Enable verbose logging output for debugging and detailed progress information.

  .. code-block:: bash

     adcirc-subgrid --verbose prep config.yaml

Subcommands
~~~~~~~~~~~

**prep** - Preprocessing Command
  Generate subgrid corrections from input data.

  .. code-block:: bash

     adcirc-subgrid prep [options] config.yaml

  * ``config.yaml``: Path to YAML configuration file
  * ``--window-memory``: Raster processing memory limit in MB (default: 64)

**plot-mesh** - Mesh Visualization
  Create visualizations of subgrid data on the computational mesh.

  .. code-block:: bash

     adcirc-subgrid plot-mesh [options] subgrid.nc

  * ``--water-level``: Water level for visualization (default: 0.0)
  * ``--variable``: Variable to plot (percent_wet, wet_depth, total_depth, cf, c_mf, c_adv)
  * ``--show``: Display plot interactively
  * ``--output-filename``: Save plot to file
  * ``--bbox``: Bounding box (minx, miny, maxx, maxy)
  * ``--range``: Value range for colorbar (min, max)
  * ``--colorbar``: Matplotlib colormap name

**plot-node** - Node-Specific Visualization
  Create detailed plots of subgrid relationships for individual nodes.

  .. code-block:: bash

     adcirc-subgrid plot-node [options] --filename subgrid.nc --node 1234

  * ``--filename``: Subgrid NetCDF file
  * ``--node``: Node number to plot
  * ``--basis``: Plot basis (wse or phi, default: phi)
  * ``--show``: Display plot interactively
  * ``--save``: Save plot to file
  * ``--index-base``: Node numbering base (0 or 1, default: 0)

Basic Workflows
---------------

Single Dataset Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest workflow processes a single set of input data:

**Step 1: Prepare configuration file**

.. code-block:: yaml

   input:
     adcirc_mesh: ./fort.14
     manning_lookup: ccap
     dem: ./elevation_6m.tif
     land_cover: ./ccap_2021.tif

   output:
     filename: ./subgrid_single.nc
     progress_bar_increment: 5

   options:
     n_subgrid_levels: 50
     n_phi_levels: 50
     subgrid_level_distribution: histogram

**Step 2: Run preprocessing**

.. code-block:: bash

   adcirc-subgrid prep single_dataset.yaml

**Step 3: Verify results**

.. code-block:: bash

   # Quick visualization
   adcirc-subgrid plot-mesh subgrid_single.nc --show --variable percent_wet

Multi-Dataset Processing
~~~~~~~~~~~~~~~~~~~~~~~~

For comprehensive coverage using multiple datasets with different priorities:

**Step 1: Process highest priority dataset**

.. code-block:: yaml

   # config_priority1.yaml
   input:
     adcirc_mesh: ./fort.14
     manning_lookup: ccap
     dem: ./lidar_high_res.tif        # Best quality data
     land_cover: ./ccap_2021.tif

   output:
     filename: ./subgrid_step1.nc

   options:
     n_subgrid_levels: 75
     n_phi_levels: 75
     subgrid_level_distribution: histogram

.. code-block:: bash

   adcirc-subgrid prep config_priority1.yaml

**Step 2: Add second dataset**

.. code-block:: yaml

   # config_priority2.yaml
   input:
     adcirc_mesh: ./fort.14
     manning_lookup: ccap
     dem: ./satellite_moderate_res.tif  # Lower quality, broader coverage
     land_cover: ./ccap_2021.tif

   output:
     filename: ./subgrid_step2.nc

   options:
     n_subgrid_levels: 75
     n_phi_levels: 75
     subgrid_level_distribution: histogram
     existing_subgrid: ./subgrid_step1.nc  # Build on previous results

.. code-block:: bash

   adcirc-subgrid prep config_priority2.yaml

**Step 3: Continue for additional datasets**

Repeat the process with progressively lower priority datasets, always referencing the most recent output as ``existing_subgrid``.

Advanced Configuration
----------------------

Memory Management
~~~~~~~~~~~~~~~~~

For large datasets, control memory usage:

**Configuration Options:**

.. code-block:: bash

   # Limit raster processing memory
   adcirc-subgrid prep config.yaml --window-memory 128

**Environment Variables:**

.. code-block:: bash

   # GDAL cache settings
   export GDAL_CACHEMAX=512    # MB
   export GDAL_MAX_DATASET_POOL_SIZE=100

   # Disable multithreading if memory constrained
   export OMP_NUM_THREADS=1

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**Configuration for Performance:**

.. code-block:: yaml

   options:
     # Balance accuracy vs. speed
     n_subgrid_levels: 30        # Fewer levels = faster
     n_phi_levels: 30

     # Histogram distribution is more efficient
     subgrid_level_distribution: histogram

**System-Level Optimizations:**

.. code-block:: bash

   # Increase available memory
   ulimit -v unlimited

   # Use local storage for temporary files
   export TMPDIR=/local/tmp

   # Optimize thread usage
   export OMP_NUM_THREADS=4

Quality Control Procedures
--------------------------

Visual Inspection
~~~~~~~~~~~~~~~~~

**Mesh-Level Visualization:**

.. code-block:: bash

   # Overview of φ values at mean sea level
   adcirc-subgrid plot-mesh subgrid.nc --variable percent_wet --water-level 0.0 --show

   # Examine specific regions
   adcirc-subgrid plot-mesh subgrid.nc --variable percent_wet --bbox -95.0 29.0 -94.0 30.0 --show

   # Check different variables
   adcirc-subgrid plot-mesh subgrid.nc --variable wet_depth --water-level 1.0 --show

**Node-Level Analysis:**

.. code-block:: bash

   # Examine φ vs. water level relationships
   adcirc-subgrid plot-node --filename subgrid.nc --node 1500 --basis phi --show

   # Check specific problematic nodes
   adcirc-subgrid plot-node --filename subgrid.nc --node 2847 --basis wse --save node_2847.png

Data Validation
~~~~~~~~~~~~~~~

**NetCDF File Inspection:**

.. code-block:: bash

   # Check file structure and metadata
   ncdump -h subgrid.nc

   # Examine variable ranges
   ncks -H -v phi subgrid.nc | grep "min\|max"

**Python-Based Validation:**

.. code-block:: python

   import xarray as xr
   import numpy as np

   # Load subgrid data
   ds = xr.open_dataset('subgrid.nc')

   # Check for valid phi ranges (0 <= φ <= 1)
   phi = ds['phi']
   invalid = (phi < 0) | (phi > 1)
   print(f"Invalid phi values: {invalid.sum().item()}")

   # Check for monotonic relationships
   for node in range(100):  # Check first 100 nodes
       node_phi = phi.isel(node=node)
       if not np.all(np.diff(node_phi) >= 0):
           print(f"Non-monotonic phi at node {node}")

Troubleshooting Common Issues
-----------------------------

Processing Errors
~~~~~~~~~~~~~~~~~

**Memory Errors:**

.. code-block:: text

   Error: MemoryError during raster processing

   Solutions:
   - Reduce --window-memory parameter
   - Process smaller geographic regions
   - Use histogram distribution (more memory efficient)
   - Increase system RAM or virtual memory

**Coordinate System Mismatches:**

.. code-block:: text

   Error: Inconsistent coordinate systems

   Solutions:
   - Verify all input files use same CRS
   - Reproject data using gdalwarp
   - Check ADCIRC mesh coordinate system documentation

**Missing Data Coverage:**

.. code-block:: text

   Warning: Incomplete DEM coverage

   Solutions:
   - Extend DEM boundaries beyond mesh extent
   - Fill no-data areas with appropriate values
   - Use multiple DEM sources to ensure complete coverage

Data Quality Issues
~~~~~~~~~~~~~~~~~~~

**Unrealistic φ Values:**

.. code-block:: text

   Issue: φ values outside [0,1] range or non-monotonic behavior

   Diagnosis:
   - Check DEM quality and artifacts
   - Verify proper datum alignment
   - Examine land cover classification accuracy

   Solutions:
   - Clean input DEM data
   - Adjust distribution_factor parameter
   - Use higher quality input datasets

**Inconsistent Results:**

.. code-block:: text

   Issue: Unexpected subgrid corrections

   Diagnosis:
   - Compare with known high-resolution results
   - Check Manning's roughness values
   - Verify mesh bathymetry consistency

   Solutions:
   - Validate input data independently
   - Adjust configuration parameters
   - Use subset processing for testing

Integration with ADCIRC
-----------------------

File Preparation
~~~~~~~~~~~~~~~~

**Subgrid File Usage:**

The generated subgrid NetCDF file is used directly by ADCIRC:

.. code-block:: fortran

   ! In ADCIRC fort.15 (control file)
   NWP = 1                    ! Enable subgrid corrections
   NCFILE = 'subgrid.nc'     ! Path to subgrid file

**Parameter Settings:**

Recommended ADCIRC configuration for subgrid runs:

.. code-block:: fortran

   ! Timestep considerations
   DTDP = 0.5                ! May need smaller timestep

   ! Solver settings
   NCOR = 2                  ! Increased iterations may help convergence

   ! Output options
   NSCREEN = 1000           ! Monitor for stability

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Computational Impact:**

Subgrid corrections add computational cost:

* **Memory**: ~20-30% increase for subgrid arrays
* **Runtime**: ~10-20% increase for correction calculations
* **I/O**: Additional file reading during initialization

**Optimization Strategies:**

* Use moderate numbers of φ levels (25-75)
* Ensure subgrid file is on fast storage
* Consider domain decomposition impacts

Workflow Automation
-------------------

Batch Processing Scripts
~~~~~~~~~~~~~~~~~~~~~~~~

**Shell Script Example:**

.. code-block:: bash

   #!/bin/bash

   # Batch subgrid processing
   CONFIG_DIR="configs"
   OUTPUT_DIR="subgrid_outputs"

   # Create output directory
   mkdir -p $OUTPUT_DIR

   # Process each configuration
   for config in $CONFIG_DIR/*.yaml; do
       echo "Processing $config"
       adcirc-subgrid prep "$config"

       # Move output to organized directory
       output_file=$(grep "filename:" "$config" | cut -d: -f2 | tr -d ' ')
       mv "$output_file" "$OUTPUT_DIR/"
   done

**Python Workflow Manager:**

.. code-block:: python

   #!/usr/bin/env python3

   import subprocess
   import yaml
   from pathlib import Path

   def process_subgrid(config_file):
       """Process a single subgrid configuration."""
       cmd = ["adcirc-subgrid", "prep", str(config_file)]
       result = subprocess.run(cmd, capture_output=True, text=True)

       if result.returncode != 0:
           print(f"Error processing {config_file}:")
           print(result.stderr)
           return False

       return True

   # Main workflow
   config_dir = Path("configs")
   for config_file in config_dir.glob("*.yaml"):
       print(f"Processing {config_file.name}")
       success = process_subgrid(config_file)

       if success:
           print(f"  ✓ Completed successfully")
       else:
           print(f"  ✗ Failed")

Best Practices and Guidelines
-----------------------------

Resolution Design Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on extensive validation studies, follow these guidelines for optimal subgrid performance:

**Mesh Resolution Selection:**

1. **Identify critical features**: Barrier islands, major channels, flow-blocking structures
2. **Minimum resolution rule**: Ensure mesh resolves primary flow-blocking features in the underlying DEM
3. **Coarsening limits**: Avoid nearshore resolutions coarser than 400-1000m unless critical features are preserved
4. **Balance principle**: Optimize for specific application (forecasting vs. design studies)

**Data Quality Requirements:**

* **DEM resolution**: At least 3-20× finer than mesh resolution
* **Coverage completeness**: Full domain coverage with no gaps
* **Coordinate consistency**: All datasets in same coordinate reference system
* **Temporal consistency**: Recent, high-quality datasets for dynamic coastal areas

Memory and Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Lookup Table Management:**

* Use φ-based tables (N_φ = 11) instead of elevation-based tables (N_ζ = 401)
* Monitor memory usage for ocean-scale domains
* Consider preprocessing on high-memory systems if needed

**Processing Strategies:**

* Process in geographic regions if memory limited
* Use highest-priority datasets first in multi-dataset workflows
* Validate results with subset domains before full-scale processing

**Computational Considerations:**

* Typical overhead: 14-80% at same resolution
* Speedup potential: 10-50× on appropriately coarsened meshes
* Memory scaling: Approximately linear with domain size and mesh resolution

Validation and Quality Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended Validation Workflow:**

1. **Visual inspection**: Use mesh and node plotting tools extensively
2. **Physical consistency**: Check φ monotonicity and boundary conditions
3. **Comparative analysis**: Validate against high-resolution simulations when possible
4. **Statistical metrics**: Use ERMS, R², and bias measures for quantitative assessment
5. **Operational testing**: Test with representative storm events before production use

**Common Issues and Solutions:**

* **Non-monotonic φ**: Usually indicates DEM quality issues or processing errors
* **Over-connectivity**: May result from inadequate mesh resolution for barrier features
* **Memory issues**: Consider domain subdivision or φ-level reduction
* **Performance problems**: Check data coverage and coordinate system consistency

Integration with ADCIRC Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File Management:**

* Store subgrid files on fast storage systems
* Verify node count consistency between mesh and subgrid files
* Use descriptive naming conventions for version control

**Parameter Recommendations:**

* **Minimum wet depth**: 0.1m (⟨H⟩_{G,min}) for stability
* **φ levels**: 11 discrete levels provide good balance of accuracy and efficiency
* **Closure level**: Level 0 adequate for most applications; Level 1 for complex shallow regions

**Operational Considerations:**

* Pre-validate subgrid files before real-time forecasting
* Monitor model stability during initial runs with subgrid corrections
* Document specific parameter choices for reproducibility

This comprehensive usage guide provides the foundation for effective subgrid preprocessing workflows based on extensive research validation and operational experience.
