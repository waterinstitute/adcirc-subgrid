Examples and Tutorials
======================

This section provides step-by-step tutorials using the Galveston Bay (GBAY) example dataset to demonstrate typical subgrid preprocessing workflows.

Galveston Bay Example
---------------------

The GBAY example demonstrates multi-dataset subgrid processing using realistic coastal data from Galveston Bay, Texas. This tutorial covers both single-pass and incremental processing workflows.

Dataset Overview
~~~~~~~~~~~~~~~~

The example includes:

* **ADCIRC Mesh**: ``fort.14`` - Computational grid for Galveston Bay
* **DEM Files**: Two overlapping elevation datasets with different coverage

  * ``galveston_13_mhw_20072.TIF`` - High-priority elevation data
  * ``galveston_13_mhw_20073.TIF`` - Secondary elevation data

* **Land Cover**: ``2021_CCAP_J1139301_4326.tif`` - CCAP classification data
* **Configuration Files**:

  * ``input.yaml`` - Initial processing configuration
  * ``input_update_existing.yaml`` - Incremental update configuration

Prerequisites
~~~~~~~~~~~~~

Before starting, ensure you have:

1. **Completed installation** following the :doc:`installation` guide
2. **Downloaded example data** (if using Git LFS):

   .. code-block:: bash

      cd examples/GBAY
      git lfs fetch
      git lfs pull
      git lfs checkout
      gunzip fort.14.gz

3. **Activated environment**:

   .. code-block:: bash

      conda activate adcirc-subgrid

Tutorial 1: Single Dataset Processing
--------------------------------------

This tutorial demonstrates basic subgrid generation using a single elevation dataset.

Step 1: Examine the Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Review the basic configuration file:

.. code-block:: bash

   cd examples/GBAY
   cat input.yaml

.. code-block:: yaml

   input:
     adcirc_mesh: ./fort.14
     manning_lookup: ccap
     dem: ./galveston_13_mhw_20072.TIF
     land_cover: ./2021_CCAP_J1139301_4326.tif

   output:
     filename: subgrid.nc
     progress_bar_increment: 5

   options:
     n_subgrid_levels: 50
     n_phi_levels: 50
     subgrid_level_distribution: histogram

**Configuration Analysis:**

* **Input Data**: Uses the first elevation dataset and CCAP land cover
* **Manning's Lookup**: Built-in CCAP lookup table for friction coefficients
* **Processing Levels**: 50 levels for both computation and output
* **Distribution**: Histogram-based level distribution for efficiency

Step 2: Run the Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the subgrid generation:

.. code-block:: bash

   adcirc-subgrid --verbose prep input.yaml

**Expected Output:**

.. code-block:: text

   [2024-03-15 10:30:00] :: INFO :: Processing subgrid data
   [2024-03-15 10:30:01] :: INFO :: Reading ADCIRC mesh: ./fort.14
   [2024-03-15 10:30:02] :: INFO :: Loading DEM: ./galveston_13_mhw_20072.TIF
   [2024-03-15 10:30:05] :: INFO :: Loading land cover: ./2021_CCAP_J1139301_4326.tif
   [2024-03-15 10:30:06] :: INFO :: Processing node 1 of 15234 (0%)
   ...
   [2024-03-15 10:45:30] :: INFO :: Writing output file: subgrid.nc
   [2024-03-15 10:45:35] :: INFO :: Processing complete

**Performance Notes:**

* Processing time depends on mesh size and system performance
* Memory usage scales with DEM resolution and processing options
* Progress updates appear every 5% (configurable via ``progress_bar_increment``)

Step 3: Examine the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Check output file structure:**

.. code-block:: bash

   ncdump -h subgrid.nc

**Key variables in the output:**

* ``phi``: Subgrid correction factors (dimensions: node × phi_level)
* ``phi_level``: Water surface elevation levels
* ``manning_avg``: Average Manning's coefficients
* ``cf``, ``c_mf``, ``c_adv``: Various correction coefficients

Step 4: Quality Control Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create visualizations to verify results:

**Mesh-level visualization:**

.. code-block:: bash

   # Visualize φ at mean sea level
   adcirc-subgrid plot-mesh subgrid.nc --water-level 0.0 --variable percent_wet --show

   # Save plot for documentation
   adcirc-subgrid plot-mesh subgrid.nc --water-level 0.0 --variable percent_wet \
       --output-filename gbay_phi_msl.png --colorbar plasma

**Node-specific analysis:**

.. code-block:: bash

   # Examine φ vs water level for a specific node
   adcirc-subgrid plot-node --filename subgrid.nc --node 5000 --basis phi --show

   # Save node analysis
   adcirc-subgrid plot-node --filename subgrid.nc --node 5000 --basis phi \
       --save node_5000_analysis.png

Tutorial 2: Multi-Dataset Processing
-------------------------------------

This tutorial demonstrates incremental subgrid building using multiple datasets with different priorities.

Step 1: Complete Tutorial 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure you have completed Tutorial 1 and have ``subgrid.nc`` available.

Step 2: Examine the Update Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Review the incremental configuration:

.. code-block:: bash

   cat input_update_existing.yaml

.. code-block:: yaml

   input:
     adcirc_mesh: ./fort.14
     manning_lookup: ccap
     dem: ./galveston_13_mhw_20073.TIF  # Different elevation dataset
     land_cover: ./2021_CCAP_J1139301_4326.tif

   output:
     filename: subgrid_updated.nc  # Different output filename

   options:
     n_subgrid_levels: 50
     n_phi_levels: 50
     subgrid_level_distribution: histogram
     existing_subgrid: ./subgrid.nc  # Reference to existing data

**Key Differences:**

* **DEM Source**: Uses second elevation dataset
* **Output File**: Different filename to preserve original
* **Existing Subgrid**: References previous results for incremental updates

Step 3: Run Incremental Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the update processing:

.. code-block:: bash

   adcirc-subgrid --verbose prep input_update_existing.yaml

**Processing Behavior:**

* New elevation data supplements existing subgrid data
* Existing areas with data are **not overwritten**
* Only areas with missing data are populated from new dataset
* Final result combines both elevation datasets appropriately

Step 4: Compare Results
~~~~~~~~~~~~~~~~~~~~~~~

**Visualize differences between single and multi-dataset results:**

.. code-block:: bash

   # Original results
   adcirc-subgrid plot-mesh subgrid.nc --water-level 0.0 --variable percent_wet \
       --output-filename single_dataset.png --colorbar plasma

   # Updated results
   adcirc-subgrid plot-mesh subgrid_updated.nc --water-level 0.0 --variable percent_wet \
       --output-filename multi_dataset.png --colorbar plasma

**Quantitative Comparison:**

.. code-block:: python

   import xarray as xr
   import numpy as np

   # Load both datasets
   ds1 = xr.open_dataset('subgrid.nc')
   ds2 = xr.open_dataset('subgrid_updated.nc')

   # Compare φ values at mean sea level
   phi1 = ds1['phi'].sel(phi_level=0.0, method='nearest')
   phi2 = ds2['phi'].sel(phi_level=0.0, method='nearest')

   # Calculate differences
   diff = phi2 - phi1
   print(f"Max difference: {diff.max().item():.4f}")
   print(f"Mean absolute difference: {np.abs(diff).mean().item():.4f}")
   print(f"Number of changed nodes: {(np.abs(diff) > 0.001).sum().item()}")

Tutorial 3: Advanced Configuration
----------------------------------

This tutorial explores advanced configuration options for specialized applications.

Custom Manning's Lookup
~~~~~~~~~~~~~~~~~~~~~~~

Create a custom roughness lookup table:

**Step 1: Create custom lookup file**

.. code-block:: bash

   cat > custom_manning.csv << EOF
   Class,Manning's n
   1,0.025
   2,0.045
   3,0.080
   4,0.120
   5,0.030
   6,0.100
   7,0.055
   8,0.035
   9,0.180
   10,0.200
   EOF

**Step 2: Create configuration with custom lookup**

.. code-block:: yaml

   input:
     adcirc_mesh: ./fort.14
     manning_lookup: ./custom_manning.csv  # Use custom lookup
     dem: ./galveston_13_mhw_20072.TIF
     land_cover: ./2021_CCAP_J1139301_4326.tif

   output:
     filename: subgrid_custom_manning.nc

   options:
     n_subgrid_levels: 75  # Higher resolution
     n_phi_levels: 75
     subgrid_level_distribution: histogram
     distribution_factor: 1.5  # Extended range

High-Resolution Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration for maximum accuracy:

.. code-block:: yaml

   input:
     adcirc_mesh: ./fort.14
     manning_lookup: ccap
     dem: ./galveston_13_mhw_20072.TIF
     land_cover: ./2021_CCAP_J1139301_4326.tif

   output:
     filename: subgrid_high_res.nc
     progress_bar_increment: 1  # More frequent updates

   options:
     n_subgrid_levels: 100  # Maximum resolution
     n_phi_levels: 100
     subgrid_level_distribution: histogram
     distribution_factor: 2.0  # Extended vertical range

**Processing with increased memory allocation:**

.. code-block:: bash

   adcirc-subgrid prep high_res_config.yaml --window-memory 256

Tutorial 4: Visualization and Analysis
---------------------------------------

Comprehensive visualization workflow for quality assurance.

Mesh-Level Analysis
~~~~~~~~~~~~~~~~~~~

**Create comprehensive visualization set:**

.. code-block:: bash

   # Different water levels
   for level in 0.0 0.5 1.0 1.5 2.0; do
       adcirc-subgrid plot-mesh subgrid.nc --water-level $level \
           --variable percent_wet --output-filename "phi_level_${level}m.png" \
           --colorbar viridis --range 0.0 1.0
   done

   # Different variables at MSL
   for var in percent_wet wet_depth total_depth cf; do
       adcirc-subgrid plot-mesh subgrid.nc --water-level 0.0 \
           --variable $var --output-filename "msl_${var}.png" \
           --colorbar plasma
   done

**Regional Focus:**

.. code-block:: bash

   # Focus on specific geographic region
   adcirc-subgrid plot-mesh subgrid.nc --water-level 0.0 \
       --variable percent_wet --bbox -95.5 29.0 -94.5 29.5 \
       --output-filename region_detail.png --colorbar viridis

Node-Level Analysis
~~~~~~~~~~~~~~~~~~~

**Systematic node analysis:**

.. code-block:: bash

   # Analyze representative nodes
   nodes=(1000 5000 10000 12000)

   for node in "${nodes[@]}"; do
       echo "Analyzing node $node"
       adcirc-subgrid plot-node --filename subgrid.nc --node $node \
           --basis phi --save "analysis_node_${node}.png"
   done

**Python-based batch analysis:**

.. code-block:: python

   import xarray as xr
   import matplotlib.pyplot as plt
   import numpy as np

   # Load subgrid data
   ds = xr.open_dataset('subgrid.nc')
   phi = ds['phi']
   levels = ds['phi_level']

   # Analyze multiple nodes
   nodes_to_check = [1000, 5000, 10000, 12000]

   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
   axes = axes.flatten()

   for i, node in enumerate(nodes_to_check):
       ax = axes[i]
       node_phi = phi.isel(node=node)
       ax.plot(levels, node_phi, 'b-', linewidth=2)
       ax.set_xlabel('Water Level (m)')
       ax.set_ylabel('φ Factor')
       ax.set_title(f'Node {node}')
       ax.grid(True, alpha=0.3)
       ax.set_xlim(levels.min(), levels.max())
       ax.set_ylim(0, 1)

   plt.tight_layout()
   plt.savefig('multi_node_analysis.png', dpi=300, bbox_inches='tight')
   plt.show()

Tutorial 5: Integration with ADCIRC
-----------------------------------

Preparing subgrid data for use in ADCIRC simulations.

ADCIRC Configuration
~~~~~~~~~~~~~~~~~~~~

**Configure ADCIRC to use subgrid corrections:**

.. code-block:: fortran

   ! In fort.15 (ADCIRC control file)

   ! Enable subgrid physics
   NWP = 1

   ! Specify subgrid file
   ! Add this line after the main parameter section
   NCFILE = 'subgrid.nc'

**Recommended parameter adjustments:**

.. code-block:: fortran

   ! Time stepping (may need adjustment for stability)
   DTDP = 0.5                    ! Potentially smaller timestep

   ! Nonlinear iterations
   NCOR = 2                      ! May need more iterations

   ! Solver parameters
   NOLIBF = 2                    ! Finite element solver
   NOLIFA = 2                    ! Advection solver
   NOLICAT = 1                   ! Category solver

File Management
~~~~~~~~~~~~~~~

**Organize files for ADCIRC run:**

.. code-block:: bash

   # Create ADCIRC run directory
   mkdir adcirc_run
   cd adcirc_run

   # Copy required files
   cp ../fort.14 .                    # Mesh file
   cp ../subgrid_updated.nc subgrid.nc  # Subgrid corrections
   # Copy other ADCIRC input files (fort.15, fort.13, etc.)

**Verify subgrid file compatibility:**

.. code-block:: bash

   # Check that node count matches
   mesh_nodes=$(grep -A1 "NE NN" fort.14 | tail -n1 | awk '{print $2}')
   subgrid_nodes=$(ncdump -h subgrid.nc | grep "node =" | grep -o '[0-9]*')

   echo "Mesh nodes: $mesh_nodes"
   echo "Subgrid nodes: $subgrid_nodes"

   if [ "$mesh_nodes" -eq "$subgrid_nodes" ]; then
       echo "✓ Node counts match"
   else
       echo "✗ Node count mismatch!"
   fi

Performance Testing
~~~~~~~~~~~~~~~~~~~

**Quick validation run:**

.. code-block:: bash

   # Short test run to verify functionality
   # Create minimal fort.15 for testing
   cat > fort.15 << EOF
   ADCIRC Test Run with Subgrid
   1                    ! NFOVER
   test_output          ! RUNID
   1                    ! NFOVER
   2                    ! NSCREEN
   0.0 10.0 0.5 0       ! RNDAY, DRAMP, DT, STATIM
   1 0.005 2 0.2        ! NOLIBF, TAU, CF, ESLM
   0 0 0                ! NCOR, NTIP, NWS
   1                    ! NWP (enable subgrid)
   subgrid.nc           ! NCFILE
   EOF

   # Run ADCIRC (adjust for your installation)
   adcirc < fort.15

Best Practices Summary
----------------------

Based on the tutorials above, follow these best practices:

Data Quality
~~~~~~~~~~~~

1. **Verify coordinate system consistency** across all input files
2. **Ensure complete domain coverage** by DEM and land cover data
3. **Use highest quality data first** in multi-dataset workflows
4. **Validate input data independently** before processing

Processing Strategy
~~~~~~~~~~~~~~~~~~~

1. **Start with moderate resolution** (25-50 levels) for initial testing
2. **Use histogram distribution** for computational efficiency
3. **Process incrementally** with multiple datasets prioritized by quality
4. **Monitor memory usage** and adjust window memory as needed

Quality Control
~~~~~~~~~~~~~~~

1. **Always visualize results** before using in production
2. **Check φ ranges and monotonicity** for physical validity
3. **Compare with known results** when available
4. **Test integration with ADCIRC** before large-scale runs

This comprehensive set of examples provides the foundation for successful subgrid preprocessing across a wide range of applications.
