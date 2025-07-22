Visualization and Analysis
===========================

The ADCIRC Subgrid Preprocessor provides comprehensive visualization capabilities for quality control, result analysis, and scientific understanding of subgrid corrections.

Overview of Visualization Tools
-------------------------------

The package includes two primary visualization commands:

1. **plot-mesh**: Creates spatial visualizations of subgrid data across the computational mesh
2. **plot-node**: Generates detailed analyses of subgrid relationships for individual mesh nodes

These tools serve different purposes in the subgrid analysis workflow:

* **Quality Control**: Identifying data issues, processing errors, or unrealistic results
* **Scientific Analysis**: Understanding the relationship between topography and subgrid corrections
* **Results Communication**: Creating publication-quality figures for reports and presentations

Mesh-Level Visualization
------------------------

The ``plot-mesh`` command creates spatial maps showing how subgrid variables vary across the computational domain.

Basic Mesh Plotting
~~~~~~~~~~~~~~~~~~~~

**Command Syntax:**

.. code-block:: bash

   adcirc-subgrid plot-mesh [options] subgrid_file.nc

**Essential Options:**

* ``--water-level LEVEL``: Water surface elevation for visualization (default: 0.0)
* ``--variable VAR``: Variable to plot (required)
* ``--show``: Display plot interactively
* ``--output-filename FILE``: Save plot to file

**Basic Example:**

.. code-block:: bash

   # Display phi factor at mean sea level
   adcirc-subgrid plot-mesh subgrid.nc --variable percent_wet --water-level 0.0 --show

Available Variables
~~~~~~~~~~~~~~~~~~~

**Primary Variables:**

* **percent_wet**: φ factor (fraction of element that is wet)
* **wet_depth**: Average depth in wet portions of element
* **total_depth**: Total water depth (including corrections)
* **cf**: Friction coefficient
* **c_mf**: Manning friction coefficient
* **c_adv**: Advection coefficient

**Variable Descriptions:**

**percent_wet (φ factor)**
  * **Range**: 0.0 to 1.0
  * **Interpretation**: Fraction of mesh element that is submerged
  * **Use Cases**: Understanding wetting patterns, identifying problematic areas
  * **Visualization Tips**: Use diverging colormaps to highlight transition zones

**wet_depth**
  * **Units**: Meters
  * **Interpretation**: Average depth within wet portions of element
  * **Use Cases**: Assessing effective water depth for flow calculations
  * **Visualization Tips**: Use sequential colormaps with appropriate depth ranges

**total_depth**
  * **Units**: Meters
  * **Interpretation**: Total water column including subgrid corrections
  * **Use Cases**: Comparing with original mesh bathymetry
  * **Visualization Tips**: Similar to wet_depth but may show larger values

Advanced Plotting Options
~~~~~~~~~~~~~~~~~~~~~~~~~

**Spatial Extent Control:**

.. code-block:: bash

   # Focus on specific geographic region
   adcirc-subgrid plot-mesh subgrid.nc --variable percent_wet \
       --bbox -95.5 29.0 -94.5 29.5 --water-level 0.0 --show

**Colormap and Range Control:**

.. code-block:: bash

   # Customize visualization appearance
   adcirc-subgrid plot-mesh subgrid.nc --variable percent_wet \
       --water-level 1.0 --range 0.0 1.0 --colorbar viridis \
       --output-filename phi_1m_elevation.png

**Multiple Water Levels:**

.. code-block:: bash

   # Create series of plots for different water levels
   for level in 0.0 0.5 1.0 1.5 2.0; do
       adcirc-subgrid plot-mesh subgrid.nc --variable percent_wet \
           --water-level $level --range 0.0 1.0 --colorbar plasma \
           --output-filename "phi_${level}m.png"
   done

Node-Level Visualization
------------------------

The ``plot-node`` command provides detailed analysis of subgrid relationships at individual mesh nodes.

Basic Node Plotting
~~~~~~~~~~~~~~~~~~~~

**Command Syntax:**

.. code-block:: bash

   adcirc-subgrid plot-node --filename subgrid.nc --node NODE_ID [options]

**Essential Options:**

* ``--filename FILE``: Subgrid NetCDF file (required)
* ``--node NODE_ID``: Node number to analyze (required)
* ``--basis BASIS``: Plot basis (phi or wse)
* ``--show``: Display plot interactively
* ``--save FILE``: Save plot to file
* ``--index-base BASE``: Node numbering convention (0 or 1)

**Basic Example:**

.. code-block:: bash

   # Analyze phi vs water level for node 5000
   adcirc-subgrid plot-node --filename subgrid.nc --node 5000 \
       --basis phi --show

Plot Basis Options
~~~~~~~~~~~~~~~~~~

**phi basis**
  * **X-axis**: Water surface elevation (m)
  * **Y-axis**: φ factor (0-1)
  * **Purpose**: Understanding wetting behavior as function of water level
  * **Interpretation**: Shows how element wetted area changes with water level

**wse basis (Water Surface Elevation)**
  * **X-axis**: φ factor (0-1)
  * **Y-axis**: Water surface elevation (m)
  * **Purpose**: Understanding inverse relationship
  * **Interpretation**: Shows water level required for given wetted fraction

Node Selection Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

**Representative Sampling:**

.. code-block:: bash

   # Analyze nodes in different topographic settings
   # Deep water node
   adcirc-subgrid plot-node --filename subgrid.nc --node 1000 --basis phi --save deep_water.png

   # Shallow water node
   adcirc-subgrid plot-node --filename subgrid.nc --node 5000 --basis phi --save shallow_water.png

   # Intertidal node
   adcirc-subgrid plot-node --filename subgrid.nc --node 8000 --basis phi --save intertidal.png

**Problem Diagnosis:**

.. code-block:: bash

   # Investigate nodes with unusual behavior
   # (identified from mesh plots)
   adcirc-subgrid plot-node --filename subgrid.nc --node 12500 \
       --basis phi --save problematic_node.png

Batch Visualization Workflows
-----------------------------

Python-Based Automation
~~~~~~~~~~~~~~~~~~~~~~~~

Create comprehensive visualization sets using Python:

.. code-block:: python

   import subprocess
   import numpy as np
   from pathlib import Path

   def create_water_level_series(subgrid_file, output_dir, variable='percent_wet'):
       """Create visualization series for multiple water levels."""

       output_dir = Path(output_dir)
       output_dir.mkdir(exist_ok=True)

       # Define water levels to visualize
       water_levels = np.arange(0.0, 3.1, 0.5)

       for level in water_levels:
           output_file = output_dir / f"{variable}_level_{level:.1f}m.png"

           cmd = [
               'adcirc-subgrid', 'plot-mesh', str(subgrid_file),
               '--variable', variable,
               '--water-level', str(level),
               '--colorbar', 'viridis',
               '--output-filename', str(output_file)
           ]

           print(f"Creating plot for water level {level:.1f}m")
           subprocess.run(cmd, check=True)

   # Usage
   create_water_level_series('subgrid.nc', 'visualization_series')

**Multi-Variable Analysis:**

.. code-block:: python

   def create_variable_comparison(subgrid_file, water_level, output_dir):
       """Create comparison plots for different variables at fixed water level."""

       variables = ['percent_wet', 'wet_depth', 'total_depth', 'cf']
       colormaps = ['viridis', 'plasma', 'cividis', 'inferno']

       output_dir = Path(output_dir)
       output_dir.mkdir(exist_ok=True)

       for var, cmap in zip(variables, colormaps):
           output_file = output_dir / f"{var}_wl_{water_level:.1f}m.png"

           cmd = [
               'adcirc-subgrid', 'plot-mesh', str(subgrid_file),
               '--variable', var,
               '--water-level', str(water_level),
               '--colorbar', cmap,
               '--output-filename', str(output_file)
           ]

           subprocess.run(cmd, check=True)

Shell Script Automation
~~~~~~~~~~~~~~~~~~~~~~~

**Comprehensive Analysis Script:**

.. code-block:: bash

   #!/bin/bash

   # Comprehensive subgrid visualization script
   SUBGRID_FILE="$1"
   OUTPUT_DIR="analysis_$(date +%Y%m%d_%H%M%S)"

   if [ -z "$SUBGRID_FILE" ]; then
       echo "Usage: $0 <subgrid_file.nc>"
       exit 1
   fi

   mkdir -p "$OUTPUT_DIR"
   cd "$OUTPUT_DIR"

   echo "Creating mesh visualizations..."

   # Water level series for phi
   for level in 0.0 0.5 1.0 1.5 2.0 2.5 3.0; do
       echo "  Water level: ${level}m"
       adcirc-subgrid plot-mesh "../$SUBGRID_FILE" \
           --variable percent_wet --water-level $level \
           --colorbar viridis --range 0.0 1.0 \
           --output-filename "phi_${level}m.png"
   done

   # Variable comparison at MSL
   echo "Creating variable comparisons at MSL..."
   for var in percent_wet wet_depth total_depth cf; do
       echo "  Variable: $var"
       adcirc-subgrid plot-mesh "../$SUBGRID_FILE" \
           --variable $var --water-level 0.0 \
           --colorbar plasma --output-filename "msl_${var}.png"
   done

   # Representative node analysis
   echo "Creating node analyses..."
   nodes=(1000 5000 10000 15000)
   for node in "${nodes[@]}"; do
       echo "  Node: $node"
       adcirc-subgrid plot-node --filename "../$SUBGRID_FILE" \
           --node $node --basis phi --save "node_${node}_phi.png"
   done

   echo "Analysis complete in directory: $OUTPUT_DIR"

Quality Control Visualization
-----------------------------

Identifying Common Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Non-Physical φ Values:**

Create diagnostic plots to identify nodes with φ values outside [0,1]:

.. code-block:: python

   import xarray as xr
   import matplotlib.pyplot as plt
   import numpy as np

   # Load subgrid data
   ds = xr.open_dataset('subgrid.nc')
   phi = ds['phi']

   # Check for values outside valid range
   invalid_low = (phi < 0).any(dim='phi_level')
   invalid_high = (phi > 1).any(dim='phi_level')

   # Create diagnostic plot
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   # Plot nodes with phi < 0
   ax1.scatter(ds['x'].where(invalid_low), ds['y'].where(invalid_low),
               c='red', s=1, alpha=0.7)
   ax1.set_title('Nodes with φ < 0')
   ax1.set_xlabel('Longitude')
   ax1.set_ylabel('Latitude')

   # Plot nodes with phi > 1
   ax2.scatter(ds['x'].where(invalid_high), ds['y'].where(invalid_high),
               c='blue', s=1, alpha=0.7)
   ax2.set_title('Nodes with φ > 1')
   ax2.set_xlabel('Longitude')
   ax2.set_ylabel('Latitude')

   plt.tight_layout()
   plt.savefig('phi_validation.png', dpi=300)
   plt.show()

**Monotonicity Checking:**

Verify that φ increases monotonically with water level:

.. code-block:: python

   def check_monotonicity(subgrid_file):
       """Check for non-monotonic phi relationships."""

       ds = xr.open_dataset(subgrid_file)
       phi = ds['phi']

       # Compute differences between consecutive phi levels
       phi_diff = phi.diff(dim='phi_level')

       # Find nodes with decreasing phi
       non_monotonic = (phi_diff < 0).any(dim='phi_level')

       # Count and locate problematic nodes
       n_problems = non_monotonic.sum().item()

       if n_problems > 0:
           print(f"Found {n_problems} nodes with non-monotonic phi")

           # Get coordinates of problematic nodes
           problem_coords = ds[['x', 'y']].where(non_monotonic, drop=True)

           # Create diagnostic plot
           plt.figure(figsize=(10, 8))
           plt.scatter(problem_coords['x'], problem_coords['y'],
                      c='red', s=5, alpha=0.7)
           plt.title('Nodes with Non-Monotonic φ Relationships')
           plt.xlabel('Longitude')
           plt.ylabel('Latitude')
           plt.savefig('monotonicity_problems.png', dpi=300)
           plt.show()
       else:
           print("All nodes have monotonic phi relationships")

Advanced Analysis Techniques
----------------------------

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

**φ Distribution Analysis:**

.. code-block:: python

   import xarray as xr
   import matplotlib.pyplot as plt
   import numpy as np

   def analyze_phi_distribution(subgrid_file, water_level=0.0):
       """Analyze statistical distribution of phi values."""

       ds = xr.open_dataset(subgrid_file)

       # Interpolate phi to specific water level
       phi_wl = ds['phi'].interp(phi_level=water_level, method='linear')

       # Create histogram
       plt.figure(figsize=(12, 4))

       # Histogram
       plt.subplot(1, 3, 1)
       plt.hist(phi_wl.values.flatten(), bins=50, alpha=0.7, density=True)
       plt.xlabel('φ Factor')
       plt.ylabel('Density')
       plt.title(f'φ Distribution at {water_level}m')

       # Cumulative distribution
       plt.subplot(1, 3, 2)
       sorted_phi = np.sort(phi_wl.values.flatten())
       cdf = np.arange(1, len(sorted_phi) + 1) / len(sorted_phi)
       plt.plot(sorted_phi, cdf)
       plt.xlabel('φ Factor')
       plt.ylabel('Cumulative Probability')
       plt.title('Cumulative Distribution')

       # Statistics
       plt.subplot(1, 3, 3)
       stats = phi_wl.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
       plt.bar(range(len(stats)), stats.values)
       plt.xticks(range(len(stats)), ['10%', '25%', '50%', '75%', '90%'])
       plt.ylabel('φ Factor')
       plt.title('Quantile Statistics')

       plt.tight_layout()
       plt.savefig(f'phi_analysis_{water_level}m.png', dpi=300)
       plt.show()

Comparative Analysis
~~~~~~~~~~~~~~~~~~~~

**Before/After Comparison:**

Compare subgrid results from different processing runs:

.. code-block:: python

   def compare_subgrid_results(file1, file2, water_level=0.0):
       """Compare phi values between two subgrid files."""

       ds1 = xr.open_dataset(file1)
       ds2 = xr.open_dataset(file2)

       # Interpolate to same water level
       phi1 = ds1['phi'].interp(phi_level=water_level)
       phi2 = ds2['phi'].interp(phi_level=water_level)

       # Compute difference
       phi_diff = phi2 - phi1

       # Create comparison plots
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))

       # Original phi
       im1 = axes[0,0].scatter(ds1['x'], ds1['y'], c=phi1, s=0.5, cmap='viridis')
       axes[0,0].set_title('Original φ')
       plt.colorbar(im1, ax=axes[0,0])

       # Updated phi
       im2 = axes[0,1].scatter(ds2['x'], ds2['y'], c=phi2, s=0.5, cmap='viridis')
       axes[0,1].set_title('Updated φ')
       plt.colorbar(im2, ax=axes[0,1])

       # Difference
       im3 = axes[1,0].scatter(ds1['x'], ds1['y'], c=phi_diff, s=0.5,
                               cmap='RdBu_r', vmin=-0.2, vmax=0.2)
       axes[1,0].set_title('Difference (Updated - Original)')
       plt.colorbar(im3, ax=axes[1,0])

       # Histogram of differences
       axes[1,1].hist(phi_diff.values.flatten(), bins=50, alpha=0.7)
       axes[1,1].set_xlabel('φ Difference')
       axes[1,1].set_ylabel('Frequency')
       axes[1,1].set_title('Difference Distribution')

       plt.tight_layout()
       plt.savefig('subgrid_comparison.png', dpi=300)
       plt.show()

Publication-Quality Figures
---------------------------

Creating High-Quality Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Matplotlib Styling:**

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.style as mplstyle

   # Use publication-ready style
   plt.style.use(['seaborn-v0_8-paper', 'seaborn-v0_8-colorblind'])

   # Custom parameters for publications
   plt.rcParams.update({
       'font.size': 12,
       'axes.titlesize': 14,
       'axes.labelsize': 12,
       'xtick.labelsize': 10,
       'ytick.labelsize': 10,
       'legend.fontsize': 10,
       'figure.dpi': 300,
       'savefig.dpi': 300,
       'savefig.bbox': 'tight',
       'savefig.pad_inches': 0.1
   })

**Multi-Panel Figures:**

.. code-block:: python

   def create_multipanel_figure(subgrid_file):
       """Create comprehensive multi-panel figure."""

       ds = xr.open_dataset(subgrid_file)

       fig = plt.figure(figsize=(16, 12))

       # Panel A: phi at MSL
       ax1 = plt.subplot(2, 3, 1)
       phi_msl = ds['phi'].interp(phi_level=0.0)
       scatter1 = ax1.scatter(ds['x'], ds['y'], c=phi_msl, s=0.5,
                             cmap='viridis', vmin=0, vmax=1)
       ax1.set_title('(a) φ Factor at MSL')
       ax1.set_xlabel('Longitude (°)')
       ax1.set_ylabel('Latitude (°)')
       plt.colorbar(scatter1, ax=ax1, shrink=0.8)

       # Panel B: phi at +1m
       ax2 = plt.subplot(2, 3, 2)
       phi_1m = ds['phi'].interp(phi_level=1.0)
       scatter2 = ax2.scatter(ds['x'], ds['y'], c=phi_1m, s=0.5,
                             cmap='viridis', vmin=0, vmax=1)
       ax2.set_title('(b) φ Factor at +1m')
       ax2.set_xlabel('Longitude (°)')
       ax2.set_ylabel('Latitude (°)')
       plt.colorbar(scatter2, ax=ax2, shrink=0.8)

       # Panel C: Manning's coefficients
       ax3 = plt.subplot(2, 3, 3)
       manning = ds['manning_avg']
       scatter3 = ax3.scatter(ds['x'], ds['y'], c=manning, s=0.5,
                             cmap='plasma', vmin=0.02, vmax=0.2)
       ax3.set_title('(c) Average Manning\'s n')
       ax3.set_xlabel('Longitude (°)')
       ax3.set_ylabel('Latitude (°)')
       plt.colorbar(scatter3, ax=ax3, shrink=0.8)

       # Panel D: Node analysis examples
       ax4 = plt.subplot(2, 3, (4, 6))
       nodes_to_plot = [1000, 5000, 10000]
       colors = ['red', 'blue', 'green']

       for node, color in zip(nodes_to_plot, colors):
           phi_node = ds['phi'].isel(node=node)
           levels = ds['phi_level']
           ax4.plot(levels, phi_node, color=color, linewidth=2,
                   label=f'Node {node}')

       ax4.set_xlabel('Water Level (m)')
       ax4.set_ylabel('φ Factor')
       ax4.set_title('(d) Representative φ vs Water Level')
       ax4.legend()
       ax4.grid(True, alpha=0.3)
       ax4.set_xlim(levels.min(), levels.max())
       ax4.set_ylim(0, 1)

       plt.tight_layout()
       plt.savefig('comprehensive_subgrid_analysis.png', dpi=300, bbox_inches='tight')
       plt.show()

This comprehensive visualization guide enables effective analysis and presentation of subgrid results for both technical validation and scientific communication.
