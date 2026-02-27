ADCIRC-Subgrid Example
=======================


ADCIRC-Subgrid Feature: subgrid_level_distribution
--------------------------------------------------

Objective
~~~~~~~~~~~~~~~~

Subgrid distributes the wet percentage in the DEM (based on the phi-levels) linearly or via a histogram. The source code ``calculation_levels.py`` is used to compute the levels of subgrid calculations or calculation intervals for water surface elevations. This example is designed to understand:

**How significant are the lookup table results when you choose histogram vs linear distribution for levels?**

For this, two different regions will be investigated. We will investigate the results of the GBAY example with both linear and histogram distribution levels and compare them with a comparatively less complex topobathy in South-west Florida, also using the two different distribution levels. We will run the ADCIRC-Subgrid model for a small section of the coast in the southwest part of Florida, as shown in _Figure 1_, and compare the results to determine which distribution best describes the specific topography. This example assumes the ADCIRC-Subgrid system is already downloaded and set up.

**Disclaimer**: The differences in water level or water depths using linear and histogram distribution levels are very subtle.

GOALS of this example:
- run GBAY again using linear distribution level
- run the model using a different set of data at two different levels
- view the results

.. raw:: html

  <p align="center">
    <img src="images/dem_extent.png" width="60%" alt="DEM and map extent with Midnight Pass location" />
    <br>
    <em> Figure 1: DEM and map extent with Midnight Pass location in the south-west part of Florida.</em>
  </p>


Dataset Overview
~~~~~~~~~~~~~~~~
The example includes:

1. **ADCIRC Mesh**: ``fort.14`` - Computational grid for south west region of Florida. This is a clipped version of the ADCIRC EGOM mesh, extracted for the area of interest.

.. raw:: html

  <p align="center">
    <img src="images/grid.png" width="40%" alt="Grid for the Midnight Pass location" />
    <br>
    <em> Figure 2: Clip of the EGOM mesh.</em>
  </p>


2. **DEM File**: The digital elevation file for the run area. It should have the same coordinate system as the fort.14, i.e., EPSG 4326
``dem3_wgs.TIF`` - 1m resolution with a geographical spatial resolution of WGS 1984.

.. raw:: html

  <p align="center">
    <img src="images/dem.png" width="40%" alt="DEM for the Midnight Pass location" />
    <br>
    <em> Figure 3: DEM of the region in Florida.</em>
  </p>


3. **Land Cover**: ``landusemap_wgs.tif`` - CCAP classification data High-resolution (1m) land-use map of Tampa, Florida, for 2021. The provided land-use map is for the modeled region.

.. raw:: html

  <p align="center">
    <img src="images/landuse_map.png" width="40%" alt="The land-use map of the modeled region." />
    <br>
    <em>Figure 4: The land-use map of the modeled region in Florida.</em>
  </p>

**link to the DEM data and landcover data**: `Download subgrid-fl.zip <https://go.ncsu.edu/subgrid-fl.zip>`_ to get the DEM file for this example as **FL.zip**

4. **Configuration Files**: The yaml file should have the following configuration specifying the input data, the output data name, and the run options used. The 'subgrid_level_distribution' will be changed between histogram and linear in this example.

  * ``input.yaml`` - Run Configuration


.. code-block:: yaml

   input:
     adcirc_mesh: ./fort.14
     manning_lookup: ccap # Either a lookup file or 'ccap' to use the default table
     dem: ./DEM3_wgs.tif
     land_cover: ./landusemap_wgs.tif

   output:
     filename: subgrid_lin.nc
     progress_bar_increment: 5

   options:
     # Control for the number of subgrid levels for calculation and output
     n_subgrid_levels: 50 # Controls the number of levels the calculation is performed on
     n_phi_levels: 50 # Controls the number of phi levels between 0 and 1 where output is written

     # Control for the way the subgrid water levels are distributed
     subgrid_level_distribution: linear # Either 'histogram' or 'linear'

Run the Preprocessing
~~~~~~~~~~~~~~~~~~~~~

Execute the subgrid generation:

.. code-block:: bash

   adcirc-subgrid prep input.yaml

**Expected Output:**

.. code-block:: text

   [2025-07-31 12:42:17,447] :: INFO :: AdcircSubgrid.subgrid_cli :: Running preprocessor with config file input.yaml
   [2025-07-31 12:42:29,466] :: INFO :: AdcircSubgrid.subarea_polygons :: Computed subarea polygons in 3.68 seconds
   [2025-07-31 12:42:32,720] :: INFO :: AdcircSubgrid.raster :: Raster will be read using 12 windows
   [2025-07-31 12:42:32,724] :: INFO :: AdcircSubgrid.preprocessor :: Subgrid parameters will be computed on 50 levels
   [2025-07-31 12:42:32,724] :: INFO :: AdcircSubgrid.preprocessor :: Subgrid parameters will be written to 50 phi levels
   [2025-07-31 12:42:32,724] :: INFO :: AdcircSubgrid.preprocessor :: Finding nodes in raster windows
   [2025-07-31 12:42:32,807] :: INFO :: AdcircSubgrid.preprocessor :: Found 7090 of 7090 nodes (100.00%) within the raster
   [2025-07-31 12:42:55,859] :: INFO :: AdcircSubgrid.preprocessor ::   0%|                                                           | 1/7090 [00:20<40:53:47, 20.77s/it]
   [2025-07-31 12:43:27,523] :: INFO :: AdcircSubgrid.preprocessor ::   5%|███                                                         | 356/7090 [00:52<07:55, 14.17it/s]
   [2025-07-31 12:43:46,477] :: INFO :: AdcircSubgrid.preprocessor ::  10%|██████                                                      | 711/7090 [01:11<03:26, 30.82it/s]
   [2025-07-31 12:44:01,359] :: INFO :: AdcircSubgrid.preprocessor ::  15%|████████▊                                                  | 1066/7090 [01:26<04:09, 24.11it/s]
   [2025-07-31 12:44:12,914] :: INFO :: AdcircSubgrid.preprocessor ::  20%|███████████▊                                               | 1421/7090 [01:37<02:47, 33.92it/s]
   [2025-07-31 12:44:20,696] :: INFO :: AdcircSubgrid.preprocessor ::  25%|██████████████▊                                            | 1776/7090 [01:45<01:48, 49.01it/s]
   [2025-07-31 12:44:29,053] :: INFO :: AdcircSubgrid.preprocessor ::  30%|█████████████████▋                                         | 2131/7090 [01:53<01:53, 43.57it/s]
   ...

   [2025-07-31 12:45:17,174] :: INFO :: AdcircSubgrid.preprocessor ::  85%|█████████████████████████████████████████████████▍        | 6036/7090 [02:42<00:07, 132.55it/s]
   [2025-07-31 12:45:21,672] :: INFO :: AdcircSubgrid.preprocessor ::  90%|█████████████████████████████████████████████████████▏     | 6391/7090 [02:46<00:17, 39.08it/s]
   [2025-07-31 12:45:26,615] :: INFO :: AdcircSubgrid.preprocessor ::  95%|████████████████████████████████████████████████████████▏  | 6746/7090 [02:51<00:04, 72.85it/s]
   [2025-07-31 12:45:31,500] :: INFO :: AdcircSubgrid.preprocessor :: 100%|███████████████████████████████████████████████████████████| 7090/7090 [02:56<00:00, 48.14it/s]
   [2025-07-31 12:45:31,503] :: INFO :: AdcircSubgrid.preprocessor :: Writing output to subgrid_lin.nc


Change the "subgrid_level_distribution" in the input.yaml file from linear to histogram and change the output name to subgrid_hist.nc. Run subgrid again.

Examine the Results
~~~~~~~~~~~~~~~~~~~
The results can be viewed by using Python, specifically the mesh_plot function provided in the source code. Wet percentage, water depth, or water levels can be viewed at a specific water level using the code below.

.. code-block:: bash

   import os
   sys.path.append(path to <\src\AdcircSubgrid>)
   import mesh_plot

   ncpath = <path to netCDF results folder>
   ncf = os.path.join(ncpath,'subgrid_lin.nc') # subgrid_hist.nc
   level = 0
   outfig = os.path.join(ncpath,f"hist_gb_wd_{level}")
   mesh_plot.plot_mesh(ncf, 'wet_depth', level, True, outfig)

When investigating the phi values ('phi_set' in the nc file) from both linear and histogram distributions, you'll notice that both of them are identical (values and numbers). This is because all of the subgrid values are interpolated to the same phi interval at the end of the code, as specified in the 'n_phi_levels' and 'n_subgrid_levels' set in the YAML files.

For the linear distribution, the lowest and highest DEM cell values within a subgrid vertex area are found, and water surface elevations are evenly spaced between those two values. For the histogram, the values are spaced according to the distribution of DEM cell values within the subgrid vertex area. Then, all subgrid variables are interpolated to match wet area fractions (phis) evenly ranging from 0 to 1.

The plot for the water depth using linear and histogram distribution levels for this example is shown in Figure 5. The visual difference might not be apparent in the plots, as mentioned, but it can be quantified using, for example, total water depth at each vertex. Here, the total water depth for the linear and histogram distribution levels was found and subtracted to get the following statistics: there are 13% more wet nodes in the histogram distribution level than in the linear.

.. code-block:: text

   Max difference: 1.8282
   Mean absolute difference: 0.0325
   Number of changed nodes from total: 960/7090

.. raw:: html

   <p align="center">
     <img src="images/lin_mp_wd_0.png" width="40%" alt="extentt" style="display:inline-block; margin-            right:10px;" />
     <img src="images/hist_mp_wd_0.png" width="40%" alt="DEMs" style="display:inline-block;" />
     <br>
     <em>Fig 5: water depth at 0m elevation using linear distribution level (left) and histogram distribution level (right)</em>
   </p>

The water depth for the GBAY example was investigated similarly, and its plot is shown in Figure 6. The visual differences can be seen
more clearly near the river source or the edges near the land. The statistics are also more apparent: 30% of nodes are wetter in the histogram than in the linear distribution method:

.. code-block:: text

   Max difference: 10.8832
   Mean absolute difference: 0.2176
   Number of changed nodes from total: 3642/12024

.. raw:: html

   <p align="center">
     <img src="images/lin_gb_wd_0.png" width="40%" alt="extentt" style="display:inline-block; margin-            right:10px;" />
     <img src="images/hist_gb_wd_0.png" width="40%" alt="DEMs" style="display:inline-block;" />
     <br>
     <em>Fig 6: water depth for GBAY example at 0m elevation using linear distribution level (left) and histogram distribution level (right)</em>
   </p>

In this example, we see that the histogram method works better for river estuaries (like in the GBAY case). For flatter areas, such as barrier islands, both the linear and histogram methods give similar results.

The reason is that the histogram method adjusts elevation levels based on cell numbers at each elevation level, which captures the elevation changes, or high elevation gradient, near the river mouth more accurately. In contrast, the linear method spreads elevations evenly from high to low, which misses some of these elevation gradients. This enhances the histogram method's resolution near the river mouth, making it more effective at modeling the narrow edges of the river mouth for domains with river estuaries.

