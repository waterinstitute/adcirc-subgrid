
# ADCIRC-Subgrid Example

## ADCIRC-Subgrid Feature

This example is trying to show what the option `subgrid_level_distribution` does. It distributes the wet percentage in the DEM (based on the phi-levels) linearly or via a histogram. It uses the source code `calculation_levels.py` to compute the levels of subgrid calculations or calculation intervals for water surface elevations.

**Research Question:**
*How significant are the lookup table results when you choose histogram vs linear distribution for levels?*

## Geographic Region

The example investigate south-west Florida, in the case of Helene, south of Tampa Bay. This area allows an investigation of water levels during the Midnight Pass formation (related to my own research).

The extent of the study area is shown in **Figure 1**.

<p align="center">
  <img src="images/dem_extent.png" alt="Screenshot" width="600">
  <br>
  <em>Fig 1:The DEM and extents of the map</em>
</p>


## Mesh

The example will use the **EGOM** mesh. It has a resolution of ~40m alongshore the barrier island, having more than 7000 nodes.

Focus area shown in **Figure 2**.

<p align="center">
  <img src="images/grid.png" alt="Screenshot" width="600">
  <br>
  <em>Fig 2:The HSOFS grid and focus sections</em>
</p>


## DEM

A section of the combined DEM from the Helene XBeach runs (~3km on either side of Helene) is used. It includes a high-resolution DEM and land use map (1m resolution). ADCIRC results from a prior project will also be reused.

DEM details shown in **Figure 3**.

<p align="center">
  <img src="images/extent.png" alt="Screenshot" width="600">
  <br>
  <em>Fig 3: The DEMs to be used within the example</em>
</p>


## Managing File Size

To reduce size, I’ll use only 2km on either side of Helene. DEM and land use map files will be zipped and cloned from the GitHub repo.

## Expected Results

We expect different DEM representations between the histogram and linear distributions. It will be insightful to compare how well each technique captures topological features and generates accurate flood maps.

## Delivery Method

This example will be delivered via a GitHub `.rst`file, follow along similar to the previous GBAY example.
