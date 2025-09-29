Norfolk Subgrid Post-Processing and Analysis
============================================

In this section, we examine the outputs from the three preprocessing runs of the `adcirc-subgrid` tool. The goal is to evaluate how **DEM resolution** affects not only runtime, but also the **structure of the generated subgrid data**.

Files Analyzed
--------------

This notebook uses the following files created during preprocessing:

- `subgrid1m.nc` – from the 10-meter DEM
- `subgrid30m.nc` – from the 30-meter DEM
- `subgrid100m.nc` – from the 100-meter DEM

Each file contains detailed subgrid metrics including:

- Mean elevation per cell
- Standard deviation of elevation
- Wet fraction values
- Subgrid volume data

What This Notebook Does
------------------------

- Loads subgrid `.nc` files into memory using `xarray`
- Extracts and compares important grid variables
- Visualizes spatial differences using:
  - Heatmaps
  - Histograms
  - Summary statistics
- Compares runtime performance from preprocessing

By the end of this notebook, you'll have a clearer understanding of how coarse vs. fine DEM resolution affects both **computational performance** and **fidelity of the subgrid output**.

