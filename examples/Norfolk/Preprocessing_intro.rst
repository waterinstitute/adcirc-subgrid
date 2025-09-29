Norfolk Subgrid Preprocessing
=============================

This section walks through the **preprocessing stage** of the Norfolk ADCIRC-Subgrid example. The focus here is to explore how **Digital Elevation Model (DEM) resolution** affects the **runtime performance** of the `adcirc-subgrid` tool.

In this experiment, we run the subgrid preprocessor using three different DEMs representing the Norfolk, VA region:

- **10-meter DEM** — high-resolution terrain
- **30-meter DEM** — medium-resolution
- **100-meter DEM** — coarse-resolution

Each configuration is set up using a YAML input file that points to:

- A shared **ADCIRC mesh** (`HSOFS_fort.14`)
- A **NOAA CCAP land cover dataset**
- A distinct **DEM file** for each resolution

.. note::

   The `adcirc-subgrid` command-line tool must be installed in your environment, and the notebook must be run from within that environment.

What This Notebook Does
-----------------------

- Activates the correct Python environment
- Runs `adcirc-subgrid` three times using different DEMs
- Records the **runtime** of each operation
- Prepares the data for downstream analysis

Next Steps
----------

Continue to the `Process_Norfolk.ipynb <Process_Norfolk.ipynb>`_ notebook to:

- Analyze runtime trends
- Visualize subgrid output differences
- Compare elevation statistics and wet fractions
