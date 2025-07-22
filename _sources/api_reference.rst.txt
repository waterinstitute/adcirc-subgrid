API Reference
=============

This section provides reference documentation for the Python API of the ADCIRC Subgrid Preprocessor. While the package is primarily designed for command-line use, the underlying Python classes can be used for custom workflows and advanced applications.

.. note::
   This API is designed for internal use and may change in future versions. For stable interfaces, use the command-line tools described in other sections.

Core Classes
------------

SubgridPreprocessor
~~~~~~~~~~~~~~~~~~~

The main processing class that orchestrates subgrid calculation.

.. py:class:: SubgridPreprocessor(input_file: InputFile, window_memory: int = 64)

   Main class for processing subgrid data.

   :param input_file: Configuration object containing processing parameters
   :type input_file: InputFile
   :param window_memory: Memory limit for raster processing windows (MB)
   :type window_memory: int, optional

   **Methods:**

   .. py:method:: process() -> None

      Execute the main subgrid processing workflow.

      This method:
      - Loads input datasets (mesh, DEM, land cover)
      - Processes each mesh node to compute φ relationships
      - Calculates Manning's roughness corrections
      - Stores results in internal data structures

   .. py:method:: write() -> None

      Write processed subgrid data to NetCDF output file.

      The output file contains:
      - φ factors as functions of water level
      - Average Manning's coefficients
      - Various correction coefficients
      - Mesh coordinate information

   **Example Usage:**

   .. code-block:: python

      from AdcircSubgrid import SubgridPreprocessor, InputFile

      # Create configuration
      config = InputFile('config.yaml')

      # Initialize processor
      processor = SubgridPreprocessor(config, window_memory=128)

      # Process and write results
      processor.process()
      processor.write()

InputFile
~~~~~~~~~

Configuration file parser and validator.

.. py:class:: InputFile(input_file: str)

   Parser for YAML configuration files with schema validation.

   :param input_file: Path to YAML configuration file
   :type input_file: str

   **Methods:**

   .. py:method:: data() -> dict

      Get validated configuration data.

      :return: Dictionary containing configuration parameters
      :rtype: dict

   **Example Usage:**

   .. code-block:: python

      from AdcircSubgrid import InputFile

      # Load and validate configuration
      config = InputFile('my_config.yaml')

      # Access configuration data
      config_data = config.data()
      mesh_file = config_data['input']['adcirc_mesh']
      output_file = config_data['output']['filename']

Mesh
~~~~

ADCIRC mesh file reader and processor.

.. py:class:: Mesh(mesh_file: str)

   Class for reading and processing ADCIRC mesh files (fort.14).

   :param mesh_file: Path to ADCIRC mesh file
   :type mesh_file: str

   **Attributes:**

   .. py:attribute:: nodes

      Number of nodes in the mesh.

      :type: int

   .. py:attribute:: elements

      Number of elements in the mesh.

      :type: int

   .. py:attribute:: x

      Node x-coordinates.

      :type: numpy.ndarray

   .. py:attribute:: y

      Node y-coordinates.

      :type: numpy.ndarray

   .. py:attribute:: depth

      Node bathymetric depths.

      :type: numpy.ndarray

   **Example Usage:**

   .. code-block:: python

      from AdcircSubgrid import Mesh

      # Load mesh
      mesh = Mesh('fort.14')

      # Access mesh properties
      print(f"Mesh has {mesh.nodes} nodes and {mesh.elements} elements")

      # Access coordinates and bathymetry
      coordinates = list(zip(mesh.x, mesh.y))
      depths = mesh.depth

Raster
~~~~~~

Raster data reader and processor for DEM and land cover data.

.. py:class:: Raster(filename: str)

   Generic raster data handler using GDAL.

   :param filename: Path to raster file
   :type filename: str

   **Methods:**

   .. py:method:: get_values_at_points(x: np.ndarray, y: np.ndarray) -> np.ndarray

      Extract raster values at specified coordinate points.

      :param x: X-coordinates of query points
      :type x: numpy.ndarray
      :param y: Y-coordinates of query points
      :type y: numpy.ndarray
      :return: Raster values at query points
      :rtype: numpy.ndarray

   .. py:method:: get_bbox() -> tuple

      Get raster bounding box.

      :return: Bounding box as (minx, miny, maxx, maxy)
      :rtype: tuple

   **Example Usage:**

   .. code-block:: python

      from AdcircSubgrid import Raster
      import numpy as np

      # Load DEM
      dem = Raster('elevation.tif')

      # Sample elevations at specific points
      x_points = np.array([-95.0, -94.5])
      y_points = np.array([29.0, 29.5])
      elevations = dem.get_values_at_points(x_points, y_points)

LookupTable
~~~~~~~~~~~

Manning's roughness coefficient lookup table handler.

.. py:class:: LookupTable

   Class for managing land cover to Manning's roughness mappings.

   **Static Methods:**

   .. py:staticmethod:: ccap_lookup() -> np.ndarray

      Generate default CCAP lookup table.

      :return: Array mapping CCAP class codes to Manning's coefficients
      :rtype: numpy.ndarray

   .. py:staticmethod:: from_file(filename: str) -> np.ndarray

      Load custom lookup table from CSV file.

      :param filename: Path to CSV lookup file
      :type filename: str
      :return: Array mapping class codes to Manning's coefficients
      :rtype: numpy.ndarray

   **Example Usage:**

   .. code-block:: python

      from AdcircSubgrid import LookupTable

      # Use default CCAP lookup
      ccap_table = LookupTable.ccap_lookup()

      # Or load custom lookup
      custom_table = LookupTable.from_file('custom_manning.csv')

      # Get Manning's coefficient for land cover class
      manning_n = ccap_table[class_code]

SubgridData
~~~~~~~~~~~

Container class for computed subgrid correction data.

.. py:class:: SubgridData(n_nodes: int, n_levels: int)

   Storage container for subgrid calculations.

   :param n_nodes: Number of mesh nodes
   :type n_nodes: int
   :param n_levels: Number of phi levels
   :type n_levels: int

   **Attributes:**

   .. py:attribute:: phi

      Phi factors for each node and water level.

      :type: numpy.ndarray
      :shape: (n_nodes, n_levels)

   .. py:attribute:: manning_avg

      Average Manning's coefficients for each node.

      :type: numpy.ndarray
      :shape: (n_nodes,)

   .. py:attribute:: phi_levels

      Water surface elevation levels.

      :type: numpy.ndarray
      :shape: (n_levels,)

SubgridOutputFile
~~~~~~~~~~~~~~~~~

NetCDF output file writer for subgrid data.

.. py:class:: SubgridOutputFile(filename: str, mesh: Mesh, subgrid_data: SubgridData, input_config: dict)

   Writer for subgrid NetCDF output files.

   :param filename: Output filename
   :type filename: str
   :param mesh: Mesh object with coordinate information
   :type mesh: Mesh
   :param subgrid_data: Computed subgrid corrections
   :type subgrid_data: SubgridData
   :param input_config: Configuration dictionary
   :type input_config: dict

   **Methods:**

   .. py:method:: write() -> None

      Write subgrid data to NetCDF file.

      Creates a CF-compliant NetCDF file with:
      - Node coordinates and connectivity
      - Phi factors and water levels
      - Manning's coefficients
      - Metadata and provenance information

Utility Functions
-----------------

Calculation Levels
~~~~~~~~~~~~~~~~~~

Functions for determining water level distributions.

.. py:function:: calculate_levels_linear(z_min: float, z_max: float, n_levels: int) -> np.ndarray

   Generate linearly spaced water levels.

   :param z_min: Minimum water level
   :type z_min: float
   :param z_max: Maximum water level
   :type z_max: float
   :param n_levels: Number of levels
   :type n_levels: int
   :return: Array of water levels
   :rtype: numpy.ndarray

.. py:function:: calculate_levels_histogram(elevations: np.ndarray, n_levels: int) -> np.ndarray

   Generate water levels based on elevation histogram.

   :param elevations: Array of elevation values
   :type elevations: numpy.ndarray
   :param n_levels: Number of levels
   :type n_levels: int
   :return: Array of water levels based on elevation quantiles
   :rtype: numpy.ndarray

JIT-Compiled Helpers
~~~~~~~~~~~~~~~~~~~~

Numba-accelerated functions for performance-critical calculations.

.. py:function:: nan_mean_jit(array: np.ndarray) -> float

   Compute mean of array ignoring NaN values (JIT-compiled).

   :param array: Input array
   :type array: numpy.ndarray
   :return: Mean value excluding NaN
   :rtype: float

.. py:function:: nan_sum_jit(array: np.ndarray) -> float

   Compute sum of array ignoring NaN values (JIT-compiled).

   :param array: Input array
   :type array: numpy.ndarray
   :return: Sum excluding NaN
   :rtype: float

Visualization Functions
-----------------------

The visualization functionality is primarily accessed through command-line tools, but some functions are available for programmatic use:

.. py:function:: plot_mesh(filename: str, variable: str, water_level: float, show: bool = False, output_filename: str = None, **kwargs)

   Create mesh-level visualization of subgrid data.

   :param filename: Subgrid NetCDF filename
   :type filename: str
   :param variable: Variable name to plot
   :type variable: str
   :param water_level: Water level for visualization
   :type water_level: float
   :param show: Display plot interactively
   :type show: bool, optional
   :param output_filename: Save plot to file
   :type output_filename: str, optional

.. py:function:: node_plot(filename: str, node: int, basis: str = 'phi', show: bool = False, save: str = None, index_base: int = 0)

   Create node-specific visualization of phi relationships.

   :param filename: Subgrid NetCDF filename
   :type filename: str
   :param node: Node number to analyze
   :type node: int
   :param basis: Plot basis ('phi' or 'wse')
   :type basis: str, optional
   :param show: Display plot interactively
   :type show: bool, optional
   :param save: Save plot to filename
   :type save: str, optional
   :param index_base: Node numbering base (0 or 1)
   :type index_base: int, optional

Example Custom Workflow
-----------------------

Here's an example of using the API for a custom processing workflow:

.. code-block:: python

   import numpy as np
   from AdcircSubgrid import (
       InputFile, SubgridPreprocessor, Mesh, Raster,
       LookupTable, SubgridData, SubgridOutputFile
   )

   def custom_subgrid_workflow(config_file, custom_processing=False):
       """
       Example custom workflow using the Python API.
       """

       # Load configuration
       config = InputFile(config_file)
       config_data = config.data()

       if custom_processing:
           # Custom processing example

           # Load mesh manually
           mesh = Mesh(config_data['input']['adcirc_mesh'])

           # Load DEM
           dem = Raster(config_data['input']['dem'])

           # Load land cover
           landcover = Raster(config_data['input']['land_cover'])

           # Get lookup table
           if config_data['input']['manning_lookup'] == 'ccap':
               lookup = LookupTable.ccap_lookup()
           else:
               lookup = LookupTable.from_file(config_data['input']['manning_lookup'])

           # Initialize data container
           n_levels = config_data['options']['n_phi_levels']
           subgrid_data = SubgridData(mesh.nodes, n_levels)

           # Custom processing loop (simplified example)
           for i in range(mesh.nodes):
               # Extract elevations around node
               # (This is a simplified example - actual implementation is more complex)
               x, y = mesh.x[i], mesh.y[i]

               # Sample DEM in neighborhood of node
               # ... custom sampling logic ...

               # Compute phi relationships
               # ... custom phi calculation ...

               # Store results
               # subgrid_data.phi[i, :] = computed_phi
               # subgrid_data.manning_avg[i] = computed_manning

           # Write results
           output_writer = SubgridOutputFile(
               config_data['output']['filename'],
               mesh,
               subgrid_data,
               config_data
           )
           output_writer.write()

       else:
           # Standard processing
           processor = SubgridPreprocessor(config)
           processor.process()
           processor.write()

   # Usage
   custom_subgrid_workflow('config.yaml', custom_processing=False)

Error Handling
--------------

The API includes several custom exceptions:

.. py:exception:: SubgridError

   Base exception class for subgrid processing errors.

.. py:exception:: ConfigurationError

   Raised when configuration file validation fails.

.. py:exception:: MeshError

   Raised when mesh file reading or processing fails.

.. py:exception:: RasterError

   Raised when raster data processing fails.

Example error handling:

.. code-block:: python

   from AdcircSubgrid import SubgridPreprocessor, InputFile, ConfigurationError

   try:
       config = InputFile('config.yaml')
       processor = SubgridPreprocessor(config)
       processor.process()
       processor.write()
   except ConfigurationError as e:
       print(f"Configuration error: {e}")
   except Exception as e:
       print(f"Processing error: {e}")

This API reference provides the foundation for advanced users who need to customize or extend the subgrid preprocessing functionality beyond what's available through the command-line interface.
