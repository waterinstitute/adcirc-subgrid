Scientific Background
=====================

This section provides a comprehensive explanation of the scientific principles, mathematical foundations, and computational algorithms underlying the ADCIRC subgrid method.

Theoretical Foundation
----------------------

Historical Development
~~~~~~~~~~~~~~~~~~~~~~

Subgrid corrections in hydrodynamic modeling evolved from the need to represent high-resolution topographic features on computationally efficient coarse meshes:

* **Early Development**: Defina (2000) introduced subgrid corrections for partially wet computational cells and advection enhancements in tidal models
* **Finite Volume Extensions**: Casulli (2009) improved mass balance and wetting/drying in unstructured finite volume models
* **Bottom Friction Improvements**: Volp et al. (2013) incorporated spatial variability in bottom roughness and depth
* **Formalization**: Kennedy et al. (2019) formalized closure approximations and added water surface gradient corrections
* **Storm Surge Integration**: First comprehensive implementation in hurricane storm surge models (ADCIRC) achieved 10-50× computational speedup while improving accuracy

Scale Separation Problem
~~~~~~~~~~~~~~~~~~~~~~~~

The fundamental challenge in coastal hydrodynamic modeling is the **scale separation** between:

* **Model scale**: Tens to hundreds of meters (mesh resolution)
* **Subgrid scale**: 1 meter or smaller (DEM resolution)
* **Feature scale**: Critical topographic features (channels, barriers, dunes)

**Computational Trade-offs:**

* High-resolution uniform meshes: Accurate but computationally prohibitive
* Coarse meshes: Fast but lose critical flow pathways and barriers
* **Subgrid solution**: Use coarse computational mesh with fine-scale topographic corrections

Subgrid-Scale Modeling Principles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subgrid modeling addresses scale separation by:

1. **Acknowledging sub-mesh heterogeneity**: Each computational cell contains unresolved topographic complexity
2. **Parameterizing sub-mesh effects**: Correction terms capture aggregate impacts of unresolved features
3. **Preserving computational efficiency**: Avoids prohibitive computational costs of uniform high resolution
4. **Improving accuracy-efficiency trade-offs**: Achieves better accuracy on coarse meshes than conventional fine meshes

**Performance Characteristics:**

* **Typical coarsening factor**: 3-20× (up to 50× in some applications)
* **Computational savings**: 1-2 orders of magnitude reduction in runtime
* **Accuracy improvement**: Subgrid coarse models often outperform conventional fine models

The Shallow Water Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard 2D shallow water equations form the basis for ADCIRC:

**Continuity Equation:**

.. math::

   \frac{\partial \zeta}{\partial t} + \frac{\partial (uH)}{\partial x} + \frac{\partial (vH)}{\partial y} = 0

**Momentum Equations:**

.. math::

   \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} - fv = -g\frac{\partial \zeta}{\partial x} - \frac{\tau_{bx}}{\rho H}

.. math::

   \frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + fu = -g\frac{\partial \zeta}{\partial y} - \frac{\tau_{by}}{\rho H}

Where:

* :math:`\zeta` = water surface elevation
* :math:`u, v` = depth-averaged velocities
* :math:`H = h + \zeta` = total water depth
* :math:`h` = bathymetric depth
* :math:`f` = Coriolis parameter
* :math:`g` = gravitational acceleration
* :math:`\tau_b` = bottom stress
* :math:`\rho` = water density

Subgrid-Modified Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~

The subgrid method uses averaged versions of the shallow water equations with closure terms.

**Averaged Continuity Equation:**

.. math::

   \phi \frac{\partial \langle \zeta \rangle_W}{\partial t} + \frac{\partial \langle UH \rangle_G}{\partial x} + \frac{\partial \langle VH \rangle_G}{\partial y} = 0

**Averaged Conservative Momentum Equations:**

**x-direction:**

.. math::

   \frac{\partial \langle UH \rangle_G}{\partial t} + \frac{\partial C_{UU} \langle U \rangle \langle UH \rangle_G}{\partial x} + \frac{\partial C_{VU} \langle V \rangle \langle UH \rangle_G}{\partial y}

.. math::

   + gC_\zeta \langle H \rangle_G \frac{\partial \langle \zeta \rangle_W}{\partial x} = -\frac{C_{M,f} \langle U \rangle \langle UH \rangle_G}{\langle H \rangle_W} - f \langle VH \rangle_G

.. math::

   - g \langle H \rangle_G \frac{\partial P_A}{\partial x} + \frac{\tau_{sx}}{\rho_0 \phi} + \frac{\partial}{\partial x}\left(E_h \frac{\partial \langle UH \rangle_G}{\partial x}\right) + \frac{\partial}{\partial y}\left(E_h \frac{\partial \langle UH \rangle_G}{\partial y}\right)

**y-direction:**

.. math::

   \frac{\partial \langle VH \rangle_G}{\partial t} + \frac{\partial C_{UV} \langle U \rangle \langle VH \rangle_G}{\partial x} + \frac{\partial C_{VV} \langle V \rangle \langle VH \rangle_G}{\partial y}

.. math::

   + gC_\zeta \langle H \rangle_G \frac{\partial \langle \zeta \rangle_W}{\partial y} = -\frac{C_{M,f} \langle V \rangle \langle VH \rangle_G}{\langle H \rangle_W} + f \langle UH \rangle_G

.. math::

   - g \langle H \rangle_G \frac{\partial P_A}{\partial y} + \frac{\tau_{sy}}{\rho_0 \phi} + \frac{\partial}{\partial x}\left(E_h \frac{\partial \langle VH \rangle_G}{\partial x}\right) + \frac{\partial}{\partial y}\left(E_h \frac{\partial \langle VH \rangle_G}{\partial y}\right)

Where:

* :math:`\langle \cdot \rangle` denotes averaged variables
* Subscripts :math:`W` and :math:`G` indicate wet-averaging and grid-averaging respectively
* :math:`U, V` are depth-averaged velocities in x and y directions
* :math:`H` is total water depth
* :math:`\zeta` is water surface elevation
* :math:`\phi` is the wet area fraction (0 ≤ φ ≤ 1)
* :math:`P_A` is atmospheric pressure
* :math:`\tau_{sx}, \tau_{sy}` are surface wind stresses
* :math:`\rho_0` is reference water density
* :math:`f` is Coriolis parameter
* :math:`E_h` is horizontal eddy viscosity
* :math:`g` is gravitational acceleration

**Closure Coefficients:**

The subgrid method introduces five key closure coefficients:

1. :math:`C_\zeta` - Surface gradient correction (typically set to 1.0)
2. :math:`C_{UU}, C_{VU}` - Advection correction terms (typically 1.06-1.10)
3. :math:`\phi` - Wet area fraction (0 ≤ φ ≤ 1)
4. :math:`C_{M,f}` - Bottom friction coefficient correction

**Closure Approximation Levels:**

**Level 0 Closure (Basic Implementation):**

* All advection corrections set to unity: :math:`C_{UU} = C_{VU} = C_{UV} = C_{VV} = 1`
* Bottom friction: :math:`C_{M,f} = gn^2/H^{1/3}` (conventional Manning's formula)
* Surface gradient: :math:`C_\zeta = 1`

**Level 1 Closure (Enhanced Implementation):**

* **Bottom friction correction:**

.. math::

   C_{M,f} = \langle H \rangle_W R_v^2 \quad \text{where} \quad R_v = \frac{\langle H \rangle_W}{\langle H^{3/2}C_f^{-1/2} \rangle_W}

* **Advection corrections:**

.. math::

   C_{UU} = \frac{\langle U^2H \rangle_G}{\langle U \rangle \langle UH \rangle_G}

Level 1 corrections provide significant accuracy improvements, particularly in shallow regions with complex topography and varying friction.

The Phi (φ) Factor
-------------------

Physical Interpretation
~~~~~~~~~~~~~~~~~~~~~~~

The φ factor represents the **wet area fraction** of a computational element or vertex at a given water surface elevation. This concept encapsulates several important physical processes:

1. **Partial Wetting**: As water levels rise, different portions of a mesh element become submerged at different rates
2. **Complex Topography**: Sub-mesh topographic variability controls the wetting pattern
3. **Flow Obstruction**: Dry areas within an element impede flow, effectively reducing the wet cross-sectional area
4. **Discrete Values**: φ ranges from 0 (fully dry) to 1 (fully wet) in discrete increments

**Vertex-Averaged Approach:**

Current implementations use **vertex-averaged areas** rather than element-averaged areas. This approach:

* Aligns more closely with conventional ADCIRC's numerical scheme
* Reduces lookup table size by approximately factor of 4
* Improves computational performance and memory usage

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

For a computational element with area :math:`A_{total}`, the φ factor is:

.. math::

   \phi(\zeta) = \frac{A_{wet}(\zeta)}{A_{total}}

Where :math:`A_{wet}(\zeta)` is the area that is wet (submerged) when the water surface elevation is :math:`\zeta`.

Key properties of φ:

* **Bounds**: :math:`0 \leq \phi \leq 1`
* **Monotonic**: :math:`\phi` increases monotonically with increasing :math:`\zeta`
* **Boundary Conditions**:

  * :math:`\phi \to 0` as :math:`\zeta \to -\infty` (completely dry)
  * :math:`\phi \to 1` as :math:`\zeta \to +\infty` (completely wet)

Computational Algorithm
~~~~~~~~~~~~~~~~~~~~~~~

The φ factor is computed using high-resolution digital elevation models (DEMs):

**Step 1: Element Elevation Sampling**

For each computational mesh element, extract all DEM pixels that fall within the element boundaries:

.. math::

   E = \{z_i : (x_i, y_i) \in \text{element}, i = 1, 2, ..., n\}

**Step 2: Water Level Discretization**

Create a series of water surface elevations spanning the range of interest:

.. math::

   \zeta_j = \zeta_{min} + j \cdot \Delta\zeta, \quad j = 0, 1, ..., N_{levels}

**Step 3: Wet Fraction Calculation**

For each water level :math:`\zeta_j`, compute the fraction of pixels that are submerged:

.. math::

   \phi_j = \frac{1}{n} \sum_{i=1}^{n} H(\zeta_j - z_i)

Where :math:`H(\cdot)` is the Heaviside step function:

.. math::

   H(x) = \begin{cases}
   1 & \text{if } x > 0 \\
   0 & \text{if } x \leq 0
   \end{cases}

Bottom Friction Modifications
-----------------------------

Manning's Roughness Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subgrid corrections also account for sub-mesh variability in bottom friction. The effective Manning's coefficient is computed as an area-weighted average:

.. math::

   n_{eff} = \frac{\sum_{i=1}^{n} n_i A_i}{\sum_{i=1}^{n} A_i}

Where :math:`n_i` is the Manning's coefficient for land cover class :math:`i`, and :math:`A_i` is the area covered by that class within the element.

Friction Coefficient Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The friction coefficient :math:`C_f` is related to Manning's coefficient:

.. math::

   C_f = \frac{g n_{eff}^2}{H^{1/3}}

In the subgrid formulation, this becomes:

.. math::

   C_f = \frac{g n_{eff}^2}{(\phi H)^{1/3}}

This modification accounts for the reduced effective depth due to partial wetting.

Water Level Distribution Methods
---------------------------------

Linear Distribution
~~~~~~~~~~~~~~~~~~~

The simplest approach distributes water levels evenly across a specified range:

.. math::

   \zeta_j = \zeta_{min} + \frac{j}{N-1}(\zeta_{max} - \zeta_{min})

**Advantages:**
- Simple implementation
- Uniform coverage of water level range

**Disadvantages:**
- May not capture critical transitions
- Inefficient for topographically diverse areas

Histogram-Based Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A more sophisticated approach bases water level distribution on the elevation histogram within each element:

**Step 1: Elevation Histogram**

Compute the histogram of elevations within each element:

.. math::

   H(z_k) = \text{number of DEM pixels with elevation } z_k

**Step 2: Cumulative Distribution**

Calculate the cumulative distribution function:

.. math::

   F(z) = \frac{\sum_{z_i \leq z} H(z_i)}{\sum_{i} H(z_i)}

**Step 3: Quantile-Based Levels**

Distribute water levels based on elevation quantiles:

.. math::

   \zeta_j = F^{-1}\left(\frac{j}{N-1}\right)

**Advantages:**
- Adaptive to local topography
- Focuses resolution on critical elevation ranges
- More efficient representation

Lookup Table Optimization
-------------------------

**Table Size Reduction:**

Modern subgrid implementations optimize lookup table sizes for computational efficiency:

**Original Table Sizes:**

.. math::

   N_\zeta \times N_{VE} \times N_E \quad \text{(element-averaged)}

.. math::

   N_\zeta \times N_V \quad \text{(vertex-averaged)}

**Optimized Table Size:**

.. math::

   N_\phi \times N_V

Where:

* :math:`N_\zeta` = number of water surface elevations (e.g., 401 for -20m to +20m at 0.1m increments)
* :math:`N_{VE}` = number of sub-elements per vertex
* :math:`N_E` = number of elements in mesh
* :math:`N_V` = number of vertices in mesh
* :math:`N_\phi` = number of φ values (typically 11)

**Key Improvements:**

1. **Elimination of element-averaged tables** - reduces memory requirements
2. **Discrete φ values** - :math:`N_\phi = 11` instead of :math:`N_\zeta = 401`
3. **Memory reduction factor** - approximately 160× for high-resolution meshes

Implementation Details
----------------------

Wetting and Drying Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Updated Approach:**

Modern implementations use **total water depth criteria** instead of minimum wet area fraction:

.. math::

   \langle H \rangle_G > \langle H \rangle_{G,min}

Where :math:`\langle H \rangle_{G,min} = 0.1` m (typical threshold).

**Benefits:**

* Improved numerical stability
* Avoids extremely small water depths that cause computational instabilities
* Uses robust conventional ADCIRC wetting/drying scheme
* For triangular elements: all 3 vertices must meet minimum criteria

Numerical Algorithms
~~~~~~~~~~~~~~~~~~~~

**Raster Processing Pipeline:**

1. **Mesh Element Identification**: For each mesh node, identify the surrounding element(s)
2. **DEM Intersection**: Extract DEM pixels within element boundaries using spatial indexing
3. **Land Cover Assignment**: Map land cover classes to Manning's coefficients
4. **Statistical Computation**: Calculate elevation statistics and φ relationships
5. **Lookup Table Generation**: Create optimized φ-based lookup tables

**Memory Management:**

The algorithm processes large raster datasets efficiently using:

- **Windowed Reading**: Process raster data in memory-efficient chunks
- **Optimized Lookup Tables**: Reduced memory footprint through φ-based indexing
- **Spatial Indexing**: Use R-tree or similar structures for fast spatial queries
- **Caching**: Cache frequently accessed data to minimize I/O

**Parallel Processing:**

Subgrid calculations are inherently parallel:

- **Vertex Independence**: Each mesh vertex can be processed independently
- **Thread Safety**: Algorithms designed for multi-threaded execution
- **Load Balancing**: Distribute computational load based on vertex complexity

Quality Control and Validation
------------------------------

Physical Consistency Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Monotonicity Verification:**

Ensure that φ increases monotonically with water level:

.. math::

   \phi(\zeta_{j+1}) \geq \phi(\zeta_j) \quad \forall j

**Boundary Condition Verification:**

Check that φ approaches appropriate limits:

.. math::

   \lim_{\zeta \to \zeta_{min}} \phi(\zeta) = 0

.. math::

   \lim_{\zeta \to \zeta_{max}} \phi(\zeta) = 1

Conservation Properties
~~~~~~~~~~~~~~~~~~~~~~~~

**Volume Conservation:**

The subgrid method should preserve volume conservation in the discrete system:

.. math::

   \frac{d}{dt} \int_{\Omega} \zeta \, dA + \oint_{\partial\Omega} \phi uH \cdot \mathbf{n} \, dl = 0

**Mass Conservation:**

Verify that the modified equations maintain mass conservation properties of the original system.

Error Analysis
--------------

Sources of Error
~~~~~~~~~~~~~~~~

1. **Discretization Error**: Finite resolution of DEM and computational mesh
2. **Averaging Error**: Spatial averaging within computational elements
3. **Interpolation Error**: Temporal interpolation of φ values during simulation
4. **Data Quality Error**: Uncertainties in input DEM and land cover data

Error Quantification
~~~~~~~~~~~~~~~~~~~~

**Convergence Analysis:**

Study convergence behavior as:
- DEM resolution increases
- Number of φ levels increases
- Computational mesh resolution changes

**Validation Against High-Resolution Results:**

Compare subgrid results with high-resolution simulations:

.. math::

   \text{Error} = \frac{|\zeta_{subgrid} - \zeta_{high-res}|}{\zeta_{high-res}}

**Sensitivity Analysis:**

Evaluate sensitivity to:
- Manning's coefficient variations
- DEM uncertainty
- φ level distribution choices

Performance Characteristics
---------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

**Preprocessing Cost:**

- **Time Complexity**: O(N_elements × N_pixels × N_levels)
- **Space Complexity**: O(N_elements × N_levels)

Where:
- N_elements = number of mesh elements
- N_pixels = average DEM pixels per element
- N_levels = number of φ levels

**Runtime Cost in ADCIRC:**

- **Additional Memory**: ~20-30% increase for subgrid arrays
- **Computational Overhead**: ~10-20% increase in runtime
- **I/O Overhead**: Initial reading of subgrid file

Scaling Properties
~~~~~~~~~~~~~~~~~~

**Mesh Resolution Scaling:**

Subgrid benefit increases with decreasing mesh resolution:

.. math::

   \text{Benefit} \propto \left(\frac{\text{DEM resolution}}{\text{Mesh resolution}}\right)^2

**Domain Size Scaling:**

Processing time scales approximately linearly with domain size for fixed resolution ratios.

Applications and Limitations
----------------------------

Optimal Application Domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subgrid corrections are most effective for:

1. **Coastal Storm Surge**: Complex nearshore bathymetry and topography
2. **Real-time Forecasting**: Requiring rapid computation with maximum accuracy
3. **Urban Flooding**: Built environment with significant sub-mesh features
4. **Wetland Modeling**: Areas with complex channel and vegetation patterns
5. **Large-Scale Simulations**: Where uniform high resolution is computationally prohibitive
6. **Hurricane Storm Surge**: Combining operational time constraints with accuracy requirements

**Validated Performance Ranges:**

* **Coarsening factors**: 3-20× (up to 50× in some applications)
* **Minimum nearshore resolution**: 60-400m (depending on coastal complexity)
* **Domain scales**: From regional bays (O(10 km)) to ocean basins (O(1000+ km))
* **Computational speedup**: 10-50× compared to equivalent-accuracy fine meshes

Resolution Guidelines and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical Resolution Thresholds:**

1. **Barrier Island Aliasing**: Primary limitation occurs when mesh resolution becomes coarser than critical flow-blocking features
2. **Recommended minimum**: Resolve primary barrier islands and major flow channels in underlying DEM
3. **Practical limits**: Beyond 400-1000m nearshore resolution, important coastal features become aliased
4. **Connectivity issues**: Over-connectivity in protected areas when barriers are under-resolved

**Physical Process Limitations:**

1. **Unidirectional Approximation**: Assumes φ depends only on local water level
2. **Flow Direction Independence**: Does not account for flow direction effects on wetting patterns
3. **Static Topography**: Does not account for morphodynamic changes
4. **Simplified Wetting/Drying**: May not capture complex hysteresis effects
5. **Sub-element Assumptions**: Requires all three sub-elements to be wet for proper channel flow representation

Future Research Directions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Potential improvements include:

1. **Dynamic φ Factors**: Account for flow-dependent wetting patterns
2. **Momentum Subgrid Terms**: Develop correction terms for advection and viscosity
3. **Morphodynamic Coupling**: Integrate with sediment transport and bed evolution
4. **Machine Learning Enhancement**: Use ML techniques for improved parameterizations

This scientific foundation provides the theoretical understanding necessary for effective application of subgrid methods in ADCIRC modeling.
