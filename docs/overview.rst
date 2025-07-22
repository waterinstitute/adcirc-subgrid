Overview
========

What is ADCIRC Subgrid?
------------------------

The ADCIRC Subgrid Preprocessor is a Python package designed to enhance the accuracy of ADCIRC hydrodynamic models by generating subgrid correction terms. These corrections account for sub-mesh scale features that cannot be explicitly resolved in the computational mesh but significantly affect the flow dynamics.

Background and Motivation
--------------------------

Traditional hydrodynamic modeling faces a fundamental trade-off between computational efficiency and spatial resolution. Higher resolution meshes can capture more detailed topographic features but come at significantly increased computational cost. Subgrid methods provide an elegant solution by:

1. **Maintaining computational efficiency** with coarser mesh resolutions
2. **Incorporating fine-scale topographic effects** through correction terms
3. **Improving model accuracy** without proportional increases in computational cost

The subgrid approach is particularly valuable for:

* **Storm surge modeling** where small-scale topographic features affect flooding patterns
* **Coastal applications** with complex bathymetry and varying land cover
* **Large-domain simulations** where uniform high resolution is computationally prohibitive

Key Concepts
------------

Subgrid Correction Terms
~~~~~~~~~~~~~~~~~~~~~~~~

The subgrid method modifies the standard shallow water equations by introducing correction terms that account for:

* **Sub-grid scale bathymetry variations**: Capturing the effect of topographic features smaller than the mesh resolution
* **Manning's roughness heterogeneity**: Accounting for sub-mesh scale variations in bottom friction
* **Wetting and drying dynamics**: Improving the representation of complex shoreline interactions

Mathematical Framework
~~~~~~~~~~~~~~~~~~~~~~

The subgrid corrections are implemented through modification of the continuity and momentum equations:

**Averaged Continuity Equation:**

.. math::

   \phi \frac{\partial \langle \zeta \rangle_W}{\partial t} + \frac{\partial \langle UH \rangle_G}{\partial x} + \frac{\partial \langle VH \rangle_G}{\partial y} = 0

**Averaged Momentum Equations (simplified form):**

.. math::

   \frac{\partial \langle UH \rangle_G}{\partial t} + \frac{\partial C_{UU} \langle U \rangle \langle UH \rangle_G}{\partial x} + \frac{\partial C_{VU} \langle V \rangle \langle UH \rangle_G}{\partial y} = -\frac{C_{M,f} \langle U \rangle \langle UH \rangle_G}{\langle H \rangle_W} + \text{other terms}

Where:
- :math:`\langle \cdot \rangle_G` indicates grid-averaged quantities
- :math:`\langle \cdot \rangle_W` indicates wet-averaged quantities
- :math:`C_{UU}, C_{VU}` are advection correction coefficients
- :math:`C_{M,f}` is the bottom friction correction coefficient
- :math:`\phi` is the wet area fraction (0 ≤ φ ≤ 1)

The φ Factor
~~~~~~~~~~~~

The central concept in subgrid modeling is the φ (phi) factor, which represents the fraction of a mesh element that is wet at a given water level. This factor:

* Varies from 0 (completely dry) to 1 (completely wet)
* Is computed based on high-resolution topographic data within each mesh element
* Accounts for complex sub-mesh topography and wetting patterns

Data Requirements
-----------------

The subgrid preprocessor requires several input datasets:

**Essential Inputs:**

1. **ADCIRC Mesh File** (``fort.14``): Defines the computational grid and bathymetry
2. **Digital Elevation Model (DEM)**: High-resolution topographic/bathymetric data
3. **Land Cover Data**: Classification data for Manning's roughness assignment
4. **Manning's Roughness Lookup**: Mapping from land cover classes to roughness values

**Data Quality Considerations:**

* **Coordinate System Consistency**: All datasets should use consistent coordinate reference systems (WGS84 recommended)
* **Resolution Requirements**: DEM resolution should be significantly higher than mesh resolution for effective subgrid corrections
* **Coverage**: Input datasets should completely cover the computational domain

Workflow Overview
-----------------

The typical subgrid generation workflow consists of:

1. **Data Preparation**: Ensuring all input datasets are properly formatted and georeferenced
2. **Configuration**: Creating YAML configuration files specifying inputs and processing options
3. **Processing**: Running the preprocessor to compute subgrid corrections
4. **Quality Control**: Using visualization tools to verify results
5. **Model Integration**: Incorporating the subgrid file into ADCIRC simulations

Advantages and Limitations
--------------------------

**Advantages:**

* Improved accuracy for flow over complex topography
* Computational efficiency compared to high-resolution meshes
* Flexible configuration options for various applications
* Integration with existing ADCIRC workflows

**Limitations:**

* Requires high-quality input data (DEM and land cover)
* Additional preprocessing step in the modeling workflow
* May not capture all physical processes affected by small-scale features
* Effectiveness depends on the quality of underlying assumptions about sub-grid processes

Applications
------------

The ADCIRC Subgrid Preprocessor is particularly well-suited for:

* **Hurricane storm surge modeling** in areas with complex coastal topography
* **Flood inundation studies** requiring accurate representation of urban infrastructure
* **Coastal restoration project assessment** where small-scale features affect flow patterns
* **Climate change impact studies** requiring efficient large-domain simulations

This overview provides the foundation for understanding how the subgrid preprocessor works and when to apply it effectively. The following sections provide detailed guidance on installation, configuration, and usage.
