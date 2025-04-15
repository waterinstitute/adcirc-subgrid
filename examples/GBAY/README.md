This folder contains 2 raster elevation files and 1 raster landcover file,
and was created to run the subgrid preprocessor code over multiple dataset.

# Step 1:

Run subgrid preprocessor with input.yaml as input. This will use one of the
dem files, landcover file, and mesh file to build a subgrid lookup table.

# Step 2:

Run subgrid preprocessor with input_update_existing.yaml. The updated yaml
contains an extra optional input line called "existing subgrid" where you
add the filepath of the existing subgrid. Running the preprocessor code again
will use the second dem file, landcover file, and mesh file to build and
updated lookup table with subgrid values for the first and second dem included.

## NOTE
  - The code will not overwrite the existing subgrid data, so use the highest priority datasets first.
  - I recommend using a different name for the updated subgrid table to keep track of everything.
