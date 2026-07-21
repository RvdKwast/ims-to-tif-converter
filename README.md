# ims-to-tif-converter

Convert Imaris/Fusion `.ims` HDF5 image stacks to ImageJ-compatible TIFF
hyperstacks, with optional power-of-two XY downsampling.

## Installation

Create a Python environment and install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

The converter depends on `h5py`, `numpy`, `scikit-image`, and the standalone
`tifffile` package.

## Usage

Run the script from a directory containing the `.ims` files to convert:

```bash
python /full/path/to/ims_to_tiff.py <downsample_factor>
```

Use `1` for a direct conversion. Use a positive power of two such as `2`, `4`,
or `8` to reduce the X and Y dimensions by that factor.

The current command scans the working directory for `*.ims` files and exits if
the directory already contains a `.tif` file.
