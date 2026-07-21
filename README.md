# ims-to-tif-converter

Convert Imaris/Fusion `.ims` HDF5 image stacks to ImageJ-compatible TIFF hyperstacks, with optional power-of-two XY downsampling.

## Installation

```bash
python -m pip install -r requirements.txt
```

Runtime dependencies are `h5py`, `numpy`, `scikit-image`, and `tifffile`.

## Usage

Convert every `.ims` file in the current directory without downsampling:

```bash
python ims_to_tiff.py
```

Convert every `.ims` file with 4× XY downsampling:

```bash
python ims_to_tiff.py 4
```

Convert specific files:

```bash
python ims_to_tiff.py 2 sample-a.ims sample-b.ims
```

The downsampling factor must be a positive power of two (`1`, `2`, `4`, `8`, ...). A factor of `1` performs direct conversion.

Outputs are written beside their corresponding inputs:

- `sample.ims` → `sample.tif`
- `sample.ims` with factor `4` → `sample_downsampled_4X.tif`

Existing output files are preserved by default. Replace them explicitly with:

```bash
python ims_to_tiff.py 1 --overwrite
```

The command is non-interactive and returns a non-zero exit status for invalid inputs, missing files, invalid downsampling factors, and output conflicts, making it suitable for scripts and batch jobs.

## Tests

```bash
python -m pip install pytest
pytest -q
```
