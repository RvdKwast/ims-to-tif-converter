import argparse
import tempfile
from pathlib import Path

import h5py
import numpy as np
from skimage.transform import pyramid_reduce
from skimage.util import img_as_float, img_as_uint
from tifffile import TiffWriter


def get_valid_z_count(h5_dataset, resolution_level, time_point, channels):
    """Return the number of Z planes through the last non-zero plane.

    Imaris IMS exports can contain trailing all-zero planes. A plane is retained
    when any channel contains data, and all planes before the last populated
    plane are preserved.
    """
    time_point_group = h5_dataset[resolution_level][time_point]
    n_z_levels = time_point_group[channels[0]]["Data"].shape[0]

    for i_z in range(n_z_levels - 1, -1, -1):
        if any(
            np.asarray(time_point_group[channel]["Data"][i_z]).any()
            for channel in channels
        ):
            return i_z + 1

    raise ValueError("The first time point contains no non-zero image planes.")


def get_h5_file_info(h5_dataset):
    resolution_levels = list(h5_dataset)
    resolution_levels.sort(key=lambda x: int(x.split(" ")[-1]))

    time_points = list(h5_dataset[resolution_levels[0]])
    time_points.sort(key=lambda x: int(x.split(" ")[-1]))
    n_time_points = len(time_points)

    channels = list(h5_dataset[resolution_levels[0]][time_points[0]])
    channels.sort(key=lambda x: int(x.split(" ")[-1]))
    n_channels = len(channels)

    n_z_levels = h5_dataset[resolution_levels[0]][time_points[0]][channels[0]][
        "Data"
    ].shape[0]
    z_levels = list(range(n_z_levels))

    n_rows, n_cols = h5_dataset[resolution_levels[0]][time_points[0]][channels[0]][
        "Data"
    ].shape[1:]

    return (
        resolution_levels,
        time_points,
        n_time_points,
        channels,
        n_channels,
        n_z_levels,
        z_levels,
        n_rows,
        n_cols,
    )


def _create_temp_memmap(input_path, shape):
    """Create a disk-backed uint16 array next to the input file."""
    input_path = Path(input_path)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f"{input_path.stem}.",
        suffix=".mmap",
        dir=input_path.parent,
        delete=False,
    )
    mmap_path = Path(temp_file.name)
    temp_file.close()

    try:
        output_stack = np.memmap(
            mmap_path,
            dtype=np.uint16,
            shape=shape,
            mode="w+",
        )
    except Exception:
        mmap_path.unlink(missing_ok=True)
        raise

    return mmap_path, output_stack


def is_power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def get_output_path(input_path, ds_factor):
    input_path = Path(input_path)
    if ds_factor == 1:
        return input_path.with_suffix(".tif")
    return input_path.with_name(f"{input_path.stem}_downsampled_{ds_factor}X.tif")


def convert_to_tif(f_name, output_path=None):
    input_path = Path(f_name)
    output_path = (
        Path(output_path) if output_path is not None else get_output_path(input_path, 1)
    )

    with h5py.File(input_path, "r") as read_file:
        base_data = read_file["DataSet"]

        (
            resolution_levels,
            time_points,
            n_time_points,
            channels,
            n_channels,
            n_z_levels,
            z_levels,
            n_rows,
            n_cols,
        ) = get_h5_file_info(base_data)

        valid_z_count = get_valid_z_count(
            base_data, resolution_levels[0], time_points[0], channels
        )

        banner_text = "File Breakdown"
        print(banner_text)
        print("_" * len(banner_text))
        print("Channels: %d" % n_channels)
        print("Time Points: %d" % n_time_points)
        print("Z Levels: %d" % valid_z_count)
        print("Native (rows, cols): (%d,%d)" % (n_rows, n_cols))
        print("_" * len(banner_text))

        mmap_path, output_stack = _create_temp_memmap(
            input_path,
            (
                n_time_points,
                valid_z_count,
                n_channels,
                n_rows,
                n_cols,
            ),
        )

        try:
            for i_t, t in enumerate(time_points):
                print("%s/%d" % (t, n_time_points - 1))
                for i_z, z_lvl in enumerate(z_levels[:valid_z_count]):
                    print(
                        "%s/%d Z %d/%d"
                        % (t, n_time_points - 1, i_z + 1, valid_z_count)
                    )
                    for i_channel, channel in enumerate(channels):
                        output_stack[i_t, i_z, i_channel] = img_as_uint(
                            np.array(
                                base_data[resolution_levels[0]][time_points[i_t]][
                                    channels[i_channel]
                                ]["Data"][i_z]
                            )
                        )

            output_stack.flush()
            with TiffWriter(output_path, imagej=True) as out_tif:
                out_tif.write(output_stack, metadata={"axes": "TZCYX"})
        finally:
            output_stack._mmap.close()
            mmap_path.unlink(missing_ok=True)

    return output_path


def downsample_to_tif(f_name, ds_factor=8, output_path=None):
    input_path = Path(f_name)
    output_path = (
        Path(output_path)
        if output_path is not None
        else get_output_path(input_path, ds_factor)
    )

    with h5py.File(input_path, "r") as read_file:
        base_data = read_file["DataSet"]

        (
            resolution_levels,
            time_points,
            n_time_points,
            channels,
            n_channels,
            n_z_levels,
            z_levels,
            n_rows,
            n_cols,
        ) = get_h5_file_info(base_data)

        if ds_factor < 2:
            raise ValueError("Downsample factor must be at least 2.")

        test_ds_frame = pyramid_reduce(
            np.array(
                base_data[resolution_levels[0]][time_points[0]][channels[0]][
                    "Data"
                ][0]
            ),
            downscale=ds_factor,
        )
        ds_n_rows, ds_n_cols = test_ds_frame.shape

        valid_z_count = get_valid_z_count(
            base_data, resolution_levels[0], time_points[0], channels
        )

        banner_text = "File Breakdown"
        print(banner_text)
        print("_" * len(banner_text))
        print("Channels: %d" % n_channels)
        print("Time Points: %d" % n_time_points)
        print("Z Levels: %d" % valid_z_count)
        print("Native (rows, cols): (%d,%d)" % (n_rows, n_cols))
        print("Downsampled (rows, cols): (%d,%d)" % (ds_n_rows, ds_n_cols))
        print("_" * len(banner_text))

        mmap_path, output_stack = _create_temp_memmap(
            input_path,
            (
                n_time_points,
                valid_z_count,
                n_channels,
                ds_n_rows,
                ds_n_cols,
            ),
        )

        try:
            for i_t, t in enumerate(time_points):
                print("%s/%d" % (t, n_time_points - 1))
                for i_z, z_lvl in enumerate(z_levels[:valid_z_count]):
                    print(
                        "%s/%d Z %d/%d"
                        % (t, n_time_points - 1, i_z + 1, valid_z_count)
                    )
                    for i_channel, channel in enumerate(channels):
                        output_stack[i_t, i_z, i_channel] = img_as_uint(
                            pyramid_reduce(
                                img_as_float(
                                    np.array(
                                        base_data[resolution_levels[0]][
                                            time_points[i_t]
                                        ][channels[i_channel]]["Data"][i_z]
                                    )
                                ),
                                downscale=ds_factor,
                            )
                        )

            output_stack.flush()
            with TiffWriter(output_path, imagej=True) as out_tif:
                out_tif.write(output_stack, metadata={"axes": "TZCYX"})
        finally:
            output_stack._mmap.close()
            mmap_path.unlink(missing_ok=True)

    return output_path


def driver(passed_files, ds_factor=1, overwrite=False):
    if not is_power_of_two(ds_factor):
        raise SystemExit(
            "Invalid downsample factor. Use a positive power of two: 1, 2, 4, 8, ..."
        )

    input_paths = [Path(path).expanduser().resolve() for path in passed_files]
    if not input_paths:
        raise SystemExit("No IMS files were provided or found in the current directory.")

    invalid_paths = [
        path
        for path in input_paths
        if not path.is_file() or path.suffix.lower() != ".ims"
    ]
    if invalid_paths:
        formatted = "\n".join(f"- {path}" for path in invalid_paths)
        raise SystemExit(f"Invalid IMS input file(s):\n{formatted}")

    jobs = [(input_path, get_output_path(input_path, ds_factor)) for input_path in input_paths]
    existing_outputs = [output_path for _, output_path in jobs if output_path.exists()]
    if existing_outputs and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing_outputs)
        raise SystemExit(
            "Output file(s) already exist. Use --overwrite to replace them:\n"
            f"{formatted}"
        )

    converter_func = downsample_to_tif if ds_factor > 1 else convert_to_tif
    output_paths = []

    for input_path, output_path in jobs:
        print(f"\nProcessing {input_path}\n")
        if ds_factor > 1:
            converter_func(input_path, ds_factor, output_path=output_path)
        else:
            converter_func(input_path, output_path=output_path)
        output_paths.append(output_path)

    print("\nProcessed:")
    for output_path in output_paths:
        print(output_path)

    return output_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert Imaris/Fusion IMS files to ImageJ-compatible TIFF files."
    )
    parser.add_argument(
        "downsample_factor",
        nargs="?",
        type=int,
        default=1,
        help="Positive power-of-two XY downsampling factor (default: 1).",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="IMS files to convert. Defaults to all *.ims files in the current directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output TIFF files that already exist.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ims_files = args.files or sorted(
        path
        for path in Path.cwd().iterdir()
        if path.is_file() and path.suffix.lower() == ".ims"
    )
    driver(ims_files, args.downsample_factor, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
