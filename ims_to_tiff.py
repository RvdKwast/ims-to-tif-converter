import glob
import os
import sys

import h5py
import numpy as np
from skimage.transform import pyramid_reduce
from skimage.util import img_as_float, img_as_uint
from tifffile import TiffWriter


def get_bad_frame_index(first_time_point):
    # Now go back through the frames looking for the zeros
    # Find the index of the first zero-frame.
    # Don't know why this bug exists, but it does, so have to deal.
    # Compensate for single z-level stacks that don't need bad frame search.
    if first_time_point.shape[0] == 1:
        return 1
    first_bad_frame_index = first_time_point.shape[0]
    for i_z in range(first_time_point.shape[0])[::-1]:
        if not first_time_point[i_z].any():
            first_bad_frame_index = i_z
    return max(1, first_bad_frame_index)


# Return the resolution levels, time points, channels, z levels, rows, cols, etc.
# from an IMS file. Pass in the opened DataSet group.
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


def convert_to_tif(f_name):
    read_file = h5py.File(f_name, "r")
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

    bad_index_start = get_bad_frame_index(
        np.array(
            base_data[resolution_levels[0]][time_points[0]][channels[0]]["Data"]
        )
    )

    banner_text = "File Breakdown"
    print(banner_text)
    print("_" * len(banner_text))
    print("Channels: %d" % n_channels)
    print("Time Points: %d" % n_time_points)
    print("Z Levels: %d" % bad_index_start)
    print("Native (rows, cols): (%d,%d)" % (n_rows, n_cols))
    print("_" * len(banner_text))

    output_name = os.path.splitext(os.path.basename(f_name))[0] + ".tif"
    with TiffWriter(output_name, imagej=True) as out_tif:
        mmap_fname = f_name + ".mmap"
        output_stack = np.memmap(
            mmap_fname,
            dtype=np.uint16,
            shape=(
                n_time_points,
                bad_index_start,
                n_channels,
                n_rows,
                n_cols,
            ),
            mode="w+",
        )

        for i_t, t in enumerate(time_points):
            print("%s/%d" % (t, n_time_points - 1))
            for i_z, z_lvl in enumerate(z_levels[:bad_index_start]):
                print(
                    "%s/%d Z %d/%d"
                    % (t, n_time_points - 1, i_z + 1, bad_index_start)
                )
                for i_channel, channel in enumerate(channels):
                    output_stack[i_t, i_z, i_channel] = img_as_uint(
                        np.array(
                            base_data[resolution_levels[0]][time_points[i_t]][
                                channels[i_channel]
                            ]["Data"][i_z]
                        )
                    )

        out_tif.write(output_stack, metadata={"axes": "TZCYX"})

        del output_stack
        os.remove(mmap_fname)
    read_file.close()


def downsample_to_tif(f_name, ds_factor=8):
    read_file = h5py.File(f_name, "r")
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
        raise SystemExit("Downsample factor must be >=2")

    test_ds_frame = pyramid_reduce(
        np.array(
            base_data[resolution_levels[0]][time_points[0]][channels[0]]["Data"][0]
        ),
        downscale=ds_factor,
    )
    ds_n_rows, ds_n_cols = test_ds_frame.shape

    bad_index_start = get_bad_frame_index(
        np.array(
            base_data[resolution_levels[0]][time_points[0]][channels[0]]["Data"]
        )
    )

    banner_text = "File Breakdown"
    print(banner_text)
    print("_" * len(banner_text))
    print("Channels: %d" % n_channels)
    print("Time Points: %d" % n_time_points)
    print("Z Levels: %d" % bad_index_start)
    print("Native (rows, cols): (%d,%d)" % (n_rows, n_cols))
    print("Downsampled (rows, cols): (%d,%d)" % (ds_n_rows, ds_n_cols))
    print("_" * len(banner_text))

    f_ending = "_downsampled_%dX.tif" % ds_factor

    with TiffWriter(
        os.path.splitext(os.path.basename(f_name))[0] + f_ending,
        imagej=True,
    ) as out_tif:
        output_stack = np.zeros(
            shape=(
                n_time_points,
                bad_index_start,
                n_channels,
                ds_n_rows,
                ds_n_cols,
            ),
            dtype=np.uint16,
        )

        for i_t, t in enumerate(time_points):
            print("%s/%d" % (t, n_time_points - 1))
            for i_z, z_lvl in enumerate(z_levels[:bad_index_start]):
                print(
                    "%s/%d Z %d/%d"
                    % (t, n_time_points - 1, i_z + 1, bad_index_start)
                )
                for i_channel, channel in enumerate(channels):
                    output_stack[i_t, i_z, i_channel] = img_as_uint(
                        pyramid_reduce(
                            img_as_float(
                                np.array(
                                    base_data[resolution_levels[0]][time_points[i_t]][
                                        channels[i_channel]
                                    ]["Data"][i_z]
                                )
                            ),
                            downscale=ds_factor,
                        )
                    )

        out_tif.write(output_stack, metadata={"axes": "TZCYX"})
        del output_stack
    read_file.close()


def driver(passed_files, ds_factor=1):
    if not bin(ds_factor).count("1") == 1:
        raise SystemExit("Invalid downsample factor. Must be a power of two.")

    if ds_factor > 1:
        downsampled = True
        converter_func = downsample_to_tif
    else:
        downsampled = False
        converter_func = convert_to_tif

    for f_name in passed_files:
        print("")
        print("Processing %s" % f_name)
        print("")

        if downsampled:
            converter_func(f_name, ds_factor)
        else:
            converter_func(f_name)

    print("")
    print("Processed:")
    for f_name in passed_files:
        print(f_name)
    input("Press Enter To Exit")
    exit(0)


def main():
    tif_files = glob.glob("*.tif")

    if len(tif_files) > 0:
        raise SystemExit("Conversion has already been run in this directory. Exiting.")

    ds_factor = int(sys.argv[1])
    ims_files = glob.glob("*.ims")
    cwd = os.getcwd() + "/"
    ims_files = [cwd + f_name for f_name in ims_files]
    driver(ims_files, ds_factor)


if __name__ == "__main__":
    main()
