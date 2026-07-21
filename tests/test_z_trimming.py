import h5py
import numpy as np
import pytest

from ims_to_tiff import get_valid_z_count


def _create_ims(path, channel_data):
    with h5py.File(path, "w") as h5_file:
        dataset = h5_file.create_group("DataSet")
        resolution = dataset.create_group("ResolutionLevel 0")
        time_point = resolution.create_group("TimePoint 0")
        for index, data in enumerate(channel_data):
            channel = time_point.create_group(f"Channel {index}")
            channel.create_dataset("Data", data=np.asarray(data, dtype=np.uint16))


def _valid_z_count(path):
    with h5py.File(path, "r") as h5_file:
        dataset = h5_file["DataSet"]
        channels = sorted(dataset["ResolutionLevel 0"]["TimePoint 0"].keys())
        return get_valid_z_count(
            dataset, "ResolutionLevel 0", "TimePoint 0", channels
        )


def test_keeps_all_planes_when_no_trailing_blanks(tmp_path):
    data = np.ones((3, 4, 5), dtype=np.uint16)
    path = tmp_path / "complete.ims"
    _create_ims(path, [data])

    assert _valid_z_count(path) == 3


def test_trims_only_trailing_blank_planes(tmp_path):
    data = np.ones((4, 4, 5), dtype=np.uint16)
    data[1] = 0
    data[3] = 0
    path = tmp_path / "trailing_blank.ims"
    _create_ims(path, [data])

    assert _valid_z_count(path) == 3


def test_checks_all_channels_before_trimming(tmp_path):
    channel_0 = np.zeros((3, 4, 5), dtype=np.uint16)
    channel_1 = np.zeros((3, 4, 5), dtype=np.uint16)
    channel_0[0] = 1
    channel_1[2] = 1
    path = tmp_path / "multichannel.ims"
    _create_ims(path, [channel_0, channel_1])

    assert _valid_z_count(path) == 3


def test_rejects_an_all_blank_stack(tmp_path):
    data = np.zeros((2, 4, 5), dtype=np.uint16)
    path = tmp_path / "blank.ims"
    _create_ims(path, [data])

    with pytest.raises(ValueError, match="no non-zero image planes"):
        _valid_z_count(path)
