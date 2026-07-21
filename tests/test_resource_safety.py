import h5py
import numpy as np
import tifffile

from ims_to_tiff import convert_to_tif, downsample_to_tif


def _create_conversion_ims(path):
    data = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4) + 1
    with h5py.File(path, "w") as h5_file:
        dataset = h5_file.create_group("DataSet")
        resolution = dataset.create_group("ResolutionLevel 0")
        time_point = resolution.create_group("TimePoint 0")
        channel = time_point.create_group("Channel 0")
        channel.create_dataset("Data", data=data)


def test_direct_conversion_cleans_up_memmap(tmp_path, monkeypatch):
    input_path = tmp_path / "sample.ims"
    _create_conversion_ims(input_path)
    monkeypatch.chdir(tmp_path)

    convert_to_tif(str(input_path))

    assert (tmp_path / "sample.tif").exists()
    assert not list(tmp_path.glob("*.mmap"))
    with tifffile.TiffFile(tmp_path / "sample.tif") as tif:
        assert tif.series[0].shape[-3:] == (3, 4, 4)


def test_downsample_conversion_uses_disk_backing_and_cleans_up(tmp_path, monkeypatch):
    input_path = tmp_path / "sample.ims"
    _create_conversion_ims(input_path)
    monkeypatch.chdir(tmp_path)

    downsample_to_tif(str(input_path), ds_factor=2)

    assert (tmp_path / "sample_downsampled_2X.tif").exists()
    assert not list(tmp_path.glob("*.mmap"))
    with tifffile.TiffFile(tmp_path / "sample_downsampled_2X.tif") as tif:
        assert tif.series[0].shape[-3:] == (3, 2, 2)
