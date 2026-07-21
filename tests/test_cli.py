import h5py
import numpy as np
import pytest

from ims_to_tiff import driver, get_output_path, is_power_of_two, main


def _create_ims(path):
    data = np.ones((2, 4, 4), dtype=np.uint16)
    with h5py.File(path, "w") as h5_file:
        dataset = h5_file.create_group("DataSet")
        resolution = dataset.create_group("ResolutionLevel 0")
        time_point = resolution.create_group("TimePoint 0")
        channel = time_point.create_group("Channel 0")
        channel.create_dataset("Data", data=data)


@pytest.mark.parametrize("value", [1, 2, 4, 8])
def test_accepts_positive_power_of_two(value):
    assert is_power_of_two(value)


@pytest.mark.parametrize("value", [-2, -1, 0, 3, 6])
def test_rejects_invalid_downsample_factor(value):
    assert not is_power_of_two(value)


def test_output_path_is_adjacent_to_input(tmp_path):
    input_path = tmp_path / "example.ims"

    assert get_output_path(input_path, 1) == tmp_path / "example.tif"
    assert get_output_path(input_path, 4) == tmp_path / "example_downsampled_4X.tif"


def test_driver_refuses_exact_existing_output(tmp_path):
    input_path = tmp_path / "sample.ims"
    output_path = tmp_path / "sample.tif"
    _create_ims(input_path)
    output_path.write_bytes(b"existing")

    with pytest.raises(SystemExit, match="--overwrite"):
        driver([input_path], ds_factor=1)


def test_main_runs_without_interactive_prompt(tmp_path, monkeypatch):
    input_path = tmp_path / "sample.ims"
    _create_ims(input_path)
    monkeypatch.chdir(tmp_path)

    main(["1"])

    assert (tmp_path / "sample.tif").exists()


def test_main_reports_empty_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="No IMS files"):
        main(["1"])
