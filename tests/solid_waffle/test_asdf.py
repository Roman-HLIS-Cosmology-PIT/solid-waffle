import os
from contextlib import chdir

import asdf
import numpy as np
import yaml
from astropy.io import fits
from solid_waffle import asdf_to_fits


def test_tvac(tmp_path):
    """Test the TVAC format. This makes ASDF files with only the fields needed for this test."""

    os.makedirs(tmp_path / "IN")

    _s = np.linspace(0, 4095, 4096).astype(np.float32)
    x, y = np.meshgrid(_s, _s)
    datacubes = []
    for ifile in range(3):
        data = np.zeros((5, 4096, 4224), dtype=np.uint16)
        data[0, :, :4096] = np.round(
            4500 + 200 * np.cos(0.01 * x) + (0.01 + 0.003 * ifile) * np.hypot(x - 1000, y - 3500)
        )
        for j in range(1, 5):
            data[j, :, :4096] = data[0, :, :4096] + np.round(j * (80 + 2 * ifile + x / 100 + y / 300))
        for j in range(5):
            data[j, :, -128:] = np.round(18100 - 100 * np.exp(-0.03 * (y[:, 0] % 256))).astype(np.uint16)[
                :, None
            ]
        for j in range(5):
            print(
                np.amin(data[j, :, :4096]),
                np.amax(data[j, :, :4096]),
                np.amin(data[j, :, -128:]),
                np.amax(data[j, :, -128:]),
            )
        datacubes.append(data)

        # save in a file
        asdf.AsdfFile(
            {
                "roman": {
                    "meta": {"exposure": {"frame_time": 3.15}},
                    "data": data[1:, :, :4096],
                    "reference_read": data[None, 1, :, :4096],
                    "reset_reads": data[None, 0, :, :4096],
                    "amp33": data[1:, :, -128:],
                    "amp33_reference_read": data[None, 1, :, -128:],
                    "amp33_reset_reads": data[None, 0, :, -128:],
                }
            }
        ).write_to(str(tmp_path) + f"/IN/im{ifile+1:02d}.asdf")

    asdf_to_fits.main(input_dir=str(tmp_path) + "/IN", output_dir=str(tmp_path) + "/OUT", format="wfi_tvac")

    # check that the outputs exist
    for ifile in range(3):
        ofile = str(tmp_path) + f"/OUT/im{ifile+1:02d}_asdf_to.fits"
        with fits.open(ofile) as f:
            assert np.all(f[0].data == datacubes[ifile][1:, :, :])
            assert 3.149 < f[0].header["TGROUP"] < 3.151
            assert 3.149 < f[0].header["TFRAME"] < 3.151

        os.remove(ofile)

    asdf_to_fits.main(
        input_dir=str(tmp_path) + "/IN", output_dir=str(tmp_path) + "/OUT", format="wfi_tvac_rst"
    )

    # check that the outputs exist
    for ifile in range(3):
        ofile = str(tmp_path) + f"/OUT/im{ifile+1:02d}_asdf_to.fits"
        with fits.open(ofile) as f:
            assert np.all(f[0].data == datacubes[ifile])
            assert 3.149 < f[0].header["TGROUP"] < 3.151
            assert 3.149 < f[0].header["TFRAME"] < 3.151

            assert f["CONFIG"].header["ORIGFILE"] == f"im{ifile+1:02d}.asdf"
            ydict = yaml.safe_load("".join(f["CONFIG"].data["config"]))
            assert ydict["roman"]["data"]["shape"] == [4, 4096, 4096]

            print(f["CONFIG"].header)
            print(yaml.safe_load("".join(f["CONFIG"].data["config"])))

        os.remove(ofile)


def test_flight(tmp_path):
    """Test the flight engineering format. This makes ASDF files with only the fields needed for this test."""

    data_encoding_offset = 4000

    os.makedirs(tmp_path / "IN")

    _s = np.linspace(0, 4095, 4096).astype(np.float32)
    x, y = np.meshgrid(_s, _s)
    datacubes = []
    for ifile in range(3):
        data = np.zeros((5, 4096, 4224), dtype=np.uint16)
        data[0, :, :4096] = np.round(
            4500 + 200 * np.cos(0.01 * x) + (0.01 + 0.003 * ifile) * np.hypot(x - 1000, y - 3500)
        )
        for j in range(1, 5):
            data[j, :, :4096] = data[0, :, :4096] + np.round(j * (80 + 2 * ifile + x / 100 + y / 300))
        for j in range(5):
            data[j, :, -128:] = np.round(18100 - 100 * np.exp(-0.03 * (y[:, 0] % 256))).astype(np.uint16)[
                :, None
            ]
        for j in range(5):
            print(
                np.amin(data[j, :, :4096]),
                np.amax(data[j, :, :4096]),
                np.amin(data[j, :, -128:]),
                np.amax(data[j, :, -128:]),
            )
        datacubes.append(data)

        _offset = data[0].astype(np.int32) - data_encoding_offset
        subdata = np.zeros_like(data[1:])
        for k in range(np.shape(subdata)[0]):
            subdata[k] = np.clip(
                data[k + 1, :, :].astype(np.int32) - _offset, 0, 65535
            )  # this test won't overflow but just in case

        # save in a file
        asdf.AsdfFile(
            {
                "roman": {
                    "meta": {
                        "exposure": {"frame_time": 3.15},
                        "instrument": {"data_encoding_offset": data_encoding_offset},
                    },
                    "data": subdata[:, :, :4096],
                    "reference_read": data[0, :, :4096],
                    "amp33": subdata[:, :, -128:],
                    "reference_amp33": data[0, :, -128:],
                }
            }
        ).write_to(str(tmp_path) + f"/IN/im{ifile+1:02d}.asdf")

    asdf_to_fits.main(input_dir=str(tmp_path) + "/IN", output_dir=str(tmp_path) + "/OUT", format="flight_eng")

    # check that the outputs exist
    for ifile in range(3):
        ofile = str(tmp_path) + f"/OUT/im{ifile+1:02d}_asdf_to.fits"
        with fits.open(ofile) as f:
            assert np.all(f[0].data == datacubes[ifile])
            assert 3.149 < f[0].header["TGROUP"] < 3.151
            assert 3.149 < f[0].header["TFRAME"] < 3.151

        os.remove(ofile)

    # test glob functionality
    asdf_to_fits.main(
        input_dir=str(tmp_path) + "/IN",
        output_dir=str(tmp_path) + "/OUT",
        fmatch="im?1.asdf",
        format="flight_eng",
    )
    for ifile in range(3):
        ofile = str(tmp_path) + f"/OUT/im{ifile+1:02d}_asdf_to.fits"
        if ifile + 1 == 1:
            os.remove(ofile)
        else:
            assert not os.path.exists(ofile)

    # another test for glob functionality
    asdf_to_fits.main(
        input_dir=str(tmp_path) + "/IN",
        output_dir=str(tmp_path) + "/OUT",
        fmatch="im0[13].asdf",
        format="flight_eng",
    )
    for ifile in range(3):
        ofile = str(tmp_path) + f"/OUT/im{ifile+1:02d}_asdf_to.fits"
        if ifile + 1 in [1, 3]:
            os.remove(ofile)
        else:
            assert not os.path.exists(ofile)

    # test directory functionality
    with chdir(tmp_path / "IN"):
        asdf_to_fits.main(fmatch="im02.asdf", format="flight_eng")
    ofile = str(tmp_path) + "/IN_fits_converted/im02_asdf_to.fits"
    with fits.open(ofile) as f:
        assert np.all(f[0].data == datacubes[1])
        assert 3.149 < f[0].header["TGROUP"] < 3.151
        assert 3.149 < f[0].header["TFRAME"] < 3.151

    # check OK if file already exists
    with chdir(tmp_path / "IN"):
        asdf_to_fits.main(fmatch="im02.asdf", format="flight_eng")
    ofile = str(tmp_path) + "/IN_fits_converted/im02_asdf_to.fits"
    with fits.open(ofile) as f:
        assert np.all(f[0].data == datacubes[1])
        assert 3.149 < f[0].header["TGROUP"] < 3.151
        assert 3.149 < f[0].header["TFRAME"] < 3.151
    os.remove(ofile)

