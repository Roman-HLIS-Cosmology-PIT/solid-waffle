import os

import asdf
import numpy as np
from astropy.io import fits
from solid_waffle.asdf_to_fits import main as convert_asdf_to_fits_main
from solid_waffle.correlation_run import run_ir_all
from solid_waffle.flat_simulator import simulate_flat
from solid_waffle.pyirc import get_ntslice, load_segment


def create_dummy_asdf(asdf_path, data_type="flat", frames=20, shape=(512, 512)):
    """
    Create dummy asdf files similar to flats and darks
    """
    rng = np.random.default_rng(42)  # fixed seed
    if data_type == "flat":
        data = 3000 + rng.normal(0, 50, size=(frames, *shape))
    else:
        data = 100 + rng.normal(0, 5, size=(frames, *shape))
    data = np.clip(data, 0, 65535)
    tree = {"roman": {"data": data}}
    with asdf.AsdfFile(tree) as af:
        af.write_to(asdf_path)
    return data


def test_asdf_to_fits(tmp_path):
    """
    Test converting multiple ASDF files (flats and darks) to FITS.
    """
    original_data = {}
    for i in range(2):
        data = create_dummy_asdf(tmp_path / f"flat_{i+1:03d}.asdf", data_type="flat")
        original_data[f"flat_{i+1:03d}"] = np.clip(data, 0, 65535).astype(np.uint16)
    for i in range(2):
        data = create_dummy_asdf(tmp_path / f"dark_{i+1:03d}.asdf", data_type="dark")
        original_data[f"dark_{i+1:03d}"] = np.clip(data, 0, 65535).astype(np.uint16)

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        convert_asdf_to_fits_main()

        output_dir = tmp_path.parent / (tmp_path.name + "_fits_converted")
        assert output_dir.exists()

        expected_files = [
            output_dir / "flat_001_asdf_to.fits",
            output_dir / "flat_002_asdf_to.fits",
            output_dir / "dark_001_asdf_to.fits",
            output_dir / "dark_002_asdf_to.fits",
        ]
        for f in expected_files:
            assert f.exists(), f"Missing file: {f.name}"
            with fits.open(f) as hdul:
                data = hdul[0].data
                assert data.shape == (20, 512, 512)
                assert data.dtype == np.uint16
                # Now we can check actual values match original!
                key = f.name.replace("_asdf_to.fits", "")
                assert np.all(data == original_data[key])

    finally:
        os.chdir(orig_cwd)


def test_run_asdf(tmp_path):
    """
    Test that solid-waffle analysis pipeline works with asdf input files (formatpars=2001)
    """

    frames, ny, nx = 20, 512, 512
    fill_value = 1000.0
    data = np.full((frames, ny, nx), fill_value)

    asdf_path = str(tmp_path / "test.asdf")
    data_out = create_dummy_asdf(asdf_path, data_type="flat", frames=frames, shape=(ny, nx))

    # Test get_ntslice
    ntslice = get_ntslice(asdf_path, formatpars=2002)
    assert ntslice == frames

    # Test load_segment
    xyrange = [0, nx, 0, ny]
    tslices = [1, 2, 3]
    result = load_segment(asdf_path, 2002, xyrange, tslices, verbose=False)

    assert result.shape == (len(tslices), ny, nx)
    # Check the 65535 - data transformation was applied
    expected = np.clip(65535 - data_out, 0, 65535)
    assert np.allclose(result, expected[0:3])


def test_run(tmp_path):
    """
    Test function to make a 512x512 simulation and run solid-waffle.

    Parameters
    ----------
    tmp_path : str or pathlib.Path
        Directory in which to run the test.

    Returns
    -------
    None

    """

    temp_dir = str(tmp_path)
    print("using", temp_dir)

    # Make the simulation

    for k in range(8):
        illum = 300.0
        ty = "light"
        if k >= 4:
            illum = 0.0
            ty = "dark"

        sim_cfg = (
            "FORMAT: 1001\n"
            "NREADS: 20\n"
            "SUBSTEPS: 3\n"
            "DT: 2.75\n"
            "GAIN: 1.5\n"
            f"ILLUMINATION: {illum}\n"
            "QE: 8e-1\n"
            f"RNGSEED: {201909+k}\n"
            "LEGACY\n"  # <-- legacy RNG is stable since numpy 1.16
            "RESET_E: 1.0e2\n"
            "NOISE: Gauss\n"
            "WAVEMODE: ir\n"
            "BFE: true\n"
            "L_IPC: true 0.01\n"
            "NL: quadratic 1.4\n"
            f"OUTPUT: {temp_dir}/{ty}_{k+1:03d}.fits\n"
        )

        with open(temp_dir + "/sim_cfg.txt", "w") as f:
            f.write(sim_cfg)
        simulate_flat.run_config(temp_dir + "/sim_cfg.txt")

    # Now analyze it
    analyze_cfg = (
        "DETECTOR: Test_simulation\n"
        "LIGHT:\n"
        f"    {temp_dir}/light_001.fits\n"
        f"    {temp_dir}/light_002.fits\n"
        f"    {temp_dir}/light_003.fits\n"
        f"    {temp_dir}/light_004.fits\n"
        "DARK:\n"
        f"    {temp_dir}/dark_005.fits\n"
        f"    {temp_dir}/dark_006.fits\n"
        f"    {temp_dir}/dark_007.fits\n"
        f"    {temp_dir}/dark_008.fits\n"
        "FORMAT: 1001\n"
        "CHAR: Advanced 1 3 3 bfe\n"
        "NBIN: 4 4\n"
        "TIME:    1 10 12 20\n"
        "TIME2A:  1 2 4 20\n"
        "TIME2B:  1 2 4 20\n"
        "TIME3:   1 2 4 20\n"
        f"OUTPUT: {temp_dir}/analysis\n"
    )
    with open(temp_dir + "/analyze_cfg.txt", "w") as f:
        f.write(analyze_cfg)
    run_ir_all(temp_dir + "/analyze_cfg.txt")

    # Load the analysis
    data = np.loadtxt(temp_dir + "/analysis_summary.txt")
    print(">>", np.mean(data, axis=0))

    # outputs from the first run
    expected_outputs = np.array(
        [
            1.50000000e00,
            1.50000000e00,
            1.55543125e04,
            1.84032388e00,
            1.66757590e00,
            1.49223302e00,
            9.88118358e-03,
            1.01165409e-02,
            1.40400538e-06,
            6.55764439e02,
            2.68351544e-04,
            1.18591758e02,
            1.20411533e02,
            3.16166804e-07,
            -4.55713429e-07,
            2.01035345e-07,
            4.62063881e-08,
            6.72640364e-08,
            6.11719038e-07,
            -2.93234974e-07,
            9.45496306e-08,
            1.16587044e-07,
            4.83671986e-07,
            8.42673374e-08,
            4.60151153e-07,
            -1.68711037e-06,
            8.93645681e-08,
            1.54687202e-07,
            1.12356304e-07,
            -2.24261168e-07,
            1.69978299e-07,
            2.51569092e-07,
            3.37570881e-07,
            -1.48281734e-07,
            4.64848787e-08,
            2.94327429e-07,
            2.93979598e-07,
            -1.48579605e-07,
            6.75570972e-06,
            2.74537451e-06,
            2.67357466e-07,
        ]
    )

    # tolerances -- if anything changes by more than these amounts,
    # pre-commit should warn the user!
    tol = np.array(
        [
            1e-4,
            1e-4,
            2.5,
            0.01,
            0.01,
            0.01,
            1e-4,
            1e-4,
            5e-8,
            0.5,
            1e-4,
            0.1,
            0.1,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            1e-7,
            5e-7,
            5e-7,
            5e-7,
        ]
    )

    diff = np.amax(np.abs(np.mean(data, axis=0) - expected_outputs) / tol)
    print(diff)
    assert diff < 1.0
