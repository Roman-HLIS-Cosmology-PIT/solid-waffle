import numpy as np
from solid_waffle.correlation_run import run_ir_all
from solid_waffle.flat_simulator import simulate_flat


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
