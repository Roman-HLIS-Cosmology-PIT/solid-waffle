import numpy as np
from solid_waffle.correlation_run import run_vis_all
from solid_waffle.flat_simulator import simulate_flat


def test_run_vis(tmp_path):
    """
    Test function to make a 512x512 simulation, including visible flats, and run solid-waffle.

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

    for k in range(12):
        illum = 300.0
        ty = "light"
        if k >= 6:
            illum = 0.0
            ty = "dark"

        # Insert quantum yield here
        qyline = "QY: 0.1 0.09 0.0 0.09" if 3 <= k < 6 else ""

        sim_cfg = (
            "FORMAT: 1001\n"
            "NREADS: 20\n"
            "SUBSTEPS: 3\n"
            "DT: 2.75\n"
            "GAIN: 1.5\n"
            f"{qyline:s}\n"
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
        "DARK:\n"
        f"    {temp_dir}/dark_007.fits\n"
        f"    {temp_dir}/dark_008.fits\n"
        f"    {temp_dir}/dark_009.fits\n"
        "VISLIGHT:\n"
        f"    {temp_dir}/light_004.fits\n"
        f"    {temp_dir}/light_005.fits\n"
        f"    {temp_dir}/light_006.fits\n"
        "VISDARK:\n"
        f"    {temp_dir}/dark_010.fits\n"
        f"    {temp_dir}/dark_011.fits\n"
        f"    {temp_dir}/dark_012.fits\n"
        "FORMAT: 1001\n"
        "CHAR: Advanced 1 3 3 bfe\n"
        "NBIN: 4 4\n"
        "TIME:    1 10 12 20\n"
        "NLPOLY: 2 1 20\n"
        "VISTIME: 1 20 1 3\n"
        "COPYIRBFE\n"
        f"OUTPUT: {temp_dir}/analysis\n"
    )
    with open(temp_dir + "/analyze_cfg.txt", "w") as f:
        f.write(analyze_cfg)
    run_vis_all(temp_dir + "/analyze_cfg.txt")

    # Load the analysis
    data = np.loadtxt(temp_dir + "/analysis_visinfo.txt")
    extracted = np.mean(data, axis=0)[50:55]
    print(">>", extracted)

    # outputs from the first run
    expected_outputs = np.array([0.1, 0.09, 0.0, 0.09, 300.0 * 0.8 * 2.75])
    print("<<", expected_outputs)

    # tolerances -- if anything changes by more than these amounts,
    # pre-commit should warn the user!
    tol = np.array([0.015, 0.015, 0.015, 0.015, expected_outputs[-1] * 0.05])

    diff = np.amax(np.abs(extracted - expected_outputs) / tol)
    print(diff)
    assert diff < -1.0  # will fail so we can look at it
