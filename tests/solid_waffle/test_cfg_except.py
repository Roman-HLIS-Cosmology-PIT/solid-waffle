import pytest
from solid_waffle.correlation_run import run_ir_all
from solid_waffle.flat_simulator import simulate_flat


def test_cfgerr(tmp_path):
    """
    Test configuration errors on a 512x512 simulation.

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

    for k in range(4):
        illum = 300.0
        ty = "light"
        if k >= 2:
            illum = 0.0
            ty = "dark"

        nlstr = "quadratic 1.4" if k != 2 else "quartic -1.4 0 0"
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
            f"NL: {nlstr}\n"
            f"OUTPUT: {temp_dir}/{ty}_{k+1:03d}.fits\n"
        )

        with open(temp_dir + "/sim_cfg.txt", "w") as f:
            f.write(sim_cfg)
        simulate_flat.run_config(temp_dir + "/sim_cfg.txt")

    # analysis configuration
    analyze_cfg = (
        "DETECTOR: Test_simulation\n"
        "LIGHT:\n"
        f"    {temp_dir}/light_001.fits\n"
        f"    {temp_dir}/light_002.fits\n"
        "DARK:\n"
        f"    {temp_dir}/dark_003.fits\n"
        f"    {temp_dir}/dark_004.fits\n"
        "FORMAT: 1001\n"
        "NBIN: 4 4\n"
        "TIME:    1 10 12 20\n"
        f"OUTPUT: {temp_dir}/analysis\n"
    )

    # this asks for VTPE without HOTPIX
    with open(temp_dir + "/analyze_cfg.txt", "w") as f:
        f.write(analyze_cfg)
        f.write("VTPE\n")
    with pytest.raises(ValueError):
        run_ir_all(temp_dir + "/analyze_cfg.txt")

    # HOTPIX_TSTART without HOTPIX
    with open(temp_dir + "/analyze_cfg.txt", "w") as f:
        f.write(analyze_cfg)
        f.write("HOTPIX_TSTART: 3\n")
    with pytest.raises(ValueError):
        run_ir_all(temp_dir + "/analyze_cfg.txt")
