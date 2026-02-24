"""
merge_runs.py - Aggregate already-completed run_ir_all outputs.

Loads results from existing _summary.txt files for each config and
combines them into a single averaged output without rerunning the
expensive fit_parameters computation.

Usage
-----
    python merge_runs.py config1.txt config2.txt config3.txt --outstem combined

Arguments
---------
config_files : list of str
    Paths to the individual configuration files. Each must have a
    corresponding <outstem>_summary.txt already on disk.
--outstem : str, optional
    Output stem for the combined results. Defaults to the first
    config's outstem with '_combined' appended.

Output
------
    <outstem>_summary.txt   combined characterization results
    <outstem>_multi.pdf     combined parameter maps figure
    <outstem>_m23.pdf       combined method 2 and 3 figure
    <outstem>_hot.txt       combined hot pixel report
    <outstem>_hotipc.pdf    combined hot pixel figure
    <outstem>_nl.txt        combined non-linearity table
"""
import argparse

from solid_waffle.multi_config import MultiConfig


def main():
    """
    Aggregate already-completed run_ir_all outputs into one combined result.

    Reads command line arguments for config file paths and optional output
    stem, loads results from existing _summary.txt files, and writes
    combined outputs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config_files", nargs="+")
    parser.add_argument("--outstem", default=None)
    args = parser.parse_args()
    mcf = MultiConfig.from_summaries(args.config_files)
    if args.outstem:
        mcf.outstem = args.outstem
    else:
        mcf.outstem = mcf.configs[0].outstem + "_combined"
    mcf.generate_nonlinearity(write_to_file=True)
    mcf.write_basic_figure()
    mcf.alt_methods(verbose=True)
    mcf.method_23_plot()
    with open(mcf.outstem + "_summary.txt", "w") as f:
        f.write(mcf.text_output())
    s = mcf.hotpix_analysis(verbose=True)
    with open(mcf.outstem + "_hot.txt", "w") as f:
        f.write(s)
    mcf.hotpix_plots()


if __name__ == "__main__":
    main()
