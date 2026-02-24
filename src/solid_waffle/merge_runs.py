import argparse
from multi_config import MultiConfig

def main():
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