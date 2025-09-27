import sys

from solid_waffle.correlation_run import run_vis_all

run_vis_all(sys.argv[1], run_ir_first=bool(len(sys.argv) >= 3))
