import re
import sys
from os import path

import numpy

from . import pyirc


def main(argv):
    """
    Constructs histograms of the input files.

    Parameters
    ----------
    argv : list of str
        Arguments, in the form of a list:
        ``[<ignore>, -f, <format>, -i, <1st input file>, -o, <output file>, -n, <number of files>]``.

        (This is designed so the calling script can be the 1st argument; in a program you can simply use
        ``None``.)

    Returns
    -------
    None

    """
    if len(argv) == 1:
        print(
            "Calling format: python histograms.py "
            "-f <format> -i <1st input file> -o <output file> -n <number of files>"
        )
        exit()

    # get command line directives
    for i in range(1, len(argv)):
        if argv[i] == "-f":
            fileformat = int(argv[i + 1])
        if argv[i] == "-i":
            infile = argv[i + 1]
        if argv[i] == "-o":
            outfile = argv[i + 1]
        if argv[i] == "-n":
            nfile = int(argv[i + 1])

    hist = numpy.zeros((65536, 32), dtype=numpy.int32)

    # Get array of new files:
    m = re.search(r"^(.+)/(\d\d\d\d\d\d\d\d)([^/]+)\_(\d+).fits", infile)
    if m:
        prefix = m.group(1)
        stamp = int(m.group(2))
        body = m.group(3)
    else:
        print("Match failed.")
        exit()
    filelist = []
    for k in range(nfile):
        thisname = prefix + f"/{stamp:d}" + body + f"_{k+1:03d}.fits"
        count = 0
        while not path.exists(thisname):
            count += 1
            thisname = prefix + f"/{stamp+count:d}" + body + f"_{k+1:03d}.fits"
            if count == 10000:
                print("Failed to find file", k)
                exit()
        filelist.append(thisname)

    nside = pyirc.get_nside(fileformat)
    ntslice = pyirc.get_num_slices(fileformat, filelist[0])

    # now build the histograms
    for ifile in range(nfile):
        print("file", ifile)
        for t in range(1, ntslice):
            frame = (
                65535 - pyirc.load_segment(filelist[k], fileformat, [0, nside, 0, nside], [t], False)[0, :, :]
            )
            for ch in range(32):
                xmin = max(ch * 128, 4)
                xmax = min((ch + 1) * 128, 4092)
                h, _ = numpy.histogram(frame[4:4092, xmin:xmax], bins=65536, range=(-0.5, 65535.5))
                hist[:, ch] += h

    numpy.savetxt(outfile, hist, fmt="%7d")


if __name__ == "__main__":
    main(sys.argv)
