import glob
import os
import sys

import asdf
import numpy as np
from astropy.io import fits


def main(input_dir=None, output_dir=None, fmatch="*.asdf", format="wfi_tvac"):
    """
    Converts all asdf files in a directory to fits files in a new directory

    Parameters
    ----------
    input_dir : str, optional
        The input directory; defaults to the current directory.
    output_dir : str, optional
        The output directory; defaults to a copy of the current directory name with
        ``_fits_converted`` appended.
    fmatch : str, optional
        Regular expression string to match filenames (glob-style); default is to take
        all asdf files.
    format : str, optional
        The input data format. Options are:

        - **"wfi_tvac_rst"** : WFI TVAC data; include the reset/read in the file.
        - **"wfi_tvac"** : WFI TVAC data; do **not** include the reset/read in the file.
        - **"flight_eng"** : Flight data, diagnostic mode; include the reset/read in the file.


    Returns
    -------
    None

    """

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    current_dir_name = os.path.basename(current_dir)

    # where to get and put files
    if input_dir is None:
        input_dir = current_dir
    if output_dir is None:
        output_dir = os.path.join(parent_dir, current_dir_name + "_fits_converted")

    os.makedirs(output_dir, exist_ok=True)
    asdf_files = glob.glob(os.path.join(input_dir, fmatch))
    print("ASDF files found: ")
    for fn in asdf_files:
        print(fn)
    for fn in asdf_files:
        base = os.path.basename(fn)
        new_file_path = os.path.join(output_dir, base.replace(".asdf", "_asdf_to.fits"))
        if os.path.exists(new_file_path):
            print(f"Skipping (already exists): {new_file_path}")
            continue
        with asdf.open(fn) as f:
            n = np.shape(f["roman"]["data"])[0]
            k = 1 if format in ["wfi_tvac_rst", "flight_eng"] else 0  # number of additional frames
            alldata = np.zeros((n + k, 4096, 4224), dtype=np.uint16)
            data = alldata[k:, :, :]
            data[:, :, :4096] = f["roman"]["data"]
            data[:, :, -128:] = f["roman"]["amp33"]
            if format == "wfi_tvac_rst":
                alldata[0, :, :4096] = f["roman"]["reset_reads"]
                alldata[0, :, -128:] = f["roman"]["amp33_reset_reads"]
            if format == "flight_eng":
                alldata[0, :, :4096] = f["roman"]["reference_read"]
                alldata[0, :, -128:] = f["roman"]["reference_amp33"]
                _offset = (
                    alldata[0].astype(np.int32) - f["roman"]["meta"]["instrument"]["data_encoding_offset"]
                )
                for j in range(n):
                    data[j, :, :] = np.clip(_offset + data[j].astype(np.int32), 0, 65535)
            h = fits.PrimaryHDU(alldata)
            h.header["TFRAME"] = f["roman"]["meta"]["exposure"]["frame_time"]
            h.header["TGROUP"] = f["roman"]["meta"]["exposure"]["frame_time"]
            h.writeto(new_file_path, overwrite=True)


if __name__ == "__main__":
    main(format=sys.argv[1] if len(sys.argv) > 1 else "wfi_tvac")
