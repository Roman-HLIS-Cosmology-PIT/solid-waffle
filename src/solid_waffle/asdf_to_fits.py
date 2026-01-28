import numpy as np
import os
import asdf
import fitsio
import glob
from astropy.io import fits
import time

def main():
    current_dir = os.getcwd();
    output_dir = os.path.join(os.path.dirname(current_dir), "fits_converted")
    asdf_files = glob.glob(os.path.join(current_dir, "*.asdf"))
    print("ASDF files found: ") 
    for fn in asdf_files:
        print(fn)
    for fn in asdf_files:
        base = os.path.basename(fn)
        new_file_path = os.path.join(output_dir, base.replace(".asdf", "asdf_to.fits"))
        if os.path.exists(new_file_path):
            print(f"Skipping (already exists): {new_file_path}")
            continue
        with asdf.open(fn) as f:
            fits.PrimaryHDU(f["roman"]["data"].value.astype(np.uint16)).writeto(new_file_path, overwrite=True)


if __name__ == "__main__":
    main()
    

