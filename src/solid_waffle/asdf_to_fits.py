import numpy as np
import os
import asdf
import fitsio
import glob
from astropy.io import fits
import time


asdf_files = glob.glob(os.path.join(os.getcwd(), "*.asdf"))
print("ASDF files found: ")
for file in asdf_files:
    print(file)

    

