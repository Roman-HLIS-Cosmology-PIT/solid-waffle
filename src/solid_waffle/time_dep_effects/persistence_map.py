import numpy as np
from astropy.io import fits
import os

LAST_BRIGHT_FILES = [
    "TVAC2_NOMOPS_SCIMON_20240419045204_WFI06_uncal_asdf_to.fits", # file 22
    "TVAC2_NOMOPS_SCIMON_20240419074832_WFI06_uncal_asdf_to.fits", # file 78
    "TVAC2_NOMOPS_SCIMON_20240419093417_WFI06_uncal_asdf_to.fits", # file 90
    "TVAC2_NOMOPS_SCIMON_20240419102745_WFI06_uncal_asdf_to.fits", # file 106
    "TVAC2_NOMOPS_SCIMON_20240419112117_WFI06_uncal_asdf_to.fits" # file 122
]

FIRST_DARK_FILES = [
    "TVAC2_NOMOPS_SCIMON_20240419045519_WFI06_uncal_asdf_to.fits", # file 23
    "TVAC2_NOMOPS_SCIMON_20240419075152_WFI06_uncal_asdf_to.fits",  # file 79
    "TVAC2_NOMOPS_SCIMON_20240419093736_WFI06_uncal_asdf_to.fits",  # file 91
    "TVAC2_NOMOPS_SCIMON_20240419103104_WFI06_uncal_asdf_to.fits",  # file 107
    "TVAC2_NOMOPS_SCIMON_20240419112436_WFI06_uncal_asdf_to.fits" # file 123
]

TRUE_DARK_FILES = [
    "TVAC2_NOMOPS_SCIMON_20240419030148_WFI06_uncal_asdf_to.fits", # file 0
    "TVAC2_NOMOPS_SCIMON_20240419030455_WFI06_uncal_asdf_to.fits", # file 1
    "TVAC2_NOMOPS_SCIMON_20240419030801_WFI06_uncal_asdf_to.fits", # file 2
    "TVAC2_NOMOPS_SCIMON_20240419031152_WFI06_uncal_asdf_to.fits", # file 3
    "TVAC2_NOMOPS_SCIMON_20240419031458_WFI06_uncal_asdf_to.fits" # file 4
]

OUTPUT_FILE = "persistence_map.fits"
OUTPUT_NORM_FILE = "normalized_persistence_map.fits"
DATA_DIR = "/fs/scratch/PAS2340/amyalbert8/TVAC2_data/sci_monitor_darks_nominal_ops_fits_converted"

def load_data(filepath):
    print(f"Loading: {os.path.basename(filepath)}")
    with fits.open(filepath) as hdul:
        data_with_ref = hdul[0].data.astype(np.float32)
    data = data_with_ref[:, 4:4092, 4:4092]
    print(f" Shape: {data.shape}, Data Type: {data.dtype}")
    return data


def make_resultants(data):
    data = data.astype(np.float32)
    resultants = np.zeros((8, data.shape[1], data.shape[2]), dtype = np.float32)
    resultants[0, :, :] = data[0, :, :]
    resultants[1, :, :] = data[1, :, :]
    resultants[2, :, :] = np.mean(data[2:4, :, :], axis=0)
    resultants[3, :, :] = np.mean(data[4:10, :, :], axis=0)
    resultants[4, :, :] = np.mean(data[10:26, :, :], axis =0)
    resultants[5, :, :] = np.mean(data[26:32, :, :], axis=0)
    resultants[6, :, :] = np.mean(data[32:34, :, :], axis=0)
    resultants[7, :, :] = data[34, :, :]

    return resultants

def make_2d_image(resultants):
    w_resultants = np.array([
        0.0000000e+00, 
        -2.0788277e-03, 
        -3.4818146e-03, 
        -6.5069445e-03, 
        -1.7643711e-10,  
        6.5069450e-03,  
        3.4818146e-03,  
        2.0788277e-03
    ], dtype = np.float32)

    image_2d = np.einsum('i,i...->...', w_resultants, resultants)

    return image_2d

def save_fits(image_2d, output_path, extra_headers=None):
    hdu = fits.PrimaryHDU(image_2d)

    hdu.header['BUNIT']   = ('DN/s',        'Units of pixel values')
    hdu.header['NRESULT'] = (8,             'Number of resultants used')
    hdu.header['MULTACC'] = ('IM_107_8_S', 'MultiAccum table used')
    hdu.header['AUTHOR']  = ('Amy Albert',  'Who made this file')
    hdu.header['COMMENT'] = 'Persistence map from Roman WFI TVAC2 first dark exposures'

    if extra_headers:
        for key, (value, comment) in extra_headers.items():
            hdu.header[key] = (value, comment)

    hdu.writeto(output_path, overwrite=True)
    print(f"  Saved: {output_path}")

def make_dark_baseline(true_dark_files):
    dark_images = []
    for fname in true_dark_files:
        filepath = os.path.join(DATA_DIR, fname)
        image_2d = process_file(filepath)
        dark_images.append(image_2d)

    dark_baseline = np.mean(np.stack(dark_images, axis=0), axis=0)
    print(f"Dark Baseline shape = {dark_baseline.shape}")
    print(f"Dark Baseline mean value = {np.mean(dark_baseline):.6f} DN/s")
    return dark_baseline

def process_file(filepath):
    data = load_data(filepath)
    resultants = make_resultants(data)
    image_2d = make_2d_image(resultants)
    return image_2d

def main():
    dark_baseline = make_dark_baseline(TRUE_DARK_FILES)
    dark_images = []
    bright_images = []

    for i, (dark_file, bright_file) in enumerate(zip(FIRST_DARK_FILES, LAST_BRIGHT_FILES)):
        print(f"\n[Pair {i+1}/{len(FIRST_DARK_FILES)}]")

        dark_images.append(process_file(os.path.join(DATA_DIR, dark_file)))
        bright_images.append(process_file(os.path.join(DATA_DIR, bright_file)))

    print("\nSubtracting dark baseline from first darks...")
    persistence_images = []

    for i, dark_image in enumerate(dark_images):
        persistence = dark_image - dark_baseline
        persistence_images.append(persistence)
        print(f"  Pair {i+1}: mean persistence signal = {np.mean(persistence):.6f} DN/s")

    print("\nStacking images...")
    persistence_stack = np.stack(persistence_images, axis=0)
    bright_stack = np.stack(bright_images, axis=0)

    persistence_map = np.mean(persistence_stack, axis=0)
    bright_map = np.mean(bright_stack, axis=0)

    print(f"  Persistence stack shape: {persistence_stack.shape}")
    print(f"  Bright stack shape:      {bright_stack.shape}")
    print(f"  Persistence map shape:   {persistence_map.shape}")
    print(f"  Mean persistence signal: {np.mean(persistence_map):.6f} DN/s")

    print("\nNormalising persistence map by bright signal...")
    normalised_map = np.where(
        bright_map > 0,
        persistence_map / np.where(bright_map > 0, bright_map, 1),
        0.0
    )
    print(f"  Mean normalised persistence: {np.mean(normalised_map):.6f}")

    print("\nSaving outputs...")

    save_fits(
        persistence_map,
        OUTPUT_FILE,
        extra_headers={
            'MAPTYPE': ('RAW',   'Raw persistence signal in DN/s'),
            'NDARK':   (len(TRUE_DARK_FILES),  'Number of true darks in baseline'),
            'NPAIRS':  (len(FIRST_DARK_FILES), 'Number of bright/dark pairs used')
        }
    )

    save_fits(
        normalised_map,
        OUTPUT_NORM_FILE,
        extra_headers={
            'BUNIT':   ('fraction',  'Persistence as fraction of bright signal'),
            'MAPTYPE': ('NORM',      'Persistence normalised by bright signal'),
            'NDARK':   (len(TRUE_DARK_FILES),  'Number of true darks in baseline'),
            'NPAIRS':  (len(FIRST_DARK_FILES), 'Number of bright/dark pairs used')
        }
    )

if __name__ == "__main__":
    main()

