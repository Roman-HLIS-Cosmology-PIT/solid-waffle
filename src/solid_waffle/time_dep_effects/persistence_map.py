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

OUTPUT_FILE = "persistence_map.fits"
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

def save_image_fits(image_2d, output_path):
    hdu = fits.PrimaryHDU(image_2d)
    hdu.header['BUNIT'] = ('DN/s', 'Units of pixel values')
    hdu.header['NRESULT'] = (8, 'Number of resultants used')
    hdu.header['MULTACC'] = ('IM_107_8_S', 'MultiAccum Table used')
    hdu.header['AUTHOR'] = ('Amy Albert', 'Who made this file')
    hdu.header['COMMENT'] = 'Persistance map from Roman WFI TVAC2 first dark exposures'

    hdu.writeto(output_path, overwrite="True")
    print(f" Saved: {output_path}")

def main():
    processed_images = []
    for i, dark_file in enumerate(FIRST_DARK_FILES):
        print(f"\n[Exposure {i+1}/{len(FIRST_DARK_FILES)}]")

        filepath = os.path.join(DATA_DIR, dark_file)
        data = load_data(filepath)

        resultants = make_resultants(data)
        print(f" Resultants shape: {resultants.shape}")

        image_2d = make_2d_image(resultants)
        print(f" 2D image shape: {image_2d.shape}")

        processed_images.append(image_2d)

    print("\n Stacking images")

    stack = np.stack(processed_images, axis=0)
    persistence_map = np.mean(stack, axis=0)

    print(f" Stack shape: {stack.shape}")
    print(f" Persistence map shape: {persistence_map.shape}")

    save_image_fits(persistence, OUTPUT_FILE)

if __name__ == "__main__":
    main()

