ASDF file support in solid-waffle
#################################

Most Roman data are being distributed in ASDF format. While solid-waffle is capable of reading these directly, it has some scripts that access large numbers of parts of data files, and currently the FITS readers are much faster for this. Therefore, we expect most users will want to convert the data to FITS format (the scripts below convert to `format=1` in solid-waffle convention).

To do so, we provide the ``asdf_to_fits`` utility. This can be used as follows:

.. code-block:: python

    from solid_waffle import asdf_to_fits

    asdf_to_fits.main(
        input_dir="myindata",  # will read from the myindata/ directory
        output_dir="myoutdata",  # will write to the myoutdata/ directory
        format="flight_eng"  # which format; here flight engineering mode
    )

The other commonly used formats would be ``wfi_tvac`` (for the Instrument-level TVAC data) or ``wfi_tvac_rst`` (if you want to include the reset-read frames as well).

The above function will make FITS versions of the data: e.g., a file ``myindata/file1.asdf`` gets mapped to ``myoutdata/file1_asdf_to.fits``.

You may also supply the "fmatch" argument in glob format if you want to only convert some files:

.. code-block:: python

    asdf_to_fits.main(
        input_dir="myindata",  # will read from the myindata/ directory
        output_dir="myoutdata",  # will write to the myoutdata/ directory
        fmatch=f"files0?.asdf",  # will convert files08.asdf, but not files10.asdf
        format="flight_eng"  # which format; here flight engineering mode
    )

The output FITS file has a 3D data cube in the Primary HDU with shape (nreads, 4096, 4224) (the "4224" is because the amp33 data is 
appended to the right side). The TFRAME and TGROUP keywords are also set.

There is also a "CONFIG" HDU that saves metadata, so you can write::

  >>> f = fits.open("OUT/im01_asdf_to.fits")
  >>> f["CONFIG"].header

      XTENSION= 'BINTABLE'           / binary table extension                         
      BITPIX  =                    8 / array data type                                
      NAXIS   =                    2 / number of array dimensions                     
      NAXIS1  =                    1 / length of dimension 1                          
      NAXIS2  =                 1188 / length of dimension 2                          
      PCOUNT  =                    0 / number of group parameters                     
      GCOUNT  =                    1 / number of groups                               
      TFIELDS =                    1 / number of table fields                         
      TTYPE1  = 'config  '                                                            
      TFORM1  = '1A      '                                                            
      EXTNAME = 'CONFIG  '                                                            
      ORIGFILE= 'im01.asdf'                                                           
      END                                                                             

  >>> origdict = yaml.safe_load("".join(f["CONFIG"].data["config"]))
  >>> origdict

      {
          'asdf_library': {'author': 'TheASDFDevelopers', 'homepage': 'http://github.com/asdf-format/asdf',
              'name': 'asdf', 'version': '5.3.1'},
          'history': {
              'extensions': [{'extension_class': 'asdf.extension._manifest.ManifestExtension',
                  'extension_uri': 'asdf://asdf-format.org/core/extensions/core-1.6.0',
                  'manifest_software': {'name': 'asdf_standard', 'version': '1.5.0'},
                  'software': {'name': 'asdf', 'version': '5.3.1'}}]},
          'roman': {
              'amp33': {'source': 0, 'datatype': 'uint16', 'byteorder': 'little', 'shape': [4, 4096, 128],
                  'offset': 34611200, 'strides': [34603008, 8448, 2]},
              'amp33_reference_read': {'source': 3, 'datatype': 'uint16', 'byteorder': 'little', 'shape': [1, 4096, 128]},
              'amp33_reset_reads': {'source': 4, 'datatype': 'uint16', 'byteorder': 'little', 'shape': [1, 4096, 128]},
              'data': {'source': 0, 'datatype': 'uint16', 'byteorder': 'little', 'shape': [4, 4096, 4096],
                  'offset': 34603008, 'strides': [34603008, 8448, 2]},
              'meta': {'exposure': {'frame_time': 3.15}},
              'reference_read': {'source': 1, 'datatype': 'uint16', 'byteorder': 'little', 'shape': [1, 4096, 4096]},
              'reset_reads': {'source': 2, 'datatype': 'uint16', 'byteorder': 'little', 'shape': [1, 4096, 4096]}}
      }

(I added some newlines in the outputs for readability on the GitHub viewer.)
