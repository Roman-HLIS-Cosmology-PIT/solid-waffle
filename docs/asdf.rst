ASDF file support in solid-waffle
#################################

Most Roman data are being distributed in ASDF format. While solid-waffle is capable of reading these directly, it has some scripts that access large numbers of parts of data files, and currently the FITS readers are much faster for this. Therefore, we expect most users will want to convert the data to FITS format (generally format "6").

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

