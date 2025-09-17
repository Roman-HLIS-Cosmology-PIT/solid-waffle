"""
Routines to run the infrared flat correlations.

Classes
-------
EmptyClass
    Blank, can add attributes later.
Config
    Extracts configuration data from a multiline string.

Functions
---------
run_ir_all
  Runs the IR characterization.

"""

import os
import shutil
import sys
import time
import re
import numpy
import pyirc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class EmptyClass:
  """Blank, can add attributes later."""
  pass

class Config(cfg):
  """
  Extracts configuration data from a multiline string.

  Some attributes may not be present depending on the configuration.

  Parameters
  ----------
  cfg : str
      The configuration (as a string).
  verbose : bool, optional
      Whether to specify lots of information when reading the configuration.

  Attributes
  ----------
  outstem : str
      The stem for output files.
  use_cmap : str
      The color bar range to use.
  mydet : str
      Name of the detector.
  lightfiles : list of str
      List of the flat field file names.
  darkfiles : list of str
      List of the dark exposure file names, same length as `lightfiles`.
  formatpars : int
      The format type for these images.
  nx, ny : int
      The number of bins to break the detector array into on the x and y axes.
  tslices : list of int
      Time slices to use for BFE correlations. Length 4, ascending.
  tslicesM2a, tslicesM2b, tslicesM3 : list of int
      Time slices to use for alternative BFE tests. Length 4, ascending.
  fullref : bool
      Apply full reference pixel corrrection?
  sensitivity_spread_cut : float
      The fractional deviation from the median flat at which to cut pixels.
  critfrac : float
      Fraction of the pixels that need to be good to keep a superpixel.
  mychar : str
      Type of characterization to use (options are "Basic" and "Advanced").
  hotpix : bool
      Whether to do the hot pixel analysis.
  hotpix_ADU_range : list of float
      Hot pixel selection range (length 4: ``[Smin, Smax, stability, f_isolation]``).
  ref_for_hotpix_is_autocorr : bool
      Plot hot pixel results relative to autocorrelation results?
  hotpix_logtspace : bool
      Space hot pixel samples logarithmically?
  hotpix_slidemed : bool
      Use sliding median method to get hot pixel IPC measuremenent?
  s_bfe : int
      Radius of BFE kernel.
  p_order : int
      If >0, builds non-linearity polynomial table through this order.
  basicpar : class
      Basic characterization parameters.
  bfepar : class
      BFE characterization parameters.
  narrowfig : bool
      Output figures in narrow format?
  maskX, maskY : list of int
      Super-pixel regions to mask out (can be empty).
  tchar1, tchar2 : int
      Time steps for advanced characterization.
  ncycle : int
      Number of iterations for advanced characterization.
  ipnltype : str
      Inter-pixel non-linearity model: 'bfe' or 'nlipc'.
  nlfit_ts, nlfit_te : int
      Range of time stamps for fitting the non-linearity curve.
  NTMAX : int
      Maximum time slice allowed.
  swi : class
      Column information.
  N : int
      Size of the SCA.
  dx, dy : int
      Size of super-pixels.
  npix : int
      Number of pixels in a super-pixel.
  full_info : np.array
      Calibration results, shape = (`ny`, `nx`, ...).
  is_good : np.array
      Good super-pixel map, shape = (`ny`, `nx`).
  lightref, darkref : np.array
      Reference pixel corrections for the flats and darks.
  mean_full_info, std_full_info : np.array
      Array mean and standard deviation of `full_info`.
  nlfit, nlder : np.array
      Array of the ramp fit and derivative, shape = (nt, `ny`, `nx`).
  used_2a, used_2b, used_3 : bool
      Whether the alternative methods were implemented.
  ntM2a, ntM2b, ntM3 : int
      Number of time stamps in each alternative method.
  Method2a_slopes, Method2b_slopes, Method3_slopes : np.array
      Alternative gain/IPC calculation slopes, shape = (`ny`, `nx`).
  Method2a_vals, Method2b_vals, Method3_vals : np.array
      Alternative gain/IPC calculation values, shape = (`ny`, `nx`, ...).
  tfmin, tfmax, tfminB, tfmaxB, tfmin3, tfmax3 : int
      Time stamps for Methods 2a, 2b, and 3.
  slope_2a_BFE, slope_2a_NLIPC, slope_2b_BFE, slope_2b_NLIPC, slope_3_beta, slope_3_BFE, slope_3_NLIPC : float
      Predicted slopes for the different alternative methods for different sources of IPNL.
  PV2a, PV2b, PV3 : float
      Peak-to-valley errors of alternative method fits.
  hotX, hotY : np.array of int
      Coordinates of the hot pixels; same length.
  htsteps : list of int
      Time steps for the hot pixel analysis.
  hotpix_signal : np.array
      Signal levels of the hot pixels, shape = (len(`hotX`), len(`htsteps`))
  hotpix_alpha : np.array
      IPC estimated from the hot pixels, shape = (len(`hotX`), len(`htsteps`))
  ipcmed_x, ipcmed_y, ipcmed_yerr : np.array
      Signal, IPC, and IPC error for the hot pixel signal-dependent IPC method. 1D arrays.
  grid_alphaCorr, grid_alphaCorrErr, grid_alphaHot, grid_alphaHotErr : np.array
      The correlation- and hot pixel-based IPC in each of the 16 hexadecants of the detector array,
      and their uncertainties.

  Methods
  -------
  __init__
      Constructor.
  fit_parameters
      Build general parameters (gain, IPC, NL) and BFE Method 1.
  generate_nonlinearity
      Generates non-linearity data.
  write_basic_figure
      Makes a PDF figure of the SCA parameter maps.
  alt_methods
      Methods 2a, 2b, and 3 trend computation.
  method_23_plot
    Method 2 and 3 characterization Multi-panel figure showing basic characterization.
  text_output
    Generate a text summary.
  hotpix_analysis
    Hot pixel analysis.
  hotpix_plots
    Makes the hot pixel plots.

  """

  def __init__(self, verbose = False):

    self.outstem = 'default_output'
    self.use_cmap = 'gnuplot'

    self.mydet = ''
    self.lightfiles = self.darkfiles = []
    self.formatpars = 1
    self.nx = self.ny = 32
    self.tslices = [3,11,13,21]
    self.tslicesM2a = self.tslicesM2b = self.tslicesM3 = []
    self.fullref = True
    self.sensitivity_spread_cut = .1
    self.critfrac = 0.75
    self.mychar = 'Basic'
    self.hotpix = False
    self.ref_for_hotpix_is_autocorr = False
    self.hotpix_logtspace = False
    self.hotpix_slidemed = False

    self.swi = pyirc.swi_init # initialize column table; will add more

    # order parameters
    self.s_bfe = 2 # order of BFE parameters
    self.p_order = 0 # non-linearity polynomial table coefficients (table at end goes through order p_order)
                       # set to zero to turn this off

    # Parameters for basic characterization
    basicpar = EmptyClass()
    basicpar.epsilon = .01
    basicpar.subtr_corr = True
    basicpar.noise_corr = True
    basicpar.reset_frame = 1
    basicpar.subtr_href = True
    basicpar.full_corr = True
    basicpar.leadtrailSub = False
    basicpar.g_ptile = 75.
    basicpar.fullnl = False
    basicpar.use_allorder = False
    self.basicpar = basicpar

    # Parameters for BFE
    bfepar = EmptyClass()
    bfepar.epsilon = .01
    bfepar.treset = basicpar.reset_frame
    bfepar.blsub = True
    bfepar.fullnl = False
    self.bfepar = bfepar

    # Plotting parameters
    self.narrowfig = False

    # Read in information
    content = cfg.splitlines()
    is_in_light = is_in_dark = False
    self.maskX = [] # list of regions to mask
    self.maskY = []
    for line in content:
      # Cancellations
      m = re.search(r'^[A-Z]+\:', line)
      if m: is_in_light = is_in_dark = False

      # Searches for files -- must be first given the structure of this script!
      if is_in_light:
        m = re.search(r'^\s*(\S.*)$', line)
        if m: self.lightfiles += [m.group(1)]
      if is_in_dark:
        m = re.search(r'^\s*(\S.*)$', line)
        if m: self.darkfiles += [m.group(1)]

      # -- Keywords go below here --

      # Search for outputs
      m = re.search(r'^OUTPUT\:\s*(\S*)', line)
      if m: self.outstem = m.group(1)
      # Search for input files
      m = re.search(r'^LIGHT\:', line)
      if m: self.is_in_light = True
      m = re.search(r'^DARK\:', line)
      if m: self.is_in_dark = True
      # Format
      m = re.search(r'^FORMAT:\s*(\d+)', line)
      if m: self.formatpars = int(m.group(1))

      # Bin sizes
      m = re.search(r'^NBIN:\s*(\d+)\s+(\d+)', line)
      if m:
        self.nx = int(m.group(1))
        self.ny = int(m.group(2))

      # Characterization type (Basic or Advanced)
      m = re.search(r'^CHAR:\s*(\S+)', line)
      if m:
         self.mychar = m.group(1)
         if self.mychar.lower()=='advanced':
           m = re.search(r'^CHAR:\s*(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)', line)
           if m:
             self.tchar1 = int(m.group(2))
             self.tchar2 = int(m.group(3))
             self.ncycle = int(m.group(4))
             self.ipnltype = m.group(5)
           else:
             print ('Error: insufficient arguments: ' + line + '\n')
             exit()

      # Time slices
      m = re.search(r'^TIME:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
      if m: self.tslices = [ int(m.group(x)) for x in range(1,5)]
      m = re.search(r'^TIME2A:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
      if m: self.tslicesM2a = [ int(m.group(x)) for x in range(1,5)]
      m = re.search(r'^TIME2B:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
      if m: self.tslicesM2b = [ int(m.group(x)) for x in range(1,5)]
      m = re.search(r'^TIME3:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
      if m: self.tslicesM3 = [ int(m.group(x)) for x in range(1,5)]
      #
      # reference time slice
      m = re.search(r'^TIMEREF:\s*(\d+)', line)
      if m: self.bfepar.treset = basicpar.reset_frame = int(m.group(1))

      # reference pixel subtraction
      m = re.search(r'^REF\s+OFF', line)
      if m: self.fullref = False

      # sensitivity spread cut
      m = re.search(r'^SPREAD:\s*(\S+)', line)
      if m: self.sensitivity_spread_cut = float(m.group(1))

      # variance parameters
      m = re.search(r'^QUANTILE:\s*(\S+)', line)
      if m: self.basicpar.g_ptile = float(m.group(1))
      # correlation parameters
      m = re.search(r'^EPSILON:\s*(\S+)', line)
      if m: self.bfepar.epsilon = basicpar.epsilon = float(m.group(1))
      m = re.search(r'^IPCSUB:\s*(\S+)', line)
      if m: self.basicpar.leadtrailSub = m.group(1).lower() in ['true', 'yes']

      # Other parameters
      m = re.search(r'^DETECTOR:\s*(\S+)', line)
      if m: self.mydet = m.group(1)
      m = re.search(r'^COLOR:\s*(\S+)', line)
      if m: self.use_cmap = m.group(1)

      # Classical non-linearity
      m = re.search(r'^NLPOLY:\s*(\S+)\s+(\S+)\s+(\S+)', line)
      if m:
        self.p_order = int(m.group(1))
        self.nlfit_ts = int(m.group(2))
        self.nlfit_te = int(m.group(3))

      m = re.search(r'^FULLNL:\s*(\S+)\s+(\S+)\s+(\S+)', line)
      if m:
        self.basicpar.fullnl = m.group(1).lower() in ['true', 'yes']
        self.bfepar.fullnl = m.group(2).lower() in ['true', 'yes']
        self.basicpar.use_allorder = m.group(3).lower() in ['true', 'yes']

      # Hot pixels (adu min, adu max, cut stability, cut isolation)
      m = re.search(r'^HOTPIX:\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
      if m:
        self.hotpix = True
        self.hotpix_ADU_range = [ float(m.group(x)) for x in range(1,5)]
      #
      # change reference for hot pixels from last point to autocorr
      m = re.search(r'^HOTREF\s+AUTOCORR', line)
      if m: self.ref_for_hotpix_is_autocorr = True
      # log spacing for times?
      m = re.search(r'^HOTPIX\s+LOGTSPACE', line)
      if m: self.hotpix_logtspace = True
      # sliding median alpha method?
      m = re.search(r'^HOTPIX\s+SLIDEMED', line)
      if m: self.hotpix_slidemed = True

      # Mask regions by hand
      m = re.search(r'^MASK:\s*(\d+)\s+(\d+)', line)
      if m:
        self.maskX.append(int(m.group(1)))
        self.maskY.append(int(m.group(2)))

      # Control figures
      m = re.search(r'^NARROWFIG', line)
        if m: self.narrowfig = True

      # set up array size parameters
      self.swi.addbfe(self.s_bfe)
      self.swi.addhnl(self.p_order)
      print ('Number of output field per superpixel =', self.swi.N)

      # Check number of slices available
      NTMAX = 16384
      for f in lightfiles+darkfiles:
        nt = pyirc.get_num_slices(formatpars, f)
        if nt<NTMAX: NTMAX=nt
      self.NTMAX = NTMAX

      # Copy basicpar parameters to bfebar
      self.bfepar.use_allorder = self.basicpar.use_allorder

    # Checks
    if len(lightfiles)!=len(darkfiles) or len(lightfiles)<2:
        raise ValueError('Failed: {:d} light files and {:d} dark files'.format(len(lightfiles), len(darkfiles)))

    if verbose:
      print ('Output will be directed to {:s}*'.format(outstem))
      print ('Light files:', lightfiles)
      print ('Dark files:', darkfiles)
      print ('Time slices:', tslices, 'max=',NTMAX)
      print ('Mask regions:', maskX, maskY)

    # Additional parameters Size of a block
    self.N = pyirc.get_nside(self.formatpars)
    # Side lengths
    self.dx = self.N//self.nx
    self.dy = self.N//self.ny
    # Pixels in a block
    self.npix = self.dx*self.dy

    # Initializations
    self.full_info = numpy.zeros((ny,nx,self.swi.N))
    self.is_good = numpy.zeros((ny,nx))

  def fit_parameters(self, verbose=False):
    """
    Build general parameters (gain, IPC, NL) and BFE Method 1.

    Parameters
    ----------
    verbose : bool
        Whether to talk a lot.

    Returns
    -------
    None

    """

    # Make table of reference pixel corrections for Method 1
    if self.fullref:
      self.lightref = pyirc.ref_array(self.lightfiles, self.formatpars, self.ny, self.tslices, False)
      self.darkref = pyirc.ref_array(self.lightfiles, self.formatpars, self.ny, self.tslices, False)
    else:
      self.lightref = numpy.zeros((len(lightfiles), ny, 2*len(tslices)+1))
      self.darkref = numpy.zeros((len(darkfiles), ny, 2*len(tslices)+1))
    self.basicpar.subtr_href = self.fullref

    if self.p_order>0:
      # now coefficients for the info table note that in 'abs' mode, the full_info[:,:,0] grid is not actually used, it
      #   is just there for consistency of the format I moved this up here since we want to have these coefficients before the main program runs
      nlcubeX, nlfitX, nlderX, pcoefX = pyirc.gen_nl_cube(self.lightfiles, self.formatpars, [self.basicpar.reset_frame, self.nlfit_ts, self.nlfit_te], [self.ny, self.nx],
        self.full_info[:,:,0], 'abs', False)
      # fill in
      for iy in range(self.ny):
        for ix in range(self.nx):
          if pcoefX[1,iy,ix]!=0:
            full_info[iy,ix,self.swi.Nbb] = -pcoefX[0,iy,ix]/pcoefX[1,iy,ix]
            for ooo in range(2,self.swi.p+1):
              full_info[iy,ix,self.swi.Nbb+ooo-1] = pcoefX[ooo,iy,ix]/pcoefX[1,iy,ix]**ooo
          else:
            self.full_info[iy,ix,pyirc.swi.Nbb] = -1e49 # error code

    # Detector characterization data in a cube (basic characterization + BFE Method 1) Stdout calls are a progress indicator
    #
    if verbose:
      print ('Method 1, progress of calculation:')
      sys.stdout.write('|')
      for iy in range(self.ny): sys.stdout.write(' ')
      print ('| <- 100%')
      sys.stdout.write('|')
    for iy in range(self.ny):
      if verbose:
        sys.stdout.write('*'); sys.stdout.flush()
      for ix in range(self.nx):
        region_cube = pyirc.pixel_data(self.lightfiles, self.formatpars, [self.dx*ix, self.dx*(ix+1), self.dy*iy, self.dy*(iy+1)], self.tslices,
                  [self.sensitivity_spread_cut, True], False)
        dark_cube = pyirc.pixel_data(self.darkfiles, self.formatpars, [self.dx*ix, self.dx*(ix+1), self.dy*iy, self.dy*(iy+1)], self.tslices,
                  [self.sensitivity_spread_cut, False], False)
        info = pyirc.basic(region_cube, dark_cube, self.tslices, self.lightref[:,iy,:], self.darkref[:,iy,:], self.basicpar, False)
        if len(info)>0:
          self.is_good[iy,ix] = 1
          thisinfo = info.copy()
          if basicpar.use_allorder:
            thisinfo[self.swi.beta] = self.full_info[iy,ix,self.swi.Nbb+1:self.swi.Nbb+self.swi.p]
          if self.mychar.lower()=='advanced':
            boxpar = [self.dx*ix, self.dx*(ix+1), self.dy*iy, self.dy*(iy+1)]
            tpar_polychar = [self.tslices[0], self.tslices[-1]+1, self.tchar1, self.tchar2]
            corrstats_data = pyirc.corrstats(self.lightfiles, self.darkfiles, self.formatpars, boxpar,
                         tpar_polychar+[1], self.sensitivity_spread_cut, self.basicpar)
            for iCycle in range(self.ncycle):
              bfeCoefs = pyirc.bfe(region_cube, self.tslices, thisinfo, self.bfepar, False)
              if numpy.isnan(bfeCoefs).any():
                bfeCoefs = numpy.zeros((2*self.swi.s+1,2*self.swi.s+1))
                self.is_good[iy,ix] = 0
              Cdata = pyirc.polychar(self.lightfiles, self.darkfiles, self.formatpars, boxpar,
                 tpar_polychar, self.sensitivity_spread_cut, self.basicpar,
                 [self.ipnltype, bfeCoefs, thisinfo[self.swi.beta]], corrstats_data = corrstats_data)
              info[self.swi.ind1:self.swi.ind2] = numpy.asarray(Cdata[self.swi.indp1:self.swi.indp2])
              thisinfo = info.copy()
              if self.basicpar.use_allorder:
                thisinfo[self.swi.beta] = self.full_info[iy,ix,self.swi.Nbb+1:self.swi.Nbb+self.swi.p]
          bfeCoefs = pyirc.bfe(region_cube, self.tslices, thisinfo, self.bfepar, False)
          if numpy.isnan(bfeCoefs).any():
            bfeCoefs = numpy.zeros((2*self.swi.s+1,2*self.swi.s+1))
            self.is_good[iy,ix] = 0
            info += bfeCoefs[0:2*self.swi.s+1,0:2*self.swi.s+1].flatten().tolist()
          else:
            info = numpy.zeros((self.swi.Nbb)).tolist()

        # save the information, putting in 0's if the super-pixel is not good.
        if len(info)==self.swi.Nbb:
          self.full_info[iy,ix,:self.swi.Nbb] = numpy.array(info)
        if info[0]<self.npix*self.critfrac:
          self.is_good[iy,ix] = 0
          self.full_info[iy,ix,1:] = 0 # wipe out this super-pixel

    if verbose: print ('|')

    # Mask regions
    for mask_index in range(len(self.maskX)):
      ix = self.maskX[mask_index]
      iy = self.maskY[mask_index]
      self.is_good[iy,ix] = 0
      self.full_info[iy,ix,:] = 0 # wipe out this super-pixel

    # if a pixel was set to not good for any other reason
    for iy in range(self.ny):
      for ix in range(self.nx):
        if self.is_good[iy,ix]<.5:
          self.full_info[iy,ix,:] = 0 # wipe out this super-pixel

    self.mean_full_info = numpy.mean(numpy.mean(self.full_info, axis=0), axis=0)/numpy.mean(self.is_good)
    self.std_full_info = numpy.sqrt(numpy.mean(numpy.mean(self.full_info**2, axis=0), axis=0)/numpy.mean(self.is_good) - self.mean_full_info**2)
    if verbose:
      print (self.full_info.shape)
      print ('Number of good regions =', numpy.sum(self.is_good))
      print ('Mean info from good regions =', self.mean_full_info)
      print ('Stdv info from good regions =', self.std_full_info)
      print ('')

  def generate_nonlinearity(self, write_to_file=True):
    """
    Generates non-linearity data.

    Parameters
    ----------
    write_to_file : bool, optional
        Write the non-linearity table to a file as well?

    Returns
    -------
    None

    """

    # Non-linearity cube
    ntSub = self.tslices[-1]
    nlcube, self.nlfit, self.nlder = pyirc.gen_nl_cube(self.lightfiles, self.formatpars, ntSub, [self.ny,self.nx],
      self.full_info[:,:,self.swi.beta]*full_info[:,:,self.swi.I], 'dev', False)
    if write_to_file:
      thisOut = open(self.outstem+'_nl.txt', 'w')
      for iy in range(ny):
        for ix in range(nx):
          thisOut.write('{:3d} {:3d} {:1d} {:9.6f} {:9.6f}'.format(iy,ix,int(is_good[iy,ix]),
            full_info[iy,ix,pyirc.swi.beta]*full_info[iy,ix,pyirc.swi.g]*1e6, full_info[iy,ix,pyirc.swi.g]))
          for it in range(ntSub):
            thisOut.write(' {:7.1f}'.format(nlcube[it,iy,ix]))
          thisOut.write('\n')
      thisOut.close()

  def write_basic_figure(self):
    """Makes a PDF figure of the SCA parameter maps."""

    # these show up so much let's save them
    dx = self.dx; dy = self.dy

    # Multi-panel figure showing basic characterization
    ar = float(self.nx/self.ny)
    spr = 2.2
    matplotlib.rcParams.update({'font.size': 8})
    F = plt.figure(figsize=(7,9))
    S = F.add_subplot(3,2,1)
    S.set_title(r'Good pixel map (%)')
    S.set_xlabel('Super pixel X/{:d}'.format(dx))
    S.set_ylabel('Super pixel Y/{:d}'.format(dy))
    im = S.imshow(full_info[:,:,0]*100/self.npix, cmap=self.use_cmap, aspect=ar, interpolation='nearest', origin='lower',
      vmin=100*self.critfrac, vmax=100)
    F.colorbar(im, orientation='vertical')
    S = F.add_subplot(3,2,2)
    S.set_title(r'Gain map $g$ (e/DN)')
    S.set_xlabel('Super pixel X/{:d}'.format(dx))
    S.set_ylabel('Super pixel Y/{:d}'.format(dy))
    svmin, svmax = pyirc.get_vmin_vmax(self.full_info[:,:,self.swi.g], spr)
    im = S.imshow(full_info[:,:,self.swi.g], cmap=self.use_cmap, aspect=ar, interpolation='nearest', origin='lower',
      vmin=svmin, vmax=svmax)
    F.colorbar(im, orientation='vertical')
    S = F.add_subplot(3,2,3)
    S.set_title(r'IPC map $\alpha$ (%)')
    S.set_xlabel('Super pixel X/{:d}'.format(dx))
    S.set_ylabel('Super pixel Y/{:d}'.format(dy))
    svmin, svmax = pyirc.get_vmin_vmax((self.full_info[:,:,self.swi.alphaH]+full_info[:,:,self.swi.alphaV])/2.*100., spr)
    im = S.imshow((self.full_info[:,:,self.swi.alphaH]+self.full_info[:,:,self.swi.alphaV])/2.*100., cmap=self.use_cmap, aspect=ar,
      interpolation='nearest', origin='lower', vmin=svmin, vmax=svmax)
    F.colorbar(im, orientation='vertical')
    S = F.add_subplot(3,2,4)
    S.set_title(r'Non-linearity map $\beta$ (ppm/e)')
    S.set_xlabel('Super pixel X/{:d}'.format(dx))
    S.set_ylabel('Super pixel Y/{:d}'.format(dy))
    svmin, svmax = pyirc.get_vmin_vmax(self.full_info[:,:,self.swi.beta]*1e6, spr)
    im = S.imshow(self.full_info[:,:,self.swi.beta]*1e6, cmap=self.use_cmap, aspect=ar, interpolation='nearest', origin='lower',
      vmin=svmin, vmax=svmax)
    F.colorbar(im, orientation='vertical')
    S = F.add_subplot(3,2,5)
    S.set_title(r'Charge $It_{n,n+1}$ (e):')
    S.set_xlabel('Super pixel X/{:d}'.format(dx))
    S.set_ylabel('Super pixel Y/{:d}'.format(dy))
    svmin, svmax = pyirc.get_vmin_vmax(self.full_info[:,:,self.swi.I], spr)
    im = S.imshow(self.full_info[:,:,self.swi.I], cmap=self.use_cmap, aspect=ar, interpolation='nearest', origin='lower',
      vmin=svmin, vmax=svmax)
    F.colorbar(im, orientation='vertical')
    S = F.add_subplot(3,2,6)
    S.set_title(r'IPNL $[K^2a+KK^\prime]_{0,0}$ (ppm/e):')
    S.set_xlabel('Super pixel X/{:d}'.format(dx))
    S.set_ylabel('Super pixel Y/{:d}'.format(dy))
    svmin, svmax = pyirc.get_vmin_vmax(self.full_info[:,:,self.swi.ker0]*1e6, spr)
    im = S.imshow(self.full_info[:,:,self.swi.ker0]*1e6, cmap=self.use_cmap, aspect=ar, interpolation='nearest', origin='lower',
      vmin=svmin, vmax=svmax)
    F.colorbar(im, orientation='vertical')
    F.set_tight_layout(True)
    F.savefig(outstem+'_multi.pdf')
    plt.close(F)

  def alt_methods(self, verbose=False):
    """
    Methods 2a, 2b, and 3 trend computation.

    Parameters
    ----------
    verbose : bool, optional
        Whether to print a lot to the output.

    Returns
    -------
    None

    """

    # Method 2a
    #
    self.used_2a = False
    if len(self.tslicesM2a)!=4 or self.tslicesM2a[-1]<=self.tslicesM2a[-2]:
      if verbose:
        print ('Error: tslicesM2a =',tslicesM2a,'does not have length 4 or has insufficient span.')
        print ('Skipping Method 2a ...')
    else:
      # Proceed to implement Method 2a
      self.used_2a = True
      if verbose:
        print ('Alt. time slices (Method 2a): ', self.tslicesM2a)
      self.tfmin = self.tslicesM2a[2]; self.tfmax = self.tslicesM2a[3]
      self.ntM2a = tfmax-tfmin+1
      if verbose:
        print ('Method 2a, progress of calculation:')
        sys.stdout.write('|')
        for iy in range(ny): sys.stdout.write(' ')
        print ('| <- 100%')
        sys.stdout.write('|')
      self.Method2a_slopes = numpy.zeros((self.ny,self.nx))
      self.Method2a_vals = numpy.zeros((self.ny,self.nx,self.ntM2a))
      lngraw = numpy.zeros((self.ntM2a))
      for iy in range(self.ny):
        if verbose:
          sys.stdout.write('*'); sys.stdout.flush()
        for ix in range(self.nx):
          if self.is_good[iy,ix]==1:
            for t in range(self.ntM2a):
              temp_tslices = [self.tslicesM2a[0], self.tslicesM2a[1], self.tslicesM2a[1], self.tfmin+t]
              if self.fullref:
                lightref = pyirc.ref_array_onerow(self.lightfiles, self.formatpars, iy, self.ny, temp_tslices, False)
                darkref = pyirc.ref_array_onerow(self.darkfiles, self.formatpars, iy, self.ny, temp_tslices, False)
              region_cube = pyirc.pixel_data(self.lightfiles, self.formatpars, [self.dx*ix, self.dx*(ix+1), self.dy*iy, self.dy*(iy+1)], temp_tslices,
                        [self.sensitivity_spread_cut, True], False)
              dark_cube = pyirc.pixel_data(self.darkfiles, self.formatpars, [self.dx*ix, self.dx*(ix+1), self.dy*iy, self.dy*(iy+1)], temp_tslices,
                        [self.sensitivity_spread_cut, False], False)
              info = pyirc.basic(region_cube, dark_cube, temp_tslices, lightref[:,iy,:], darkref[:,iy,:], self.basicpar, False)
              self.Method2a_vals[iy,ix,t] = lngraw[t] = numpy.log(info[1])
            # Build least squares fit
            mS, cS = numpy.linalg.lstsq(numpy.vstack([numpy.array(range(self.ntM2a)), numpy.ones(self.ntM2a)]).T, lngraw, rcond=-1)[0]
            self.Method2a_slopes[iy,ix] = mS/self.full_info[iy,ix,self.swi.I]
      if verbose:
        print ('|')
        print ('Mean slope d[ln graw]/d[I td] at fixed ta,tb =', numpy.mean(self.is_good*self.Method2a_slopes)/numpy.mean(self.is_good))
        print ('')
      # Predicted slopes
      self.slope_2a_BFE = 3*self.mean_full_info[self.swi.beta] - (1+4*self.mean_full_info[self.swi.alphaH]
                 +4*self.mean_full_info[self.swi.alphaV])*self.mean_full_info[self.swi.ker0]
      self.slope_2a_NLIPC = 3*self.mean_full_info[self.swi.beta] - 2*(1+4*self.mean_full_info[self.swi.alphaH]
                   +4*self.mean_full_info[self.swi.alphaV])*self.mean_full_info[self.swi.ker0]

    # Method 2b
    #
    self.used_2b = False
    if len(self.tslicesM2b)!=4 or self.tslicesM2b[-1]<=self.tslicesM2b[-2]:
      print ('Error: tslicesM2b =', self.tslicesM2b,'does not have length 4 or has insufficient span.')
      print ('Skipping Method 2b ...')
    else:
      # Proceed to implement Method 2b
      self.used_2b = True
      if verbose:
        print ('Alt. time slices (Method 2b): ',self.tslicesM2b)
      self.tfminB = self.tslicesM2b[2]; self.tfmaxB = self.tslicesM2b[3]
      self.ntM2b = self.tfmaxB-self.tfminB+1
      if verbose:
        print ('Method 2b, progress of calculation:')
        sys.stdout.write('|')
        for iy in range(self.ny): sys.stdout.write(' ')
        print ('| <- 100%')
        sys.stdout.write('|')
      self.Method2b_slopes = numpy.zeros((self.ny,self.nx))
      self.Method2b_vals = numpy.zeros((self.ny,self.nx,self.ntM2b))
      lngraw = numpy.zeros((self.ntM2b))
      for iy in range(self.ny):
        if verbose:
          sys.stdout.write('*'); sys.stdout.flush()
        for ix in range(self.nx):
          if self.is_good[iy,ix]==1:
            for t in range(self.ntM2b):
              temp_tslices = [self.tslicesM2b[0]+t, self.tslicesM2b[1]+t, self.tslicesM2b[1]+t, self.tslicesM2b[2]+t]
              if self.fullref:
                lightref = pyirc.ref_array_onerow(self.lightfiles, self.formatpars, iy, self.ny, temp_tslices, False)
                darkref = pyirc.ref_array_onerow(self.darkfiles, self.formatpars, iy, self.ny, temp_tslices, False)
              region_cube = pyirc.pixel_data(self.lightfiles, self.formatpars, [self.dx*ix, self.dx*(ix+1), self.dy*iy, self.dy*(iy+1)], temp_tslices,
                        [self.sensitivity_spread_cut, True], False)
              dark_cube = pyirc.pixel_data(self.darkfiles, self.formatpars, [self.dx*ix, self.dx*(ix+1), self.dy*iy, self.dy*(iy+1)], temp_tslices,
                        [self.sensitivity_spread_cut, False], False)
              info = pyirc.basic(region_cube, dark_cube, temp_tslices, lightref[:,iy,:], darkref[:,iy,:], self.basicpar, False)
              self.Method2b_vals[iy,ix,t] = lngraw[t] = numpy.log(info[1])
            # Build least squares fit
            mS, cS = numpy.linalg.lstsq(numpy.vstack([numpy.array(range(self.ntM2b)), numpy.ones(self.ntM2b)]).T, lngraw, rcond=-1)[0]
            self.Method2b_slopes[iy,ix] = mS/full_info[iy,ix,sef.swi.I]
      if verbose:
        print ('|')
        print ('Mean slope d[ln graw]/d[I tb] at fixed tab,tad =', numpy.mean(self.is_good*self.Method2b_slopes)/numpy.mean(self.is_good))
        print ('')
      # Predicted slopes
      self.slope_2b_BFE = 2*self.mean_full_info[self.swi.beta]
      self.slope_2b_NLIPC = 2*self.mean_full_info[self.swi.beta] + 2*(1+4*self.mean_full_info[self.swi.alphaH]
                   +4*self.mean_full_info[self.swi.alphaV])*self.mean_full_info[self.swi.ker0]

    # Method 3
    #
    self.used_3 = False
    if len(self.tslicesM3)!=4 or self.tslicesM3[-1]<=self.tslicesM3[-2]:
      if verbose:
        print ('Error: tslicesM3 =',self.tslicesM3,'does not have length 4 or has insufficient span.')
        print ('Skipping Method 3 ...')
    else:
      # Proceed to implement Method 3
      self.used_3 = True
      if verbose: print ('Alt. time slices (Method 3): ', self.tslicesM3)
      self.tfmin3 = self.tslicesM3[2]; self.tfmax3 = self.tslicesM3[3]
      self.ntM3 = self.tfmax3-self.tfmin3+1
      if verbose:
        print ('Method 3, progress of calculation:')
        sys.stdout.write('|')
        for iy in range(self.ny): sys.stdout.write(' ')
        print ('| <- 100%')
        sys.stdout.write('|')
      self.Method3_slopes = numpy.zeros((self.ny,self.nx))
      self.Method3_vals = numpy.zeros((self.ny,self.nx,self.ntM3))
      # Method3_alphas = numpy.zeros((self.ny,self.nx,self.ntM3))
      CCraw = numpy.zeros((self.ntM3))
      for iy in range(self.ny):
        if verbose:
          sys.stdout.write('*'); sys.stdout.flush()
        for ix in range(self.nx):
          if self.is_good[iy,ix]==1:
            for t in range(self.ntM3):
              temp_tslices = [self.tslicesM3[0], self.tslicesM3[1], self.tslicesM3[0], self.tfmin3+t]
              if self.fullref:
                lightref = pyirc.ref_array_onerow(self.lightfiles, self.formatpars, iy, self.ny, temp_tslices, False)
                darkref = pyirc.ref_array_onerow(self.darkfiles, self.formatpars, iy, self.ny, temp_tslices, False)
              region_cube = pyirc.pixel_data(self.lightfiles, self.formatpars, [self.dx*ix, self.dx*(ix+1), self.dy*iy, self.dy*(iy+1)], temp_tslices,
                        [self.sensitivity_spread_cut, True], False)
              dark_cube = pyirc.pixel_data(self.darkfiles, self.formatpars, [self.dx*ix, self.dx*(ix+1), self.dy*iy, self.dy*(iy+1)], temp_tslices,
                        [self.sensitivity_spread_cut, False], False)
              info = pyirc.basic(region_cube, dark_cube, temp_tslices, lightref[:,iy,:], darkref[:,iy,:], self.basicpar, False)
              self.Method3_vals[iy,ix,t] = CCraw[t] = (info[self.swi.tCH]+info[self.swi.tCV])/2.*self.full_info[iy,ix,self.swi.g]**2\
                      /(self.full_info[iy,ix,self.swi.I]*(temp_tslices[-1]-temp_tslices[0]))
              # Method3_alphas[iy,ix,t] = (info[pyirc.swi.alphaH]+info[pyirc.swi.alphaV])/2.
            # Build least squares fit
            mS, cS = numpy.linalg.lstsq(numpy.vstack([numpy.array(range(self.ntM3)), numpy.ones(self.ntM3)]).T, CCraw, rcond=-1)[0]
            self.Method3_slopes[iy,ix] = mS/full_info[iy,ix,self.swi.I]
      if verbose:
        print ('|')
        print ('Mean slope d[g^2/(Itad) Cadj,ad]/d[I td] at fixed ta,tb =', numpy.mean(self.is_good*self.Method3_slopes)/numpy.mean(self.is_good))
        print ('')
      # Predicted slopes
      ave = (self.mean_full_info[self.swi.ker0-1]+self.mean_full_info[self.swi.ker0+1]+self.mean_full_info[self.swi.ker0-(2*self.swi.s+1)]
            +self.mean_full_info[self.swi.ker0+(2*self.swi.s+1)])/4.
      self.slope_3_beta = -4*(self.mean_full_info[self.swi.alphaH]+self.mean_full_info[self.swi.alphaV])*self.mean_full_info[self.swi.beta]
      self.slope_3_BFE = -4*(self.mean_full_info[self.swi.alphaH]+self.mean_full_info[self.swi.alphaV])*self.mean_full_info[self.swi.beta] + ave
      self.slope_3_NLIPC = -4*(self.mean_full_info[self.swi.alphaH]+self.mean_full_info[self.swi.alphaV])*self.mean_full_info[self.swi.beta] + ave*2.

    # Non-linearity corrections, Methods 2 and 3:
    if verbose:
      print ('Non-linearity correction tables:')
    if self.used_2a:
      if verbose:
        print ('2a:')
      vec = []
      for t in range(self.tslicesM2a[2], self.tslicesM2a[3]+1):
        offsets = pyirc.compute_gain_corr_many(self.nlfit, self.nlder, self.full_info[:,:,self.swi.I]*self.full_info[:,:,self.swi.beta],
                  [self.tslicesM2a[0],self.tslicesM2a[1],t], self.basicpar.reset_frame, self.is_good)
        if verbose:
          print (t, numpy.mean(self.offsets*self.is_good)/numpy.mean(self.is_good))
        vec += [numpy.mean(self.offsets*self.is_good)/numpy.mean(self.is_good)]
      self.PV2a = max(vec)-min(vec)
      if verbose:
        print ('PV: ', PV2a)
    if used_2b:
      if verbose:
        print ('2b:')
      vec = []
      dt1 = self.tslicesM2b[1] - self.tslicesM2b[0]
      dt2 = self.tslicesM2b[2] - self.tslicesM2b[0]
      for t in range(self.tslicesM2b[0], self.tslicesM2b[3]-self.tslicesM2b[2]+1):
        offsets = pyirc.compute_gain_corr_many(self.nlfit, self.nlder, self.full_info[:,:,self.swi.I]*self.full_info[:,:,self.swi.beta],
                  [t,t+self.dt1,t+self.dt2], self.basicpar.reset_frame, self.is_good)
        if verbose:
          print (t, numpy.mean(offsets*self.is_good)/numpy.mean(self.is_good))
        vec += [numpy.mean(offsets*self.is_good)/numpy.mean(self.is_good)]
      self.PV2b = max(vec)-min(vec)
      if verbose:
        print ('PV: ', PV2b)
    if used_3:
      if verbose:
        print ('3:')
      vec = []
      for t in range(self.tslicesM3[2], self.tslicesM3[3]+1):
        offsets = pyirc.compute_xc_corr_many(self.nlfit, self.nlder, self.full_info[:,:,self.swi.I]*self.full_info[:,:,self.swi.beta],
                 [self.tslicesM3[0],t], self.basicpar.reset_frame, self.is_good)
        alpha3 = (full_info[:,:,pyirc.swi.alphaH]+full_info[:,:,pyirc.swi.alphaV])/2.
        offsets *= 2. * alpha3 * (1.-4*alpha3)
        if verbose:
          print (t, numpy.mean(self.offsets*self.is_good)/numpy.mean(self.is_good))
        vec += [numpy.mean(self.offsets*self.is_good)/numpy.mean(self.is_good)]
      self.PV3 = max(vec)-min(vec)
      if verbose:
        print ('PV: ', PV3)
        print ('Method 3 implied slopes =', self.slope_3_beta, self.slope_3_BFE, self.slope_3_NLIPC)
        print ('')

  def method_23_plot(self):
    """
    Method 2 and 3 characterization Multi-panel figure showing basic characterization.
    """

    matplotlib.rcParams.update({'font.size': 8})
    F = plt.figure(figsize=(3.5,9))
    if used_2a:
      S = F.add_subplot(3,1,1)
      S.set_title(r'Raw gain vs. interval duration')
      S.set_xlabel(r'Signal level $It_{'+'{:d}'.format(self.tslicesM2a[0])+r',d}$ [ke]')
      S.set_ylabel(r'$\ln g^{\rm raw}_{' +'{:d},{:d}'.format(self.tslicesM2a[0],self.tslicesM2a[1]) +r',d}$')
      SX = [numpy.mean(self.is_good*self.full_info[:,:,self.swi.I]*myt)/numpy.mean(self.is_good)/1.0e3 for myt in range(self.tfmin-self.tslicesM2a[0], self.tfmax+1-self.tslicesM2a[0])]
      SY = [numpy.mean(self.is_good*self.Method2a_vals[:,:,t])/numpy.mean(self.is_good) for t in range(self.ntM2a)]
      SS = [] # std. dev. on the mean
      for t in range(self.ntM2a):
        SS += [ numpy.sqrt((numpy.mean(self.is_good*self.Method2a_vals[:,:,t]**2)/numpy.mean(self.is_good)-SY[t]**2)/(numpy.sum(self.is_good)-1)) ]
      xc = numpy.mean(numpy.array(SX))
      yc = numpy.mean(numpy.array(SY))
      S.set_xlim(min(SX)-.05*(max(SX)-min(SX)), max(SX)+.05*(max(SX)-min(SX)))
      xr = numpy.arange(min(SX), max(SX), (max(SX)-min(SX))/256.)
      S.errorbar([xc], [min(SY)], yerr=[self.PV2a/2.], marker=',', color='k', ls='None')
      S.text(xc+.05*(max(SX)-min(SX)), min(SY), 'sys nl', color='k')
      S.errorbar(SX, SY, yerr=SS, marker='x', color='r', ls='None')
      S.plot(xr, yc+(xr-xc)*self.slope_2a_BFE*1e3, 'g--', label='pure BFE')
      S.plot(xr, yc+(xr-xc)*self.slope_2a_NLIPC*1e3, 'b-', label='pure NL-IPC')
      S.legend(loc=2)
    if used_2b:
      S = F.add_subplot(3,1,2)
      S.set_title(r'Raw gain vs. interval center')
      S.set_xlabel(r'Signal level $It_{'+'{:d}'.format(self.tslicesM2b[0])+r',a}$ [ke]')
      S.set_ylabel(r'$\ln g^{\rm raw}_{' +'a,a+{:d},a+{:d}'.format(self.tslicesM2b[1]-self.tslicesM2b[0],self.tslicesM2b[2]-self.tslicesM2b[0]) +r'}$')
      SX = [numpy.mean(self.is_good*self.full_info[:,:,self.swi.I]*myt)/numpy.mean(self.is_good)/1.0e3 for myt in range(self.ntM2b)]
      SY = [numpy.mean(self.is_good*self.Method2b_vals[:,:,t])/numpy.mean(self.is_good) for t in range(self.ntM2b)]
      SS = [] # std. dev. on the mean
      for t in range(self.ntM2b):
        SS += [ numpy.sqrt((numpy.mean(self.is_good*self.Method2b_vals[:,:,t]**2)/numpy.mean(self.is_good)-SY[t]**2)/(numpy.sum(self.is_good)-1)) ]
      xc = numpy.mean(numpy.array(SX))
      yc = numpy.mean(numpy.array(SY))
      S.set_xlim(min(SX)-.05*(max(SX)-min(SX)), max(SX)+.05*(max(SX)-min(SX)))
      xr = numpy.arange(min(SX), max(SX), (max(SX)-min(SX))/256.)
      S.errorbar([xc], [min(SY)], yerr=[self.PV2b/2.], marker=',', color='k', ls='None')
      S.text(xc+.05*(max(SX)-min(SX)), min(SY), 'sys nl', color='k')
      S.errorbar(SX, SY, yerr=SS, marker='x', color='r', ls='None')
      S.plot(xr, yc+(xr-xc)*self.slope_2b_BFE*1e3, 'g--', label='pure BFE')
      S.plot(xr, yc+(xr-xc)*self.slope_2b_NLIPC*1e3, 'b-', label='pure NL-IPC')
      S.legend(loc=2)
    if used_3:
      S = F.add_subplot(3,1,3)
      S.set_title(r'CDS ACF vs. signal')
      S.set_xlabel(r'Signal level $It_{'+'{:d}'.format(self.tslicesM3[0])+r',d}$ [ke]')
      S.set_ylabel(r'$g^2C_{'+'{:d}'.format(self.tslicesM3[0])+r'd'+'{:d}'.format(self.tslicesM3[0])+r'd}(\langle1,0\rangle)/[It_{'\
        +'{:d}'.format(self.tslicesM3[0])+r'd}]$')
      SX = [numpy.mean(self.is_good*self.full_info[:,:,self.swi.I]*myt)/numpy.mean(self.is_good)/1.0e3 for myt in range(self.tfmin3-self.tslicesM3[0], self.tfmax3+1-self.tslicesM3[0])]
      SY = [numpy.mean(self.is_good*self.Method3_vals[:,:,t])/numpy.mean(self.is_good) for t in range(self.ntM3)]
      SS = [] # std. dev. on the mean
      for t in range(self.ntM3):
        SS += [ numpy.sqrt((numpy.mean(self.is_good*self.Method3_vals[:,:,t]**2)/numpy.mean(self.is_good)-SY[t]**2)/(numpy.sum(self.is_good)-1)) ]
      xc = numpy.mean(numpy.array(SX))
      yc = numpy.mean(numpy.array(SY))
      S.set_xlim(min(SX)-.05*(max(SX)-min(SX)), max(SX)+.05*(max(SX)-min(SX)))
      xr = numpy.arange(min(SX), max(SX), (max(SX)-min(SX))/256.)
      S.errorbar([xc], [min(SY)], yerr=[self.PV3], marker=',', color='k', ls='None')
      S.text(xc+.05*(max(SX)-min(SX)), min(SY)+PV3, 'sys nl', color='k')
      S.errorbar(SX, SY, yerr=SS, marker='x', color='r', ls='None')
      S.plot(xr, yc+(xr-xc)*self.slope_3_BFE*1e3, 'g--', label='pure BFE')
      S.plot(xr, yc+(xr-xc)*self.slope_3_NLIPC*1e3, 'b-', label='pure NL-IPC')
      S.plot(xr, yc+(xr-xc)*self.slope_3_beta*1e3, 'k:', label='beta only')
      S.legend(loc=2)
    F.set_tight_layout(True)
    F.savefig(outstem+'_m23.pdf')
    plt.close(F)

  def text_output(self):
    """Generate a text summary.

    Parameters
    ----------
    None

    Returns
    -------
    str
        The output text table.

    """

    # Text output
    thisOut = ''

    # Print information in the file header
    thisOut += '# This summary created at {:s}\n'.format(time.asctime(time.localtime(time.time())))
    thisOut += '# Uses pyirc v{:d}\n'.format(pyirc.get_version())
    thisOut += '# Detector: '+self.mydet+'\n'
    thisOut += '#\n# Files used:\n'
    thisOut += '# Light:\n'
    for f in self.lightfiles: thisOut += '#   {:s}\n'.format(f)
    thisOut += '# Dark:\n'
    for f in self.darkfiles: thisOut += '#   {:s}\n'.format(f)
    thisOut += '# Input format {:d}\n'.format(self.formatpars)
    thisOut += '# Time slices:'
    for t in self.tslices: thisOut += ' {:3d}'.format(t)
    thisOut += '\n'
    thisOut += '# Mask: ' + str(self.maskX) + ',' + str(self.maskY) + '\n'
    thisOut += '#\n'
    thisOut += '# Cut on good pixels {:7.4f}% deviation from median\n'.format(100*self.sensitivity_spread_cut)
    thisOut += '# Dimensions: {:3d}(x) x {:3d}(y) super-pixels, {:4d} good\n'.format(self.nx,self.ny,int(numpy.sum(self.is_good)))
    thisOut += '# Frame number corresponding to reset: {:d}\n'.format(self.basicpar.reset_frame)
    thisOut += '# Reference pixel subtraction for linearity: {:s}\n'.format(str(self.fullref))
    thisOut += '# Quantile for variance computation = {:9.6f}%\n'.format(self.basicpar.g_ptile)
    thisOut += '# Clipping fraction epsilon = {:9.7f}\n'.format(self.basicpar.epsilon)
    thisOut += '# Lead-trail subtraction for IPC correlation = ' + str(basicpar.leadtrailSub) + '\n'
    thisOut += '# Characterization type: '+self.mychar+'\n'
    if self.mychar.lower()=='advanced':
      thisOut += '#   dt = {:d},{:d}, ncycle = {:d}\n'.format(self.tchar1,self.tchar2,self.ncycle)
    thisOut += '# Non-linearity settings: basicpar.fullnl={:s} bfepar.fullnl={:s} basicpar.use_allorder={:s}\n'.format(
      str(self.basicpar.fullnl), str(self.bfepar.fullnl), str(self.basicpar.use_allorder) )
    thisOut += '# BFE Method 1\n#   Baseline subtraction = {:s}\n'.format(str(self.bfepar.blsub))
    thisOut += '# BFE Method 2a\n#   Enabled = {:s}\n'.format(str(self.used_2a))
    thisOut += '# BFE Method 2b\n#   Enabled = {:s}\n'.format(str(self.used_2b))
    thisOut += '# BFE Method 3\n#   Enabled = {:s}\n'.format(str(self.used_3))
    thisOut += '# Hot pixels\n#   Enabled = {:s}\n'.format(str(self.hotpix))
    if self.hotpix:
      thisOut += '#   Parameters = {:s}\n'.format(str(self.hotpix_ADU_range))
      if self.ref_for_hotpix_is_autocorr:
        thisOut += '#   ref for delta alpha = autocorr\n'
      else:
        thisOut += '#   ref for delta alpha = last frame used\n'
      if self.hotpix_slidemed:
        thisOut += '#   median method = sliding\n'
      else:
        thisOut += '#   median method = normal\n'
    thisOut += '# Associated figures:\n'
    thisOut += '#   {:s}\n'.format(self.outstem+'_multi.pdf')
    thisOut += '#   {:s}\n'.format(self.outstem+'_m23.pdf')
    thisOut += '#   {:s}\n'.format(self.outstem+'_hotipc.pdf')
    thisOut += '#\n'
    thisOut += '# Columns:\n'
    col=1
    thisOut += '# {:3d}, X (super pixel grid)\n'.format(col)
    col+=1
    thisOut += '# {:3d}, Y (super pixel grid)\n'.format(col)
    col+=1
    thisOut += '# {:3d}, number of good pixels\n'.format(col)
    col+=1
    thisOut += '# {:3d}, raw gain (e/DN)\n'.format(col)
    col+=1
    thisOut += '# {:3d}, alpha-corrected gain (e/DN)\n'.format(col)
    col+=1
    thisOut += '# {:3d}, alpha,beta-corrected gain (e/DN)\n'.format(col)
    col+=1
    thisOut += '# {:3d}, IPC alpha horizontal\n'.format(col)
    col+=1
    thisOut += '# {:3d}, IPC alpha vertical\n'.format(col)
    col+=1
    thisOut += '# {:3d}, nonlinearity beta (e^-1)\n'.format(col)
    col+=1
    thisOut += '# {:3d}, charge per time slice (e)\n'.format(col)
    col+=1
    thisOut += '# {:3d}, IPC alpha diagonal (if computed; otherwise all 0s)\n'.format(col)
    col+=1
    thisOut += '# {:3d}, C_H at slices {:d},{:d} (DN^2)\n'.format(col, self.tslices[0], self.tslices[-1])
    col+=1
    thisOut += '# {:3d}, C_V at slices {:d},{:d} (DN^2)\n'.format(col, self.tslices[0], self.tslices[-1])
    col+=1
    for jb in range(2*self.swi.s+1):
      for ib in range(2*self.swi.s+1):
        thisOut += '# {:3d}, BFE kernel K^2a (+NL-IPC) at ({:2d},{:2d}) (e^-1)\n'.format(col, ib-self.swi.s, jb-self.swi.s)
        col+=1
    if self.swi.p>0:
      thisOut += '# {:3d}, time intercept\n'.format(col)
      col += 1
      for ip in range(2, self.swi.p+1):
        thisOut += '# {:3d}, additional non-linearity coefficient, order {:d} (DN^-{:d})\n'.format(col, ip, ip-1)
        col+=1
    if used_2a:
      thisOut += '# {:3d}, Method 2a slope (e^-1)\n'.format(col)
      col+=1
    if used_2b:
      thisOut += '# {:3d}, Method 2b slope (e^-1)\n'.format(col)
      col+=1
    if used_3:
      thisOut += '# {:3d}, Method 3 slope (e^-1)\n'.format(col)
      col+=1
    thisOut += '#\n'
    # Now make the data table
    for iy in range(self.ny):
      for ix in range(self.nx):
        # Print the column first, then row (normal human-read order, note this is the reverse of internal Python)
        thisOut += '{:3d} {:3d}'.format(ix,iy)
        for col in range(my_dim):
          thisOut += ' {:14.7E}'.format(self.full_info[iy,ix,col])
        if used_2a:
          thisOut += ' {:14.7E}'.format(self.Method2a_slopes[iy,ix])
        if used_2b:
          thisOut += ' {:14.7E}'.format(self.Method2b_slopes[iy,ix])
        if used_3:
          thisOut += ' {:14.7E}'.format(self.Method3_slopes[iy,ix])
        thisOut += '\n'

    return thisOut

  def hotpix_analysis(self, verbose=False):
    """
    Hot pixel analysis.

    Parameters
    ----------
    verbose : bool, optional
        Whether to talk a lot.

    Returns
    -------
    str
        Hot pixel report.

    """

    # exit if the hot pixel analysis is not enabled.
    if not self.hotpix:
      return

    if verbose:
      print ('Start hot pixels ...')
    self.hotY, self.hotX = pyirc.hotpix(self.darkfiles, self.formatpars, range(1,self.NTMAX), self.hotpix_ADU_range, True)
    if verbose:
      print ('Number of pixels selected:', len(self.hotX)) # only printed for de-bugging -> , len(hotY)
    dtstep = 5 # <-- right now this is hard coded
    self.htsteps = range(1,self.NTMAX,dtstep)
    if self.hotpix_logtspace:
      self.htsteps = [1]
      for k in range(1,12):
        if 2**k<NTMAX-1: self.htsteps += [2**k]
        if k>=2:
          if 5*2**(k-2)<self.NTMAX-1: self.htsteps += [5*2**(k-2)]
        if 3*2**(k-1)<self.NTMAX-1: self.htsteps += [3*2**(k-1)]
        if k>=2:
          if 7*2**(k-2)<self.NTMAX-1: self.htsteps += [7*2**(k-2)]
      self.htsteps += [self.NTMAX-1]
    beta_gain = self.full_info[:,:,self.swi.beta]*self.full_info[:,:,self.swi.g]
    if verbose: print (beta_gain)
    hotcube = pyirc.hotpix_ipc(self.hotY, self.hotX, self.darkfiles, self.formatpars, htsteps, [beta_gain, False], True)
    nhstep = len(self.htsteps)
    if verbose: print ('number of time steps ->', nhstep)
    fromcorr_alpha = numpy.zeros((len(self.hotX)))
    hotpix_alpha = numpy.zeros((len(self.hotX), nhstep))
    hotpix_alpha_num = numpy.zeros((len(self.hotX), nhstep))
    hotpix_alpha_den = numpy.zeros((len(self.hotX), nhstep))
    hotpix_alphaD = numpy.zeros((len(self.hotX), nhstep))
    hotpix_signal = numpy.zeros((len(self.hotX), nhstep))
    #
    # generate and write hot pixel information
    thisOut = ""
    for jpix in range(len(self.hotX)):
      iy = self.hotY[jpix]//self.dy
      ix = self.hotX[jpix]//self.dx
      fromcorr_alpha[jpix] = self.full_info[iy,ix,self.swi.alphaH]/2.+self.full_info[iy,ix,self.swi.alphaV]/2.
      thisOut += '{:4d} {:4d} {:8.6f}'.format(self.hotX[jpix], self.hotY[jpix], fromcorr_alpha[jpix])
      for t in range(1,nhstep):
        R = ( numpy.mean(hotcube[jpix,t,1:5]) - hotcube[jpix,t,-1] ) / (hotcube[jpix,t,0]-hotcube[jpix,t,-1] )
        S = ( numpy.mean(hotcube[jpix,t,5:9]) - hotcube[jpix,t,-1] ) / (hotcube[jpix,t,0]-hotcube[jpix,t,-1] )
        hotpix_alpha[jpix, t] = R/(1.+4*R+4*S)
        hotpix_alpha_num[jpix, t] = R
        hotpix_alpha_den[jpix, t] = 1.+4*R+4*S
        hotpix_alphaD[jpix, t] = S/(1.+4*R+4*S)
        hotpix_signal[jpix, t] = hotcube[jpix,t,0]-hotcube[jpix,t,-1]
        thisOut += ' {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f}'.format(hotcube[jpix,t,0], numpy.mean(hotcube[jpix,t,1:5]), hotcube[jpix,t,-1],
          (hotcube[jpix,t,1]+hotcube[jpix,t,3]-hotcube[jpix,t,2]-hotcube[jpix,t,4])/4., numpy.mean(hotcube[jpix,t,5:9]))
      thisOut += '\n'

    # report median levels
    if verbose: print ('IPC relative to nominal (signal, median, uncert):')
    ipcmed_x = numpy.zeros((nhstep))
    ipcmed_y = numpy.zeros((nhstep))
    ipcmed_yerr = numpy.zeros((nhstep))
    delta = .5/numpy.sqrt(len(self.hotX))
    for t in range(1,nhstep):
      my_y = hotpix_alpha[:,t]-hotpix_alpha[:,-1]
      if ref_for_hotpix_is_autocorr: my_y = hotpix_alpha[:,t] - fromcorr_alpha
      ipcmed_x[t] = numpy.nanpercentile(hotpix_signal[:, t], 50.)
      if hotpix_slidemed:
        X_ = hotpix_alpha_den[:,t]
        Y_ = hotpix_alpha_num[:,t]-fromcorr_alpha*hotpix_alpha_den[:,t]
        ipcmed_y[t] = pyirc.slidemed_percentile(X_, Y_, 50)
        ipcmed_yerr[t] = ( pyirc.slidemed_percentile(X_, Y_, 50.+100*delta)
          - pyirc.slidemed_percentile(X_, Y_, 50.-100*delta) ) / 2.
      else:
        ipcmed_y[t] = numpy.nanpercentile(my_y, 50.)
        ipcmed_yerr[t] = (numpy.nanpercentile(my_y, 50.+100*delta)-numpy.nanpercentile(my_y, 50.-100*delta))/2.
      print ('{:10.2f} {:9.6f} {:9.6f}'.format(ipcmed_x[t], ipcmed_y[t], ipcmed_yerr[t]))
    print ('')
    print ('median alphaD:', '{:9.6f} {:9.6f}'.format(numpy.nanpercentile(hotpix_alphaD[:,-1], 50.),
      (numpy.nanpercentile(hotpix_alphaD[:,-1], 50.+100*delta)-numpy.nanpercentile(hotpix_alphaD[:,-1], 50.-100*delta))/2.))

    # bigger grid for IPC comparisons
    NG=4
    grid_alphaCorr = numpy.zeros((NG,NG)); grid_alphaCorrErr = numpy.zeros((NG,NG))
    grid_alphaHot = numpy.zeros((NG,NG)); grid_alphaHotErr = numpy.zeros((NG,NG))
    if self.ny%NG==0 and self.nx%NG==0:
      stepx = self.nx//NG; stepy = self.ny//NG
      sp = self.N//NG;
      for ix in range(NG):
        for iy in range(NG):
          # bin the auto-correlation measurements
          suba = full_info[iy*stepy:(iy+1)*stepy, ix*stepx:(ix+1)*stepx, :]
          mya = (suba[:,:,self.swi.alphaH]+suba[:,:,self.swi.alphaV])/2.
          pmask = suba[:,:,0] > 0
          n = mya[pmask].size
          if n>1:
            grid_alphaCorr[iy,ix] = mya[pmask].mean()
            grid_alphaCorrErr[iy,ix] = mya[pmask].std()/numpy.sqrt(n-1)
          #
          # bin the hot pixel measurements -- use final time slice!
          u = hotpix_alpha[numpy.logical_and(self.hotY//sp==iy, self.hotX//sp==ix), -1]
          if u.size>1:
            grid_alphaHot[iy,ix] = numpy.nanpercentile(u,50)
            delta = .5/numpy.sqrt(u.size)
            grid_alphaHotErr[iy,ix] = (numpy.nanpercentile(u, 50.+100*delta)-numpy.nanpercentile(u, 50.-100*delta))/2.
    if verbose: print (grid_alphaCorr, grid_alphaCorrErr, grid_alphaHot, grid_alphaHotErr)

    # save information
    self.hotpix_signal = hotpix_signal
    self.hotpix_alpha = alpha
    self.ipcmed_x = ipcmed_x
    self.ipcmed_y = ipcmed_y
    self.ipcmed_yerr = ipcmed_yerr
    self.grid_alphaCorr = grid_alphaCorr
    self.grid_alphaCorrErr = grid_alphaCorrErr
    self.grid_alphaHot = grid_alphaHot
    self.grid_alphaHotErr = grid_alphaHotErr

    return thisOut

  def hotpix_plots(self):
    """Makes the hot pixel plots."""

    nhstep = len(self.htsteps)

    # hot pixel plots
    matplotlib.rcParams.update({'font.size': 8})
    if narrowfig:
      F = plt.figure(figsize=(3.5,6))
    else:
      F = plt.figure(figsize=(7,6))
    #
    # hot pixel locations
    if narrowfig:
      S = F.add_subplot(2,1,1)
    else:
      S = F.add_subplot(2,2,1)
    S.set_title('hot pixel locations: '+self.mydet)
    S.set_xlim(0,self.N); S.set_ylim(0,self.N); S.set_aspect('equal')
    S.xaxis.set_ticks(numpy.linspace(0,self.N,num=5)); S.yaxis.set_ticks(numpy.linspace(0,self.N,num=5))
    S.grid(True, color='g', linestyle='-')
    S.scatter(self.hotX+.5, self.hotY+.5, s=3, marker='.', color='r')
    #
    # these two panels only in the full version, not the narrow version
    if not self.narrowfig:
      # hot pixel level
      S = F.add_subplot(2,2,2)
      S.set_xlabel(r'Signal level $S_{1,' + '{:d}'.format(self.htsteps[-1]) +'}$ [DN]')
      S.set_ylabel(r'IPC $\alpha$ [%]')
      SX = self.hotpix_signal[:,-1]
      SY = self.hotpix_alpha[:,-1]/.01
      S.set_title(r'IPC $\alpha$ for hot pixels')
      S.set_xlim(.95*(self.htsteps[-1]-1)/(self.NTMAX-1.0)*self.hotpix_ADU_range[0], 1.05*self.hotpix_ADU_range[1])
      S.set_ylim(0,4.)
      S.grid(True, color='g', linestyle='-')
      S.scatter(SX, SY, s=3, marker='.', color='r')
      #
      # dependence on signal level
      S = F.add_subplot(2,2,3)
      S.set_xlabel(r'Signal level $S_{1,b}$ [DN]')
      S.set_ylabel(r'IPC $\alpha(S_{1,b})-\alpha(S_{1,' + '{:d}'.format(self.htsteps[-1]) + '})$ [%]')
      for t in range(1,nhstep):
        SXa = self.hotpix_signal[:,t]
        SYa = (self.hotpix_alpha[:,t]-self.hotpix_alpha[:,-1])/.01
        if t==1:
          SX = SXa; SY = SYa
        else:
          print (t, numpy.shape(SX), numpy.shape(SXa))
          SX = numpy.concatenate((SX,SXa)); SY = numpy.concatenate((SY,SYa))
      S.set_title(r'IPC signal dependence $\Delta\alpha$')
      S.set_xlim(0., self.hotpix_ADU_range[1])
      S.set_ylim(-1.5,1.5)
      S.xaxis.set_ticks(numpy.linspace(0,self.hotpix_ADU_range[1],num=6)); S.yaxis.set_ticks(numpy.linspace(-1.5,1.5,num=11))
      S.grid(True, color='g', linestyle='-', linewidth=.5)
      S.scatter(SX, SY, s=.25, marker='+', color='r')
      S.errorbar(self.ipcmed_x[1:], self.ipcmed_y[1:]/.01, yerr=ipcmed_yerr[1:]/.01, ms=2, marker='o', color='k', ls='None')
    #
    # comparison with auto-correlations
    if self.narrowfig:
      S = F.add_subplot(2,1,2)
    else:
      S = F.add_subplot(2,2,4)
    S.set_title(r'hot pixels vs. autocorr. IPC $\alpha$')
    scale_test = numpy.concatenate((self.grid_alphaCorr.flatten(), self.grid_alphaHot.flatten()))
    smin = 0.92 * numpy.min(scale_test[scale_test>0]) / .01
    smax = 1.08 * numpy.max(scale_test) / .01
    S.set_xlim(smin,smax); S.set_ylim(smin,smax); S.set_aspect('equal')
    S.set_xlabel(r'autocorrelation $\alpha$ [%]'); S.set_ylabel(r'hot pixel $\alpha$ [%]')
    S.grid(True, color='g', linestyle='-', linewidth=.5)
    S.errorbar(self.grid_alphaCorr.flatten()/.01, self.grid_alphaHot.flatten()/.01,
      xerr=self.grid_alphaCorrErr.flatten()/.01, yerr=self.grid_alphaHotErr.flatten()/.01,
      ms=1, marker='o', color='k', capsize=1, ls='None')
    xr = numpy.linspace(0,4,num=65)
    S.plot(xr, xr, 'r-')

    F.set_tight_layout(True)
    F.savefig(self.outstem+'_hotipc.pdf')
    plt.close(F)

def run_ir_all(infile):
  """
  Runs the IR characterization.

  Parameters
  ----------
  infile : str
      The input file.

  Returns
  -------
  None

  """

  # Get configuration and copy to the output.
  with open(infile) as f:
    cf = Config(f.readlines())
  shutil.copyfile(infile, cf.outstem + '_config.txt')
  cf.fit_parameters(verbose=True)
  cf.generate_nonlinearity(write_to_file=True)
  cf.write_basic_figure()
  cf.alt_methods(verbose=True)
  cf.method_23_plot()
  with open(cf.outstem + '_summary.txt', 'w') as f:
    f.write(cf.text_output())
  s = cf.hotpix_analysis(verbose=True)
  with open(cf.outstem+'_hot.txt', 'w') as f:
    f.write(s)
  cf.hotpixel_plots()

### <== MAIN PROGRAM BELOW HERE ==> ###

if __name__ == "__main__":
  """Main run."""

  run_ir_all(sys.argv[1])
