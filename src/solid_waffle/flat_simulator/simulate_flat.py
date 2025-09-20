"""
Flat simulator.

Notes
-----
Long description::

    Start with a realization of a perfect detector plus some BFE effect to
    the charge in a given pixel
    1.  The mean charge <Q_a(i,j)> = It_a + 0.5*Sigma_a*I^2*t_a^2 (Eqn 36) 
    2.  Realization of a 4096 x 4096 pixel^2 grid with 66 time samples
    This final result data cube will then be [4k, 4k, 66] in dimensions
    and will require some kind of identifying header, currently taken
    from one of the DCL flats.  This is charge.
    3.  Convert to ADU/DN via gain ~ number(e-)/counts;  set ~ FWD/2^16
    
    Run tests with 128x128 or something small...
    current I is per pixel (units: e/s)
    time t (units: s)
    time step for read from t_a to t_a+1 (will need to check # for convergence)
    
    NOTE: to run this, one needs: 
     ** a copy of a DCL flat file with the name set below
    
"""
    
import sys
import numpy as np
from numpy.random import randn,poisson,normal
import scipy.signal as signal
import astropy.io.fits as fits
import fitsio
from fitsio import FITS,FITSHDR
import re
from ..pyirc import *
from ..ftsolve import p2kernel
from .detector_functions import *

def PairPoisson(myMean, myShape, QY_offset, QY_p2):
      """
      Build distribution of pairs of points.

      Parameters
      ----------
      myMean : float
          mean number of *photons* per pixel (number of e is 2x larger).
      myShape : (int, int)
          Shape of numpy array to generate.
      QY_offset : int
          Maximum offset of the "other" electron to consider (usually 2 is enough).
      QY_p2 : np.array
          The pairwise probability array.

      Returns
      -------
      np.array
          Pair-Poisson random array, including both Poisson photon generation
          and then the electron pairs being collected into pixels; shape = `myShape`..

      """

      Ptot = np.zeros(myShape)
      (yg,xg) = myShape
      for j in range(2*QY_offset+1):
        for i in range(2*QY_offset+1):
          dP = np.random.poisson(myMean*QY_p2[j,i], (yg+2*QY_offset, xg+2*QY_offset))
          Ptot += dP[QY_offset:yg+QY_offset, QY_offset:xg+QY_offset]
          Ptot += dP[j:yg+j, i:xg+i]
      return(Ptot)
    

class Simulation():

  """
  Run a flat field simulation.

  Parameters
  ----------
  cfg : str
      The configuration as a multiline string.
  verbose : bool, optional
      Whether to talk a lot.

  Attributes
  ----------
  pars : dict
      Parameter dictionary for this simulation.

  Methods
  -------
  __init__
      Constructor.
  run
      Runs the simulation.

  """

  def __init__(self, cfg, verbose=True):
    
    # Defaults
    formatpars = 1
    tsamp = 66
    substep = 2
    I = 10.0
    QE = 0.8
    delta_tsamp = 3.0 # arbitrary for now (s)
    gain = 1.5 # arbitrary scalar e-/DN
    outfile = 'DefaultOutput.fits'
    wavemode = 'ir' # options are 'ir' or 'vis'; used only for BFE kernel choice
    rngseed = 1000
    noisemode = 'none'
    bfemode = 'true'
    lipcmode = 'false'
    lipc_alpha = [0.01]
    nlmode = 'false'
    nlbeta = 1.5 # (ppm/e-)
    # nlcoeffs_arr are [c_2,c_3,c_4] with c_j in units of ppm electrons^(1-j) 
    nlcoeffs_arr = [-1.5725,1.9307e-5,-1.4099e-10]
    reset_frames = [0]
    resetlevel = 0.
    
    # quantum yield defaults
    QY_omega = 0.
    QY_offset = 2
    QY_p2 = np.zeros((5,5)); QY_p2[2,2] = 1
    
    # Read in information
    for line in cfg.splitlines():
      # Format
      m = re.search(r'^FORMAT:\s*(\d+)', line)
      if m: formatpars = int(m.group(1))
    
      # Number of reads
      m = re.search(r'^NREADS:\s*(\d+)', line)
      if m: tsamp = int(m.group(1))
      # substeps
      m = re.search(r'^SUBSTEPS:\s*(\d+)', line)
      if m: substep = int(m.group(1))
    
      # Time step (s)
      m = re.search(r'^DT:\s*(\S+)', line)
      if m: delta_tsamp = float(m.group(1))
    
      # Gain (e/DN)
      m = re.search(r'^GAIN:\s*(\S+)', line)
      if m: gain = float(m.group(1))
    
      # Quantum yield
      m = re.search(r'^QY:\s*(\d.*)', line)
      if m:
        QY_pars_str = m.group(1).split(" ")
        QY_omega = float(QY_pars_str[0])
        #sig = float(m.group(2))
        QY_cov = [ float(QY_pars_str[x]) for x in range(1, len(QY_pars_str)) ]
        #QY_cxx = float(m.group(2))
        #QY_cxy = float(m.group(3))
        #QY_cyy = float(m.group(4))
        QY_offset = 2
        #QY_p2 = p2kernel(np.asarray([sig**2,0,sig**2]), QY_offset, 256)
        QY_p2 = p2kernel(np.asarray(QY_cov), QY_offset, 256)
      # Illumination (photons/s/pixel)
      m = re.search(r'^ILLUMINATION:\s*(\S+)', line)
      if m: I = float(m.group(1))
      # QE (Illumination * QE = current, e/s/pixel)
      m = re.search(r'^QE:\s*(\S+)', line)
      if m: QE = float(m.group(1))
    
      # RNG seed
      m = re.search(r'^RNGSEED:\s*(\d+)', line)
      if m: rngseed = int(m.group(1))
    
      # Noise
      m = re.search(r'^NOISE:\s*(\S+)\s+(\S+)', line)
      if m:
        noisemode = m.group(1)
        if noisemode != 'none':
          noisefile = m.group(2)
    
      # Wavelength mode, affects BFE choice
      m = re.search(r'^WAVEMODE:\s*(\S+)', line)
      if m: wavemode = m.group(1)
    
      # BFE
      m = re.search(r'^BFE:\s*(\S+)', line)
      if m: bfemode = m.group(1)
    
      # linear IPC
      m = re.search(r'^L_IPC:\s*(\S+)\s+(\S.*)', line)
      if m:
        lipcmode = m.group(1)
        if lipcmode == 'true':
          lipc_alpha_str = m.group(2).split(" ")
          lipc_alpha = [ float(lipc_alpha_str[x]) for x in range(len(lipc_alpha_str)) ]
    
      # non-linearity beta
      m = re.search(r'^NL:\s*(\S+)\s+(\S.*)', line)
      if m:
        nlmode = m.group(1)
        if nlmode == 'quadratic':
          nlbeta = float(m.group(2))
        elif nlmode == 'quartic':
          nlcoeffs_str = m.group(2).split(" ")
          nlcoeffs_arr = [ float(nlcoeffs_str[x]) for x in range(len(nlcoeffs_str)) ]
      # Reset level (in e)
      m = re.search(r'^RESET_E:\s*(\S+)', line)
      if m: resetlevel = float(m.group(1))
    
      # Output file
      m = re.search(r'^OUTPUT:\s*(\S+)', line)
      if m: outfile = m.group(1)

    # Stuff to save
    save_pars = ["formatpars", "tsamp", "substep", "I", "QE", "delta_tsamp", "gain",
                 "outfile", "wavemode", "rngseed", "noisemode", "bfemode", "lipcmode",
                 "lipc_alpha", "nlmode", "nlbeta", "nlcoeffs_arr", "reset_frames", "resetlevel",
                 "QY_omega", "QY_cov", "QY_offset", "QY_p2"]

    # now save the dictionary
    self.pars = {}
    for i in save_pars:
        self.pars[i] = locals()[i]

  def run(self, verbose=False):
    """
    Runs the simulation.

    Parameters
    ----------
    verbose : bool, optional
        Whether to print lots of status updates.

    Returns
    -------
    None

    """
    
    # data cube attributes
    N = nx = ny = get_nside(self.pars["formatpars"]) # possibly redundant with nx,ny
    # Reference pixels hard-coded to 4 rows/cols around border, true for
    # all except WFC3 which has 5
    xmin,xmax,ymin,ymax = 4,N-4,4,N-4 # Extent of non-reference pixels
    nt_step = self.pars["tsamp"]*self.pars["substep"] # number of tot timesteps depending on convergence needs
    delta_t = (self.pars["delta_tsamp"]*self.pars["tsamp"])/nt_step # time between timesteps
    allQ = np.zeros((self.pars["substep"], nx, ny))
    data_cube_Q = np.zeros((self.pars["tsamp"], nx, ny))
    data_cube_S = np.zeros_like(data_cube_Q)
    offset_frame = np.zeros((1, nx, ny))
    count = 1
    reset_count = 0
    
    if verbose:
      print('side length =',N)
      print('samples:', self.pars["tsamp"], 'x', self.pars["delta_tsamp"], 's; # substep =', self.pars["substep"])
      print('Illumination:', self.pars["I"], 'ph/s/pix; QE =', self.pars["QE"])  
      print('RNG seed ->', self.pars["rngseed"])
    np.random.seed(self.pars["rngseed"])
    
    # Reset first frame if needed
    offset_frame[:,:,:] = self.pars["resetlevel"]
    if 0 in self.pars["reset_frames"]:
      allQ[0,:,:] = data_cube_Q[0,:,:] = offset_frame

    # Start with 0 charge in the first frame (t=0)
    mean = self.pars["I"]*delta_t
    
    for tdx in range(1, nt_step):
      # Create realization of charge
      # This version uses less memory, but probably still sub-optimal
      idx = tdx%self.pars["substep"]
      # Charge accumulates, dependent on quantum efficiency of the pixel
      # and the brighter-fatter effect.  First timestep is Poisson realization
      if (tdx==1):
        allQ[idx,:,:] = allQ[idx-1,:,:]
        allQ[idx,xmin:xmax,ymin:ymax] += np.random.poisson(
          self.pars["QE"]*mean*(1.-self.pars["QY_omega"])/(1.+self.pars["QY_omega"]), allQ[idx,xmin:xmax,xmin:xmax].shape)
        allQ[idx,xmin:xmax,ymin:ymax] += PairPoisson(self.pars["QE"]*mean*self.pars["QY_omega"]/(1.+self.pars["QY_omega"]),
                                                     allQ[idx,xmin:xmax,xmin:xmax].shape,
                                                     self.pars["QY_offset"], self.pars["QY_p2"])
      else:
        # If not the first step, and the brighter-fatter effect is turned
        # on, then now loop through all pixels
        if self.pars["bfemode"]=='true':
          # Calculate the area defect by taking a convolution of the bfe
          # kernel (flipped in the calc_area_defect function) and the charge Q
          # Area defects are magnified by (1+QY_omega)/(1-QY_omega) since I am only applying them
          # to the 1-electron events
          if self.pars["wavemode"]=='ir':
            a_coeff = TestKernels.get_bfe_kernel_5x5_ir()
          elif self.pars["wavemode"]=='vis':
            a_coeff = TestKernels.get_bfe_kernel_5x5_vis()
          else:
            if verbose: print("wavemode set to unknown value, defaulting BFE to IR")
            a_coeff = TestKernels.get_bfe_kernel_5x5()
          area_defect = calc_area_defect(
            (1.+self.pars["QY_omega"])/(1.-self.pars["QY_omega"])*a_coeff, allQ[idx-1,xmin:xmax,ymin:ymax])
          meanQ = area_defect*mean*self.pars["QE"]*(1.-self.pars["QY_omega"])/(1.+self.pars["QY_omega"])
          allQ[idx,xmin:xmax,ymin:ymax] = allQ[idx-1,xmin:xmax,ymin:ymax] + \
              np.random.poisson(meanQ)
          allQ[idx,xmin:xmax,ymin:ymax] += PairPoisson(self.pars["QE"]*mean*self.pars["QY_omega"]/(1.+self.pars["QY_omega"]), allQ[idx,xmin:xmax,xmin:xmax].shape,
                                                     self.pars["QY_offset"], self.pars["QY_p2"])
        else:
          # Otherwise Poisson draw the charge as before
          allQ[idx,:,:] = allQ[idx-1,:,:]
          allQ[idx,xmin:xmax,ymin:ymax] += np.random.poisson(
            self.pars["QE"]*mean*(1.-self.pars["QY_omega"])/(1.+self.pars["QY_omega"]), allQ[idx,xmin:xmax,ymin:ymax].shape)
          allQ[idx,xmin:xmax,ymin:ymax] += PairPoisson(self.pars["QE"]*mean*self.pars["QY_omega"]/(1.+self.pars["QY_omega"]),
                                                 allQ[idx,xmin:xmax,xmin:xmax].shape,
                                                 self.pars["QY_offset"], self.pars["QY_p2"])

      if (idx==0):
        data_cube_Q[count,:,:] = allQ[idx,:,:]
        allQ = np.zeros((self.pars["substep"], nx, ny))
        # if this is a reset frame set start to offset
        if count in self.pars["reset_frames"]:
          allQ[0,:,:] = offset_frame
        else:
          allQ[0,:,:] = data_cube_Q[count,:,:]
    
        count += 1
        if verbose: print("time: %d" %count)
        
    # Add in IPC before the noise if the mode is turned on
    if (self.pars["lipcmode"]=='true'):
      data_cube_Q[:,xmin:xmax,ymin:ymax] = calculate_ipc(
        data_cube_Q[:,xmin:xmax,ymin:ymax], self.pars["lipc_alpha"])
    else:
      pass
    
    # Apply non-linearity if mode turned on; assumed to act after IPC
    if (self.pars["nlmode"]=='quadratic'):
      data_cube_Q[:,xmin:xmax,ymin:ymax] -= (1.E-6*self.pars["nlbeta"]) * \
          data_cube_Q[:,xmin:xmax,ymin:ymax]**2
      if verbose: print("Applying non-linearity at leading order coefficient (quadratic term)")
    elif (self.pars["nlmode"]=='quartic'):
      data_cube_Q[:,xmin:xmax,ymin:ymax] += 1.E-6*self.pars["nlcoeffs_arr"][0] * \
          data_cube_Q[:,xmin:xmax,ymin:ymax]**2 + 1.E-6*self.pars["nlcoeffs_arr"][1] * \
          data_cube_Q[:,xmin:xmax,ymin:ymax]**3 + 1.E-6*self.pars["nlcoeffs_arr"][2] * \
          data_cube_Q[:,xmin:xmax,ymin:ymax]**4
      if verbose: print("Applying non-linearity polynomial to quartic term")
    else:
      if verbose: print("No additional non-linearity (Beta) applied")
    
    # Read in the read noise from a fits file generated with Bernie's ngxhrg
    # noisemode 'last' uses one realization because the full one takes
    # a long time to create
    if self.pars["noisemode"].lower() == 'last':
      noise = fitsio.read(self.pars["noisefile"])
      data_cube_Q[-1,:,:] += noise  # Adding to only the final time
    elif self.pars["noisemode"].lower() == 'full':
      noise = fitsio.read(noisefile)
      data_cube_Q += noise  # Adding the noise at all reads
    elif self.pars["noisemode"].lower() == 'none':
      pass
    else:
      # a small amount of Gaussian noise, 12 e
      data_cube_Q += 10*normal(size=(self.pars["tsamp"],N,N))
    
    # Convert charge to signal, clipping values<0 and >2**16
    data_cube_S = np.array(
      np.clip(data_cube_Q/self.pars["gain"], 0, 65535), dtype=np.uint16)
    
    # Write simple header, todo: add more thorough comments
    hdr = FITSHDR()
    hdr['GAIN'] = self.pars["gain"]
    hdr['ILLUMIN'] = I
    hdr['QE'] = self.pars["QE"]
    hdr['RNGSEED'] = self.pars["rngseed"]
    if (self.pars["lipcmode"]=='true'):
      hdr['LINIPC'] = self.pars["lipc_alpha"][0]
    if (self.pars["nlmode"]=='quadratic'):
      hdr['BETA'] = self.pars["nlbeta"]
    if (self.pars["nlmode"]=='quartic'):
      hdr['NLCOEFFS_c2'] = self.pars["nlcoeffs_arr"][0]
      hdr['NLCOEFFS_c3'] = self.pars["nlcoeffs_arr"][1]
      hdr['NLCOEFFS_c4'] = self.pars["nlcoeffs_arr"][2]
    if (self.pars["bfemode"]=='true'):
      hdr['BFE_A00'] = a_coeff[2][2]  # Hard-coded to expect 5x5 a coeffs
    
    # Open up an example DCL flat file and save the data cube
    #dclfile = 'Set_001_Test_0002.fits'
    fitsio.write(self.pars["outfile"], data_cube_S, header=hdr, clobber=True)
    
    # Try compression of data cube into file
    # End of script


def run_config(filename):
    """
    Runs a configuration from a file name.

    Parameters
    ----------
    filename : str
        The name of the file.

    Returns
    -------
    None

    """

    with open(filename) as myf:
        sim = Simulation(myf.read(), verbose=True)
    print(sim.pars)
    sim.run(verbose=True)

if __name__ == "__main__":
    run_config(sys.argv[1])
