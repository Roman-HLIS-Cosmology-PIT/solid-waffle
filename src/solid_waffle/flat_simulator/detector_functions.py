"""
Functions to create various detector effect.
The structure will change, but for now this will be the location for 
functions related to IPC, BFE, etc.

Functions
---------
simple_ipc_kernel
    Makes a simple IPC kernel with single nearest-neighbor coupling.
ipc_kernel_HV
    A 3x3 kernel with horizontal and vertical alpha, which can be different.
calculate_ipc
    Convolves the input charge data cube with an IPC kernel and returns an output data cube.
calc_area_defect
    Computes the area defect of an array of pixels due to a BFE kernel.
ipc_invkernel_HV
    Return an inverse 3x3 kernel with horizontal and vertical alpha, which can be different.
auto_convolve_kernel
    Return the auto-convolution of the kernel.
K2a
    Return the convolution of the BFE a coefficients with the IPC kernel.

Classes
-------
TestKernels
  Makes BFE kernels for testing inputs.

"""

import sys
import numpy as np
from numpy.random import randn,poisson
from ..pyirc import *
import scipy.signal as signal

def simple_ipc_kernel(alpha=0.01):
  """
  Makes a simple IPC kernel with single nearest-neighbor coupling.

  Simple function to return a 3 x 3 kernel with an alpha where `alpha`
  is the kernel value for the 4 adjacent pixels, and the central value is
  1-4*`alpha`.  This is a symmetric kernel.

  Parameters
  ----------
  alpha : float, optional
      Nearest-neighbor coupling.

  Returns
  -------
  np.array of float
      The shape (3,3) IPC kernel, centered at zero.

  """

  kernel = np.zeros((3, 3))
  kernel[1,0] = kernel[0,1] = kernel[1,2] = kernel[2,1] = alpha
  kernel[1,1] = 1-4*alpha
  return kernel

def ipc_kernel_HV(alpha_H=0.01,alpha_V=0.01):
  """
  A 3x3 kernel with horizontal and vertical alpha, which can be different.

  Parameters
  ----------
  alphaH, alphaV : float, optional
      The horizontal and vertical IPC couplings.

  Returns
  -------
  np.array of float
      The shape (3,3) IPC kernel, centered at zero.

  """

  kernel = np.zeros((3, 3))
  kernel[0,1] = kernel[2,1] = alpha_H
  kernel[1,0] = kernel[1,2] = alpha_V
  kernel[1,1] = 1-2*alpha_H-2*alpha_V
  return kernel

def calculate_ipc(data_cube_Q, ipc_list, npad=2):
  """
  Convolves the input charge data cube with an IPC kernel and returns an output data cube.

  Calls "simple" or "HV" IPC kernel depending on `ipc_list`,
  which is either one value (all directions) or two (horiz, vertical)
  Currently cannot specify horiz or vertically asymmetric alpha.

  Parameters
  ----------
  data_cube_Q : np.array
      A 3D charge image (t,y,x). Updated in place.
  ipc_list : np.array or list of float
      If length 1, an isotropic alpha; if length 2, interpreted as
      [alpha_H, alphaV].
  npad : int, optional
      Amount by which to pad the image before IPC convolution
      (adjusts how boundaries are handled; default of 2 is probably
      fine for almost all purposes).

  Return
  ------
  np.array
      Returns a view of the new charge image.

  """

  if len(ipc_list)==1:
    ipc_kern = simple_ipc_kernel(ipc_list[0])
  elif len(ipc_list)==2:
    ipc_kern = ipc_kernel_HV(ipc_list[0], ipc_list[1])
  else:
    raise Exception('Incorrect format of IPC alpha entered')

  # The time samples are given by the first dim of the cube
  for tdx in range(data_cube_Q.shape[0]):
    Q_pad = np.pad(
      data_cube_Q[tdx,:,:], pad_width=(npad,npad), 
      mode='symmetric')
    Q_pad_ipc = signal.convolve(Q_pad, ipc_kern)
    # Dimensions/side for Q_pad_ipc are now 
    # data_cube_Q.shape[0]+ipc_kern.shape[0]+npad-1
    extra_dim = (2*npad+ipc_kern.shape[0]-1)//2
    data_cube_Q[tdx,:,:] = Q_pad_ipc[extra_dim:-extra_dim,
                                     extra_dim:-extra_dim]
  return data_cube_Q

class TestKernels():
  """
  Makes BFE kernels for testing inputs.

  Methods
  -------
  get_bfe_kernel_3x3
      Returns a simple, currently arbitrary bfe 3 x 3 kernel.
  get_bfe_kernel_5x5_ir
      Returns a bfe 5x5 kernel like for SCA 20829. 
  get_bfe_kernel_5x5_vis
      Returns a bfe 5x5 kernel like for SCA 20829 vis.
  get_bfe_kernel_5x5_18237ir
      Returns an arbitrary bfe 5x5 kernel; these numbers are similar to SCA 18237.
  get_bfe_kernel_5x5_symm
      Returns a symmetric bfe 5x5 kernel.
  get_bfe_kernel_zeros
      Returns a 5x5 matrix of 0s for testing.

  """

  @classmethod
  def get_bfe_kernel_3x3(cls):
    """
    Returns a simple, currently arbitrary bfe 3 x 3 kernel.
    """

    bfe_kernel_3x3 = 1.E-6*np.array(
      [[0.065, 0.23, 0.065],[0.24, -1.2, 0.24], [0.065, 0.23, 0.065]]) 
    # Currently symmetrical but can put in something more complex
    return bfe_kernel_3x3

  @classmethod
  def get_bfe_kernel_5x5_ir(cls):
    """
    Returns a bfe 5x5 kernel like for SCA 20829.
    """

    bfe_kernel_5x5 = 1.E-6*np.array(
      [[ 0.0016,  0.0128,  0.02  ,  0.0166, -0.002 ],
        [-0.0039,  0.1   ,  0.4343,  0.1068,  0.0044],
        [ 0.0077,  0.3598, -2.0356,  0.3797,  0.0157],
        [-0.0047,  0.0835,  0.3807,  0.1068, -0.0083],
        [-0.0048,  0.0051,  0.0297, -0.0023, -0.0036]])

    # If the sum of this isn't close to 0, this will cause issues in the sim
    if np.sum(bfe_kernel_5x5)>=1.E-9:
      raise ValueError("sum check failed")

    return np.fliplr(bfe_kernel_5x5)

  @classmethod
  def get_bfe_kernel_5x5_vis(cls):
    """
    Returns a bfe 5x5 kernel like for SCA 20829 vis.
    """

    bfe_kernel_5x5 = 1.E-6*np.array(
      [[ 0.0413,  0.0392,  0.0533,  0.0494,  0.035 ],
       [ 0.0518,  0.14  ,  0.4207,  0.1531,  0.0637],
       [ 0.0579,  0.3834, -3.0462,  0.403 ,  0.0589],
       [ 0.0488,  0.1466,  0.4633,  0.142 ,  0.0499],
       [ 0.0446,  0.0403,  0.0693,  0.0448,  0.0459]])

    # If the sum of this isn't close to 0, this will cause issues in the sim
    if np.sum(bfe_kernel_5x5)>=1.E-9:
      raise ValueError("sum check failed")

    return np.fliplr(bfe_kernel_5x5)

  @classmethod
  def get_bfe_kernel_5x5_18237ir(cls):
    """
    Returns an arbitrary bfe 5x5 kernel; these numbers are similar to SCA 18237.
    """

    bfe_kernel_5x5 = 1.E-6*np.array(
      [[-0.01, 0.0020, -0.0210, -0.019, 0.028],
       [0.0040, 0.0490, 0.2480, 0.01, -0.0240],
       [-0.0170, 0.2990, -1.372, 0.2840, 0.0150],
       [0.0130, 0.0560, 0.2890, 0.0390, 0.02],
       [0.035, 0.0070, 0.0380, 0.0010, 0.026]])
    return np.fliplr(bfe_kernel_5x5)

  @classmethod
  def get_bfe_kernel_5x5_symm(cls):
    """
    Returns a symmetric bfe 5x5 kernel.
    """

    bfe_kernel_5x5 = 1.E-6*np.array(
      [[0.0802, 0.0020, 0.002, 0.002, 0.002],
       [0.002, 0.01, 0.2840, 0.01, 0.002],
       [0.002, 0.2840, -1.372, 0.2840, 0.002],
       [0.002, 0.01, 0.2840, 0.01, 0.002],
       [0.002, 0.002, 0.002, 0.0020, 0.0878]])
    return np.fliplr(bfe_kernel_5x5)

  @classmethod
  def get_bfe_kernel_zeros(cls):
    """
    Returns a 5x5 matrix of 0s for testing.
    """

    nobfe_kernel_5x5 = np.zeros((5,5))
    return nobfe_kernel_5x5

def calc_area_defect(ap, Q, npad=2):
  """ 
  Computes the area defect of an array of pixels due to a BFE kernel.

  Paramters
  ---------
  ap : np.array
      The a_{deltai,deltaj} coefficient matrix, units of 1/e, centered at (0,0).
      (Size should be an odd number and should be square.)
  Q : np.array
      A 2D array of the charge.
  npad : int, optional
      Amount of padding to apply (radius of BFE kernel is a good choice).

  Return
  ------ 
  np.array
      The area defect, i.e. area of each pixel relative to the unperturbed area. The
      area defect is unitless. The shape is the same as for `Q`.

  """

  # checks
  if np.shape(ap)[0] != np.shape(ap)[1] or np.shape(ap)[0]%2==0:
    raise ValueError("ap must be square with an odd side length.")

  # Q_pad is a padded array with mirror reflection along the boundaries
  Q_pad = np.pad(Q, pad_width=(npad,npad), mode='symmetric')

  # Larger-dimensional array must be first arg to convolve
  aQ = signal.convolve(Q_pad, ap) # ap[::-1,::-1])
  W = 1 + aQ
  # Final dimensions of W will be 2*npad+Q.shape[0]+ap.shape[0]-1
  # on each side
  extra_dim = (2*npad+ap.shape[0]-1)//2
  return W[extra_dim:-extra_dim,extra_dim:-extra_dim]

def ipc_invkernel_HV(alpha_H=0.01,alpha_V=0.01):
  """
  Return an inverse 3x3 kernel with horizontal and vertical alpha, which can be different.

  Parameters
  ----------
  alpha_H, alpha_V : float, optional
      The IPC coupling coefficients in the horizontal and vertical directions.

  Returns
  -------
  np.array
      The shape (3,3) approximation to the inverse IPC kernel.

  """

  kernel = np.zeros((3, 3))
  kernel[0,1] = kernel[2,1] = -2.*alpha_H
  kernel[1,0] = kernel[1,2] = -2.*alpha_V
  kernel[1,1] = 1+4*alpha_H+4*alpha_V
  return kernel

def auto_convolve_kernel(kern1):
  """
  Return the auto-convolution of the kernel.

  Parameters
  ----------
  kern1: np.array
      The input kernel as a 2D array.

  Returns
  -------
  np.array
      The autoconvolution of the input kernel as a 2D array.

  """

  kern_autoconv = signal.convolve(kern1, kern1)
  return kern_autoconv

def K2a(ipc_kernel2, bfe_a, round=None):
  """
  Return the convolution of the BFE a coefficients with the IPC kernel.

  Parameters
  ----------
  ipc_kernel2 : np.array
      Autoconvolution of the IPC kernel (centered at 0).
  bfe_a : np.array
      The BFE kernel.
  round : int, optional
      If given, rounds for ease of display; default is no rounding.

  Returns
  -------
  np.array
      The convolition K*K*a, trimmed to the same size as `bfe_a`.

  """

  # First need to pad the BFE coefficients (compare to case without)
  npad = 2
  bfe_a_pad = np.pad(bfe_a, pad_width=(npad,npad), mode='symmetric')
  #ipc2_bfe = signal.convolve(ipc_kernel2, bfe_a_pad)
  ipc2_bfe = signal.convolve(bfe_a_pad, ipc_kernel2)
  extra_dim = (2*npad+ipc_kernel2.shape[0]-1)//2
  ipc2_bfe_out = ipc2_bfe[extra_dim:-extra_dim, extra_dim:-extra_dim]
  if round is not None:
      ipc2_bfe_out = np.around(ipc2_bfe_out, round)
  return ipc2_bfe_out

def a_symmetric_avg(coeffs, round=None):
  """
  Return the symmetric averages over a given 5x5 matrix of coefficients.

  Parameters
  ----------
  coeffs : np.array
      A 5x5 matrix of coefficients.
  round : int, optional
      If given, rounds for ease of display; default is no rounding.

  Returns
  -------
  np.array
      A length 6 array of the symmetrized kernel. Ordering is
      <0,0>, <1,0>, <1,1>, <2,0>, <2,1>, <2,2>.

  """

  zerozero1 = coeffs[2,2]
  onezero1 = np.mean((coeffs[2,3], coeffs[2,1], coeffs[3,2],
                      coeffs[1,2]))
  oneone1 = np.mean((coeffs[1,1], coeffs[3,1], coeffs[3,3],
                     coeffs[1,3]))

  twozero1 = np.mean((coeffs[2,0], coeffs[0,2], coeffs[4,2],
                      coeffs[2,4]))
  twoone1 = np.mean((coeffs[4,3], coeffs[3,4], coeffs[1,4],
                     coeffs[0,3], coeffs[0,1], coeffs[1,0],
                     coeffs[3,0], coeffs[4,1]))
  twotwo1 = np.mean((coeffs[0,0], coeffs[0,4], coeffs[4,0],
                     coeffs[4,4]))

  means = np.array((zerozero1, onezero1, oneone1, twozero1, twoone1,
                    twotwo1))
  if round is not None:
    means = np.around(means, round)
  return means

if __name__=="__main__":
  """Tests from the Choi et al. paper."""

  # Print out the K^2 a coefficients (symmetrically averaged) for the
  # simulated input
  # First get the IPC kernel
  kern = ipc_kernel_HV(0.0169,0.0169) # alphah,alphav for the sims
  # Convolve to get K^2
  kern2 = auto_convolve_kernel(kern)
  # Some of this flipping might not be quite right, but since these are
  # symmetric averages it's ok for now
  input_bfe_a = 1.E6*np.fliplr(TestKernels.get_bfe_kernel_5x5_ir())
  K2a_out = K2a(kern2, input_bfe_a, round=4)
  print("<0,0>, <1,0>, <1,1>, <2,0>, <2,1>, <2,2>:")
  print(a_symmetric_avg(K2a_out, round=4))
  print(np.around(np.fliplr(input_bfe_a), 4))

