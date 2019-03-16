"""
Wrap LSD(http://www.ipol.im/pub/art/2012/gjmr-lsd/) line segment detector.
"""
import numpy as np
cimport numpy as np

cdef extern from 'lsd_1.6/lsd.h':
    double* LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double scale, double sigma_scale, double quant,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y )

np.import_array()

def detect_line_segments(np.ndarray[np.float64_t,ndim=2] image,
    scale=0.8, sigma_scale=0.6, quant=2.0, ang_th=22.5, log_eps=0.0, density_th=0.7, n_bins=1024):
    """
    Detect line segments in grayscale image.

    Args:
        image: grayscale image in double
    Kwargs:
        scale[0.8]: subsampling scale factor (1.0 runs faster)
        sigma_scale[0.6]: control Gaussian filter (sigma = sigma_scale/scale)
        quant[2.0]: bound to the quantization error on the gradient norm
        ang_th[22.5]: gradient angle tolerance in degrees
        log_eps[0.0]: detection threshold: -log10(NFA) > log_eps
        density_th[0.7]: minimal density of region points in rectangle
        n_bins[1024]: number of bins in pseudo-ordering of gradient modulus
    Returns:
        (N,7) array [(x0,y0,x1,y1,width,p,-10log(NFA))]
    """
    cdef int n_segs
    cdef double* segs = LineSegmentDetection(&n_segs,
        <double*>image.data, image.shape[1], image.shape[0],
        scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins, NULL, NULL, NULL)
    
    cdef np.npy_intp shape[2]
    shape[0] = n_segs
    shape[1] = 7
    result = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, segs)
    np.PyArray_UpdateFlags(result, result.flags.num | np.NPY_OWNDATA)
    return result
