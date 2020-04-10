
"""
Trial definition of a Trace as a 2D polynomial over the slit.
If it were a 1-D polynomial, it would give the location of a
single trace as a function y(x), or more generally s(w) where
s is spatial coordinate and w is wavelength coordinate.
(Ideally specreduce routines would respect some overall setting
that controlled whether w = x or w = y.)

Generalizing to a 2-d polynomial allows the trace to be computed at
different locations along the slit. A simple version is that the
model is a polynomial in x plus a linear term in y, 
so the trace at a different location along the slit
retains the same shape but is offset by a constant number of pixels in y.
This is something like Chebyshev1D(x) + Linear1D(y)
or Chebyshev2D(x, ydummy) + Planar2D(xdummy, y).
That's different from Chebyshev2D(x,y) = T(x) * T(y).

The parameters of the trace are a model with its model params,
and the w_ref and s_ref.  These specify a reference trace, which
will usually be the middle of the spectrum in w, and the location
measured by compute_trace in s.  For ex, we measure the trace of a
bright star and find that it goes through pixel (1024, 25).  The
idea is that we might also want to calculate the trace that goes
through e.g. pixel (1024, 30), and do an extraction there.

bjw 2020-04-09

"""

import numpy as np
from astropy.modeling import fitting
from astropy.modeling.functional_models import Linear1D, Planar2D
from astropy.modeling.polynomial import Chebyshev2D, Chebyshev1D

class Trace:
    def __init__(self, model, w_degree=11, s_degree=1, w_ref=0.0, s_ref=0.0, *args, **kwargs):
        # super().__init__(w_degree, s_degree, *args, **kwargs)
        self.w_degree = w_degree
        self.s_degree = s_degree
        self.w_ref = w_ref
        self.s_ref = s_ref
        # Kludgy since we should respect the model argument
        self.model = Chebyshev2D(w_degree, s_degree)
        
    def compute_trace1d(self, img):
        # img is a 2d numpy array or CCDData
        # Here we would do the trace measurement on the image
        # with the trace() function from jradavenport's apextract.py
        # which returns an array of y for each x pixel
        # This is what actually does the finding work.
        wtrace, strace = measure_trace(img, dofit=False)
        # same treatment of axes as in jradavenport's trace()
        # this could be turned into an argument to allow
        # transposing the axes. Other routines would have to respect that
        # argument.
        Waxis = 1
        Saxis = 0
        nw = img.shape[Waxis]
        # wtrace = np.arange(0, nw)
        # Fit a Chebyshev1Dto the wtrace, strace?
        # Make this a special case of a Chebyshev2D with order 0 in
        # the second dimension.  Using a Chebyshev2D here because
        # we might actually use the 2D someday, plus I want to
        # add the Pla
        # Probably don't need to mess with the domain/window values
        poly_init = Chebyshev2D(self.w_degree, 0)
        y_dummy = np.zeros_like(wtrace)
        # what fitter is best here?  May need LevMarLSQ since it seems
        # to handle the dummy dimension better than LinearLSQ
        # fitter = fitting.LinearLSQFitter()
        fitter = fitting.LevMarLSQFitter()
        fitted_poly = fitter(poly_init, wtrace, y_dummy, strace)
        # set w_ref and s_ref to be some point on the measured trace
        self.w_ref = int(nw / 2)
        self.s_ref = fitted_poly(self.w_ref, 0.0)
        # return the polynomial model
        self.model = fitted_poly
        return fitted_poly

    def compute_trace2d(self, img):
        # For right now, we just punt and transfer the 1-d params to
        # a composite model that has the Chebyshev plus a linear term
        # in s-direction
        # Ideally, there would be a way of measuring more than one
        # trace (probably using more than one image) and interpolating
        # between them to make a 2-d map
        trace_1d = self.compute_trace1d(img)
        # Somehow copy the 1-D parameters into the x part of the 2D model
        # look at https://docs.astropy.org/en/stable/modeling/parameters.html
        # coeffs1 = dict((name, idx) for idx, name in enumerate(trace_1d.param_names))
        # coeffs2 = coeffs1
        # coeffs2['c0_1'] = 0.0
        # coeffs2['c1_1'] = 1.0
        # trace_2d = Chebyshev2D(self.wdegree, 1, **coeffs2)
        ## copy over coefficients
        # self. something  = trace_2d. something ...
        # But it turns out for a simple model I want to make the model 
        # additive T(x) + L(y), rather than multiplicative T(x) * T(y)
        # use a 2-D linear model = -s_ref + 1.0 * s-coordinate
        # ie slope of 1 pixel per pixel. Later this could be
        # a Polynomial2D.
        if self.s_degree > 1:
            raise ValueError('Trace spatial degree > 1 isnt implemented yet')
        s_term = Planar2D(slope_x=0.0, slope_y=1.0, intercept=-self.s_ref)
        trace_2d = trace1d + s_term
        self.model = trace_2d
        return trace_2d

    def return_trace(self, warray, s=-1):
        # Return an array of spatial positions corresponding to the
        # warray = input pixel array (on wavelength axis), for the
        # trace that runs through pixel (w_ref, s)
        # Should we raise an error if the model hasn't been fitted yet?
        # Kludge to set the spatial pixel of the desired trace if not set
        if s < 0.0:
            s = self.s_ref
        svalue = np.zeros_like(warray) + s
        trace_array = self.model(warray, svalue)
        return trace_array

    # Do we need other functions, for example something that takes
    # a pixel (w, s) and figures out what trace it is on?

    

# trace() is a function that takes an image and measures the trace
# of an object, returning an array of y-position of the trace
# at each x index, here borrowing the trace() function from jradavenport's
# specreduce/apextract.py

### begin wholesale cut-paste from jradavenport's apextract.py
###  with a small modification to return x and y arrays

# import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
# from astropy.table import Table


# __all__ = ['trace', 'extract']


def _gaus(x, a, b, x0, sigma):
    """
    Define a simple Gaussian curve
    Could maybe be swapped out for astropy.modeling.models.Gaussian1D
    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian
    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def measure_trace(img, ilum=(1,), nbins=20, guess=-1, window=0, display=False, dofit=False):
    """
    Trace the spectrum aperture in an image
    Assumes wavelength axis is along the X, spatial axis along the Y.
    Chops image up in bins along the wavelength direction, fits a Gaussian
    within each bin to determine the spatial center of the trace. Finally,
    draws a cubic spline through the bins to up-sample trace along every X pixel.
    Parameters
    ----------
    img : 2d numpy array, or CCDData object
        This is the image to run trace over
    ilum : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine.
    nbins : int, optional
        number of bins in wavelength (X) direction to chop image into. Use
        fewer bins if trace is having difficulty, such as with faint
        targets (default is 20, minimum is 4)
    guess : int, optional
        A guess at where the desired trace is in the spatial direction (Y). If set,
        overrides the normal max peak finder. Good for tracing a fainter source if
        multiple traces are present.
    window : int, optional
        If set, only fit the trace within a given region around the guess position.
        Useful for tracing faint sources if multiple traces are present, but
        potentially bad if the trace is substantially bent or warped.
    display : bool, optional
        If set to true display the trace over-plotted on the image
    dofit : bool, optional
        If set to true spline-fit the points in bins, otherwise return the bin values only
    Returns
    -------
    my : array
        The spatial (Y) positions of the trace, interpolated over the
        entire wavelength (X) axis
    Improvements Needed
    -------------------
    1) switch to astropy models for Gaussian (?)
    2) return info about trace width (?)
    3) add re-fit trace functionality (or break off into another method)
    4) add other interpolation modes besides spline (?)
    """

    # define the wavelength & spatial axis, if we want to enable swapping programatically later
    # defined to agree with e.g.: img.shape => (1024, 2048) = (spatial, wavelength)
    Waxis = 1 # wavelength axis
    Saxis = 0 # spatial axis

    # Require at least 4 big bins along the trace to define shape. Sometimes can get away with very few
    if (nbins < 4):
        raise ValueError('nbins must be > 4')

    # if illuminated portion not defined, just use all of spatial axis extent
    if (len(ilum)<2):
        ilum = np.arange(img.shape[Saxis])

    # Pick the highest peak, bad if mult. obj. on slit...
    ztot = np.nansum(img, axis=Waxis)[ilum] / img.shape[Waxis] # average data across all wavelengths
    peak_y = ilum[np.nanargmax(ztot)]
    # if the user set a guess for where the peak was, adopt that
    if guess > 0:
        peak_y = guess

    # guess the peak width as the FWHM, roughly converted to gaussian sigma
    width_guess = np.size(ilum[ztot > (np.nanmax(ztot)/2.)]) / 2.355
    # enforce some (maybe sensible?) rules about trace peak width
    if width_guess < 2.:
        width_guess = 2.
    if width_guess > 25:
        width_guess = 25

    # [avg peak height, baseline, Y location of peak, width guess]
    peak_guess = [np.nanmax(ztot), np.nanmedian(ztot), peak_y, width_guess]

    # fit a Gaussian to peak
    popt_tot, pcov = curve_fit(_gaus, ilum, ztot, p0=peak_guess)

    if (window > 0):
        if (guess > 0):
            ilum2 = ilum[np.arange(guess-window, guess+window, dtype=np.int)]
        else:
            ilum2 = ilum[np.arange(popt_tot[2] - window, popt_tot[2] + window, dtype=np.int)]
    else:
        ilum2 = ilum

    xbins = np.linspace(0, img.shape[Waxis], nbins+1, dtype='int')
    ybins = np.zeros(len(xbins)-1, dtype='float') * np.NaN

    for i in range(0,len(xbins)-1):
        #-- fit gaussian w/i each window
        zi = np.nansum(img[ilum2, xbins[i]:xbins[i+1]], axis=Waxis)
        peak_y = ilum2[np.nanargmax(zi)]
        width_guess = np.size(ilum2[zi > (np.nanmax(zi) / 2.)]) / 2.355
        if width_guess < 2.:
            width_guess = 2.
        if width_guess > 25:
            width_guess = 25
        pguess = [np.nanmax(zi), np.nanmedian(zi), peak_y, width_guess]
        try:
            popt, _ = curve_fit(_gaus, ilum2, zi, p0=pguess)

            # if gaussian fits off chip, then fall back to previous answer
            if (popt[2] <= min(ilum2)) or (popt[2] >= max(ilum2)):
                ybins[i] = popt_tot[2]
            else:
                ybins[i] = popt[2]
                popt_tot = popt  # if a good measurment was made, switch to these parameters for next fall-back

        except RuntimeError:
            popt = pguess

    # recenter the bin positions
    xbins = (xbins[:-1] + xbins[1:]) / 2.

    yok = np.where(np.isfinite(ybins))[0]
    if len(yok) > 0:
        xbins = xbins[yok]
        ybins = ybins[yok]

        # We could omit the fit if the trace class will do its own fit - BJW
        if dofit is True:
            # run a cubic spline thru the bins
            ap_spl = UnivariateSpline(xbins, ybins, k=3, s=0)

            # interpolate the spline to 1 position per column
            mx = np.arange(0, img.shape[Waxis])
            my = ap_spl(mx)
        else:
            mx = xbins
            my = ybins
    else:
        mx = np.arange(0, img.shape[Waxis])
        my = np.zeros_like(mx) * np.NaN
        import warnings
        warnings.warn("TRACE ERROR: No Valid points found in trace")

    if display is True:
        plt.figure()
        plt.imshow(img, origin='lower', aspect='auto', cmap=plt.cm.Greys_r)
        plt.clim(np.percentile(img, (5, 98)))
        plt.scatter(xbins, ybins, alpha=0.5)
        plt.plot(mx, my)
        plt.show()

    # return both x and y arrays, not just y - BJW
    return mx, my

### end wholesale cut-paste from jradavenport

        
