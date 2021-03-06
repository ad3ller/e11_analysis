# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:06:21 2018

@author: Adam

    fitting tools

"""
from  inspect import signature
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit, leastsq
from scipy.special import erfc

""" 
    1D data
    -------
"""

class _1D(ABC):
    """ Fit a function to 1D data
    """
    def __init__(self, xdata, ydata, sigma=None):
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        self.sigma = np.array(sigma) if sigma is not None else sigma
        self.popt = None
        self.pcov = None
        self.perr = None
        # step size
        if xdata is not None:
            self.dx = np.mean(np.diff(self.xdata))

    @classmethod
    @abstractmethod
    def func(cls, *args):
        pass

    @classmethod
    def apply_func(cls, xdata, *params):
        """ Apply cls.func() to xdata.

        Notes
        -----
        apply_func(*args) trys to return func(*args) but if that
        raises an Exception then an array of infs like xdata is 
        returned instead. This allows assert statements defined 
        in func to be used to constrain fit parameters.
        """
        try:
            result = cls.func(xdata, *params)
        except:
            result = np.full_like(xdata, np.inf)
        finally:
            return result

    @classmethod
    def get_signature(cls, **kwargs):
        return signature(cls.func, **kwargs)

    @classmethod
    def get_variables(cls, **kwargs):
        return tuple(cls.get_signature(**kwargs).parameters.keys())

    @abstractmethod
    def approx(self):
        return None

    def fit(self, p0=None, uncertainty=True, **kwargs):
        """ least-squares fit of func() to data """
        sigma = kwargs.pop("sigma", self.sigma)
        if p0 is None:
            p0 = self.approx()
        self.popt, self.pcov = curve_fit(self.apply_func,
                                         self.xdata,
                                         self.ydata,
                                         p0=p0,
                                         sigma=sigma,
                                         **kwargs)
        if uncertainty:
            self.perr = np.sqrt(np.diag(self.pcov))
            return self.popt, self.perr
        else:
            return self.popt

    @property
    def best_fit(self):
        """ best fit vals """
        return self.func(self.xdata, *self.popt)

    @property
    def residuals(self):
        """ best fit vals """
        return self.ydata - self.best_fit

    def asdict(self, names=None, uncertainty=True):
        """ get best fit parameters as a dictionary"""
        if names is None:
            names = list(self.get_variables()[1:])
        if uncertainty and self.perr is not None:
            return dict(zip(names, zip(self.popt, self.perr)))
        else:
            return dict(zip(names, self.popt))


class Linear(_1D):
    """ Fit a line to 1D data
    """
    @classmethod
    def func(cls, x, m, c):
        """ 1D line, y = m x + c"""
        return m * x + c

    def approx(self):
        """ estimate func pars"""
        m = (self.ydata[-1] - self.ydata[0]) / (self.xdata[-1] - self.xdata[0])
        c =  self.ydata[0] -  m * self.xdata[0]
        return m, c


class Quadratic(_1D):
    """ Fit a quadratic to 1D data
    """
    @classmethod
    def func(cls, x, x0, amp, offset):
        """ 1D line, y = amp * (x - x0)^2 + offset"""
        return amp * (x - x0)**2.0 + offset

    def approx(self):
        """ estimate func pars """
        num = len(self.xdata)
        if num > 20:
            # resample
            step = len(self.xdata) // 20
            xs, ys = self.xdata[::step], self.ydata[::step]
        else:
            xs, ys = self.xdata, self.ydata
        g1 = np.diff(ys) / np.diff(xs)
        g2 =  np.nanmean(np.diff(ys, n=2) / (np.diff(xs)[:-1])**2.0)
        amp = g2 / 2.0
        x0 = np.nanmean(xs[:-1] - g1 / g2) + (xs[1] - xs[0]) / 2
        offset = np.nanmean(ys - amp * (xs - x0)**2.0)
        return x0, amp, offset


class Gaussian(_1D):
    """ Fit a Gaussian to 1D data
    """
    @classmethod
    def func(cls, x, x0, amp, sigma, offset):
        """ 1D Gaussian function with offset"""
        assert sigma > 0.0
        return amp * np.exp(-0.5 * ((x - x0) / sigma)**2.0) + offset

    def approx(self):
        """ estimate func pars (assumes positive amplitude) """
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.min(self.ydata)
        amp = np.max(self.ydata) - offset
        sigma = ((self.ydata - offset) >= 0.5 * amp).sum() * abs(self.dx)
        return x0, amp, sigma, offset


class DoubleGaussian(_1D):
    """ Fit a double Gaussian to 1D data
    """
    @classmethod
    def func(cls, x, x0, a0, w0, x1, a1, w1, offset):
        """ 1D Gaussian function with offset"""
        assert w0 > 0.0
        assert w1 > 0.0
        return (a0 * np.exp(-0.5 * ((x - x0) / w0)**2.0) 
                + a1 * np.exp(-0.5 * ((x - x1) / w1)**2.0)
                + offset)

    def approx(self):
        """ estimate func pars (assumes positive amplitude) """
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.min(self.ydata)
        a0 = np.max(self.ydata) - offset
        w0 = 0.25 * ((self.ydata - offset) >= 0.5 * a0).sum() * abs(self.dx)
        idx1 = np.argmax(np.logical_or(self.xdata < (x0 - w0),  self.xdata > (x0 + w0)) * self.ydata)
        x1 = self.xdata[idx1]
        a1 = self.ydata[idx1] - offset
        w1 = w0
        return x0, a0, w0, x1, a1, w1, offset


class Lorentzian(_1D):
    """ Fit a Lorentzian to 1D data
    """
    @classmethod
    def func(cls, x, x0, amp, gamma, offset):
        """ 1D Lorentzian function """
        assert gamma > 0.0
        return amp * gamma**2 / ((x - x0)**2 + gamma**2) + offset

    def approx(self):
        """ estimate func pars (assumes positive amplitude)"""
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.min(self.ydata)
        amp = np.max(self.ydata) - offset
        gamma = ((self.ydata - offset) >= 0.5 * amp).sum() * abs(self.dx)
        return x0, amp, gamma, offset


class DoubleLorentzian(_1D):
    """ Fit a double Lorentzian to 1D data
    """
    @classmethod
    def func(cls, x, x0, a0, g0, x1, a1, g1, offset):
        """ 1D Lorentzian function """
        assert g0 > 0.0
        assert g1 > 0.0
        return (a0 * g0**2 / ((x - x0)**2 + g0**2) 
                + a1 * g1**2 / ((x - x1)**2 + g1**2) 
                + offset)

    def approx(self):
        """ estimate func pars (assumes positive amplitude)"""
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.min(self.ydata)
        a0 = np.max(self.ydata) - offset
        g0 = 0.25 * ((self.ydata - offset) >= 0.5 * a0).sum() * abs(self.dx)
        idx1 = np.argmax(np.logical_or(self.xdata < (x0 - g0),  self.xdata > (x0 + g0)) * self.ydata)
        x1 = self.xdata[idx1]
        a1 = self.ydata[idx1] - offset
        g1 = g0
        return x0, a0, g0, x1, a1, g1, offset


class EMG(_1D):
    """ Fit an exponentially modified Gaussian to 1D data
    """
    @classmethod
    def func(cls, x, x0, amp, sigma, tau, offset):
        """ 1D Gaussian convolved with an exponential decay (tau)"""
        assert sigma > 0.0
        assert tau > 0.0
        return amp \
               * np.exp(0.5 * (2.0 * x0 + sigma**2.0 / tau - 2 * x) / tau) \
               * erfc((x0 + sigma**2.0 / tau - x) / (2.0**0.5 * sigma)) \
               + offset

    def approx(self):
        """ estimate func pars (assumes positive amplitude)"""
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.mean(self.ydata[:20])
        amp = np.max(self.ydata) - offset
        sigma = 0.2 * ((self.ydata - offset) >= 0.5 * amp).sum() * abs(self.dx)
        tau = 2 * sigma
        return x0, amp, sigma, tau, offset


class DoubleEMG(_1D):
    """ Fit the sum of two exponentially modified Gaussians to 1D data
    """
    @classmethod
    def func(cls, x, x0, a0, s0, t0, x1, a1, s1, t1, offset):
        """ The sum of two Gaussians convolved with two exponential decay curve"""
        assert x0 < x1
        assert s0 > 0.0
        assert s1 > 0.0
        assert t0 > 0.0
        assert t1 > 0.0
        def emg(mu, amp, sigma, tau):
            return amp \
                   * np.exp(0.5 * (2.0 * mu + sigma**2.0 / tau - 2.0 * x) / tau) \
                   * erfc((mu + sigma**2.0 / tau - x) / (2.0**0.5 * sigma))
        return emg(x0, a0, s0, t0) + emg(x1, a1, s1, t1) + offset

    def approx(self):
        """ estimate func pars (assumes positive amplitude)"""
        offset = np.mean(self.ydata[:20])
        rng = np.max(self.ydata) - offset
        mask = ((self.ydata - offset) >= 0.3 * rng)
        mid_point = ((np.max(self.xdata[mask]) - np.min(self.xdata[mask])) / 2
                     + np.min(self.xdata[mask]))
        sample0 = mask & (self.xdata < mid_point)
        sample1 = mask & (self.xdata > mid_point)
        x0 = np.median(self.xdata[sample0])
        x1 = np.median(self.xdata[sample1])
        a0 = np.max(self.ydata[sample0]) - offset
        a1 = np.max(self.ydata[sample1]) - offset
        s0 = np.std(self.xdata[sample0])
        s1 = np.std(self.xdata[sample1])
        t0 = 0.8 * s0
        t1 = 0.8 * s1
        return x0, a0, s0, t0, x1, a1, s1, t1, offset


class Pulse(_1D):
    """ Fit a square pulse to 1D data
    """
    @classmethod
    def func(cls, t, t0, width, amp, offset):
        """ A pulse waveform.      

        variables :
            t0         pulse on time
            width      pulse duration
            offset     data offset
        """
        pw = np.piecewise(t,
                          [t < t0,
                           (t0 <= t) & (t < t0 + width), 
                           (t0 + width <= t)],
                          [0.0, 1.0, 0.0])
        return pw * amp + offset 
    
    def approx(self):
        """ estimate func pars (positive or negative amplitudes)"""
        offset = np.mean(self.ydata[:3])
        rng_min = abs(np.min(self.ydata) - offset)
        rng_max = abs(np.max(self.ydata) - offset)
        if rng_min > rng_max:
            # negative
            amp = -rng_min
            mid_point = np.argmin(self.ydata)
        else:
            # positive
            amp = rng_max
            mid_point = np.argmax(self.ydata)
        t0 = self.xdata[np.argmin(abs(self.ydata[:mid_point] - offset - 0.5 * amp))]
        t1 = self.xdata[mid_point:][np.argmin(abs(self.ydata[mid_point:] - offset - 0.5 * amp))]
        width = t1 - t0
        return t0, width, amp, offset

class Trapezoid(_1D):
    """ Fit a trapezoidal pulse to 1D data
    """
    @classmethod
    def func(cls, t, t0, width, edge, amp, offset):
        """ A pulse waveform with finite rise and fall time.      

        variables :
            t0         pulse on time
            width      pulse duration
            edge       rise and fall time
            offset     data offset
        """
        assert edge <= width
        pw = np.piecewise(t,
                          [t < t0 - 0.5 * edge,
                           (t0 - 0.5 * edge <= t) & (t < t0 + 0.5 * edge),
                           (t0 + 0.5 * edge <= t) & (t < t0 - 0.5 * edge + width),
                           (t0 - 0.5 * edge + width <= t) & (t < t0 + 0.5 * edge + width), 
                           (t0 + 0.5 * edge + width <= t)],
                          [0.0,
                           lambda t: (t - t0 + 0.5 * edge) / edge,
                           1.0,
                           lambda t: 1.0 - (t - t0 + 0.5 * edge - width) / edge,
                           0.0])
        return pw * amp + offset 
    
    def approx(self):
        """ estimate func pars (positive or negative amplitudes)"""
        offset = np.mean(self.ydata[:3])
        rng_min = abs(np.min(self.ydata) - offset)
        rng_max = abs(np.max(self.ydata) - offset)
        if rng_min > rng_max:
            # negative
            amp = -rng_min
            mid_point = np.argmin(self.ydata)
        else:
            # positive
            amp = rng_max
            mid_point = np.argmax(self.ydata)
        t0 = self.xdata[np.argmin(abs(self.ydata[:mid_point] - offset - 0.5 * amp))]
        t1 = self.xdata[mid_point:][np.argmin(abs(self.ydata[mid_point:] - offset - 0.5 * amp))]
        width = t1 - t0
        edge = 0.1 * width
        return t0, width, edge, amp, offset


class BoxLucas(_1D):
    """ Fit a charging curve to 1D data
    """
    @classmethod
    def func(cls, t, tau, amp, offset):
        """ Charging curve
            
            amp * (1 - exp(- t / tau)) + offset

        """
        return offset + amp * (1 - np.exp(- t / tau))

    def approx(self):
        """ estimate func pars"""
        offset = np.mean(self.ydata[:3])
        amp = np.mean(self.ydata[-3:])
        tau = self.xdata[np.argmin(abs(self.ydata - offset - 0.63 * amp))]
        return tau, amp, offset

class Charge(_1D):
    """ Fit a capactivie charging curve to 1D data
    """
    @classmethod
    def func(cls, t, t0, tau, amp, offset):
        """ Capactivie charging curve
            
            t < t0 :
                offset
            t > t0 :
                amp * (1 - exp(-(t - t0) / tau) + offset

        """
        return offset + amp * ((1 - np.exp(- (t - t0) / tau))
                               * np.heaviside(t - t0, 0.5))

    def approx(self):
        """ estimate func pars (positive or negative amplitudes)"""
        offset = np.mean(self.ydata[:3])
        rng_min = abs(np.min(self.ydata) - offset)
        rng_max = abs(np.max(self.ydata) - offset)
        if rng_min > rng_max:
            amp = -rng_min
        else:
            amp = rng_max
        t0 = self.xdata[np.argmin(abs(self.ydata - offset - 0.05 * amp))]
        tau = 1.2 * (self.xdata[np.argmin(abs(self.ydata - offset - 0.5 * amp))] - t0)
        return t0, tau, amp, offset


class ChargeDecay(_1D):
    """ Fit a capactivie charge-decay curve to 1D data
    """
    @classmethod
    def func(cls, t, t0, width, tau, amp, offset):
        """ Capactivie charging and decay curve
        
            t < t0 :
                offset
            t0 < t < t0 + width :
                amp * (1 - exp(-(t - t0) / tau) + offset
            t > t0 + width :
                amp * exp(-(t - t0 - width) / tau) + offset
        
        """
        return offset + amp * ((1 - np.exp(- (t - t0) / tau))
                               * np.heaviside(t - t0, 0.5)
                               - (1 - np.exp(-(t - t0 - width) / tau))
                               * np.heaviside(t - t0 - width, 0.5))

    def approx(self):
        """ estimate func pars (positive or negative amplitudes)"""
        offset = np.mean(self.ydata[:3])
        rng_min = abs(np.min(self.ydata) - offset)
        rng_max = abs(np.max(self.ydata) - offset)
        if rng_min > rng_max:
            # negative
            amp = -rng_min
            mid_point = np.argmin(self.ydata)
        else:
            # positive
            amp = rng_max
            mid_point = np.argmax(self.ydata)
        t0 = self.xdata[np.argmin(abs(self.ydata[:mid_point] - offset - 0.05 * amp))]
        tau = 1.2 * (self.xdata[np.argmin(abs(self.ydata[:mid_point] - offset - 0.5 * amp))] - t0)
        width = 1.5 * (self.xdata[mid_point] - t0 - tau)
        return t0, width, tau, amp, offset


class Decay(_1D):
    """ Fit an exponential decay curve to 1D data
    """
    @classmethod
    def func(cls, t, amp, tau):
        """ Decay curve
               
        """
        return amp * np.exp(- (t / tau))

    def approx(self):
        """ estimate func pars (positive amplitude)"""
        amp = np.mean(self.ydata[:3])
        mid_point = int(len(self.xdata) / 2)
        tau = abs(self.xdata[mid_point] - self.xdata[0])
        return amp, tau

class DoubleDecay(_1D):
    """ Fit a double exponential decay curve to 1D trace data
    """
    @classmethod
    def func(cls, t, a0, tau0, a1, tau1):
        """ Double decay curve
               
        """
        assert tau1 > tau0 > 0.0
        def decay(amp, tau):
            """ decay curve """
            return amp * np.exp(- (t / tau))
        return decay(a0, tau0) + decay(a1, tau1)

    def approx(self):
        """ estimate func pars (positive amplitude)"""
        amp = np.mean(self.ydata[:3])
        a0 = 0.7 * amp
        a1 = 0.3 * amp
        mid_point = int(len(self.xdata) / 2)
        tau0 = 0.7 * abs(self.xdata[mid_point] - self.xdata[0])
        tau1 = 3.0 * tau0
        return a0, tau0, a1, tau1

class TripleDecay(_1D):
    """ Fit a triple exponential decay curve to 1D data
    """
    @classmethod
    def func(cls, t, a0, tau0, a1, tau1, a2, tau2):
        """ Triple decay curve
               
        """
        assert tau2 > tau1 > tau0 > 0.0
        def decay(amp, tau):
            """ decay curve """
            return amp * np.exp(- (t / tau))
        return decay(a0, tau0) + decay(a1, tau1) + decay(a2, tau2)

    def approx(self):
        """ estimate func pars (positive amplitude)"""
        amp = np.mean(self.ydata[:3])
        a0 = 0.6 * amp
        a1 = 0.4 * amp
        a2 = 0.2 * amp
        mid_point = int(len(self.xdata) / 2)
        tau0 = abs(self.xdata[mid_point] - self.xdata[0])
        tau1 = 2 * tau0
        tau2 = 2 * tau1
        return a0, tau0, a1, tau1, a2, tau2

""" 
    2D data
    -------
"""

class _2D(ABC):
    """ Fit to 2D image data
    """
    def __init__(self, *data):
        if len(data) == 3:
            X, Y, Z = data
            self.X = X
            self.Y = Y
            self.Z = Z
            self.xvals = X[0, :]
            self.yvals = Y[:, 0]
            self.dx = np.mean(np.diff(self.xvals))
            self.dy = np.mean(np.diff(self.yvals))
        else:
            self.Z = data[0]
            ny, nx = self.Z.shape
            self.xvals = np.arange(nx)
            self.yvals = np.arange(ny)
            self.X, self.Y = np.meshgrid(self.xvals, self.yvals)
            self.dx = self.dy = 1
        self.popt = None

    @classmethod
    @abstractmethod
    def func(cls, *args):
        pass

    @classmethod
    def apply_func(cls, X, Y, *params):
        """ Apply cls.func() to data. 
        
        Notes
        -----
        apply_func(*args) trys to return func(*args) but if that
        raises an Exception then an array of infs like X is returned
        instead. This allows assert statements defined in func to
        be used to constrain fit parameters. 
        """
        try:
            result = cls.func(X, Y, *params)
        except:
            result = np.full_like(X, np.inf)
        finally:
            return result

    @classmethod
    def get_signature(cls, **kwargs):
        return signature(cls.func, **kwargs)

    @classmethod
    def get_variables(cls, **kwargs):
        return tuple(cls.get_signature(**kwargs).parameters.keys())

    @abstractmethod
    def approx(self):
        pass

    def fit(self, p0=None, maxfev=100):
        """ least-squares fit of func() to data """
        if p0 is None:
            p0 = self.approx()
        popt, success = leastsq(lambda p: np.ravel(self.apply_func(self.X,
                                                                   self.Y,
                                                                   *p)
                                                   - self.Z),
                                p0, maxfev=maxfev)
        if not success:
            raise Exception("Failed to fit data")
        self.popt = popt
        return popt

    @property
    def best_fit(self):
        """ best fit vals """
        return self.func(self.X, self.Y, *self.popt)

    @property
    def residuals(self):
        """ best fit vals """
        return self.Z - self.best_fit


class Gauss2D(_2D):
    """ Fit a 2D Gaussian to image data
    """
    @classmethod
    def func(cls, x, y, x0, y0, amp, width, offset):
        """ 2D Gaussian function with offset"""
        return amp * np.exp(-0.5 *
                            (((x - x0) / width)**2.0 +
                             ((y - y0) / width)**2.0)) + offset

    def approx(self):
        """ estimate func pars """
        # gauss pars
        offset = np.mean(self.Z)
        amp = np.max(self.Z) - offset
        width = (0.5 * ((self.Z - offset >= 0.5 * amp).sum()**0.5)
                 * (abs(self.dx) + abs(self.dy)) / 2.0)
        # position
        i, j = np.median(np.argwhere(self.Z - offset >= 0.95 * amp),
                         axis=0).astype(int)
        x0 = self.X[i, j]
        y0 = self.Y[i, j]
        return x0, y0, amp, width, offset

    def text(self, pars=None):
        """ fit result """
        if pars is None:
            if self.popt is None:
                raise Exception("popt is None. Try fit() method.")
            else:
                pars = self.popt
        return f"xy = ({pars[0]:.2f}, {pars[1]:.2f}) \n" + \
               f"amplitude = {pars[2]:.2f} \n" + \
               f"FWHM = {2.35482 * pars[3]:.2f} \n" + \
               f"offset = {pars[4]:.2f}"


class Gauss2DAngle(_2D):
    """ Fit an asymmetric 2D Gaussian to image data
    """
    @classmethod
    def func(cls, x, y, x0, y0, amp, width, epsilon, angle, offset):
        """ Asymmetric 2D Gaussian function with offset and angle """
        w1 = width
        w2 = epsilon * width
        a = (np.cos(angle)**2.0 / (2.0 * w1**2.0)
             + np.sin(angle)**2.0 / (2.0 * w2**2.0))
        b = (np.sin(2.0 * angle) / (4.0 * w1**2.0)
             - np.sin(2.0 * angle) / (4.0 * w2**2.0))
        c = (np.sin(angle)**2.0 / (2.0 * w1**2.0)
             + np.cos(angle)**2.0 / (2.0 * w2**2.0))
        return amp * np.exp(-(a * (x - x0)**2.0
                              + 2.0 * b * (x - x0) * (y - y0)
                              + c * (y - y0)**2.0)) + offset

    def approx(self):
        """ estimate func pars """
        # gauss pars
        offset = np.mean(self.Z)
        amp = np.max(self.Z) - offset
        width = (0.35 * ((self.Z - offset >= 0.5 * amp).sum()**0.5)
                 * (abs(self.dx) + abs(self.dy)) / 2.0)
        epsilon = 1.5
        angle = -1.0
        # position
        i, j = np.median(np.argwhere(self.Z - offset >= 0.95 * amp),
                         axis=0).astype(int)
        x0 = self.X[i, j]
        y0 = self.Y[i, j]
        return x0, y0, amp, width, epsilon, angle, offset

    def text(self, pars=None):
        """ fit result """
        if pars is None:
            if self.popt is None:
                raise Exception("popt is None. Try .fit() method.")
            else:
                pars = self.popt
        w1 = pars[3]
        w2 = pars[3] * pars[4]
        w_eff = (w1 * w2)**0.5
        return f"xy = ({pars[0]:.2f}, {pars[1]:.2f}) \n" + \
               f"amplitude = {pars[2]:.2f} \n" + \
               f"FWHM = {2.35482 * w_eff:.2f} \n" + \
               f"epsilon = {pars[4]:.2f} \n" + \
               f"angle = {pars[5]:.2f} \n" + \
               f"offset = {pars[6]:.2f}"
