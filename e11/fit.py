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
    """ Fit a 1D function to trace data
    """
    def __init__(self, xdata, ydata, sigma=None):
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        self.sigma = sigma
        self.popt = None
        self.pcov = None
        self.perr = None
        # func
        self.sig = signature(self.func)
        self.variables = tuple(self.sig.parameters.keys())[1:]

    def robust_func(self, data, *params):
        """ errors -> infs """
        try:
            result = self.func(data, *params)
        except:
            result = np.full_like(data, np.inf)
        finally:
            return result

    def fit(self, p0=None, uncertainty=True, **kwargs):
        """ least-squares fit of func() to data """
        sigma = kwargs.pop("sigma", self.sigma)
        if p0 is None:
            p0 = self.approx()
        self.popt, self.pcov = curve_fit(self.robust_func,
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
            names = self.variables
        if uncertainty and self.perr is not None:
            return dict(zip(names, zip(self.popt, self.perr)))
        else:
            return dict(zip(names, self.popt))

    @abstractmethod
    def func(self, *vars):
        pass

    @abstractmethod
    def approx(self):
        pass


class Gaussian(_1D):
    """ Fit a 1D Gaussian to trace data
    """
    def func(self, x, x0, amp, sigma, offset):
        """ 1D Gaussian function with offset"""
        assert sigma >= 0.0
        return amp * np.exp(-0.5 * ((x - x0) / sigma)**2.0) + offset

    def approx(self):
        """ estimate func pars (assumes positive amplitude) """
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.min(self.ydata)
        amp = np.max(self.ydata) - offset
        dx = np.mean(np.diff(self.xdata))
        sigma = ((self.ydata - offset) >= 0.5 * amp).sum() * dx
        return x0, amp, sigma, offset


class Lorentzian(_1D):
    """ Fit a 1D Lorentzian to trace data
    """
    def func(self, x, x0, amp, gamma, offset):
        """ 1D Lorentzian function """
        assert gamma >= 0.0
        return amp * gamma**2 / ((x - x0)**2 + gamma**2) + offset

    def approx(self):
        """ estimate func pars (assumes positive amplitude)"""
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.min(self.ydata)
        amp = np.max(self.ydata) - offset
        dx = abs(np.mean(np.diff(self.xdata)))
        gamma = ((self.ydata - offset) >= 0.5 * amp).sum() * dx
        return x0, amp, gamma, offset


class EMG(_1D):
    """ Fit an exponentially modified Gaussian to 1D trace data
    """
    def func(self, x, x0, amp, sigma, tau, offset):
        """ 1D Gaussian convolved with an exponential decay (tau)"""
        assert sigma >= 0.0
        assert tau >= 0.0
        return amp \
               * np.exp(0.5 * (2.0 * x0 + sigma**2.0 / tau - 2 * x) / tau) \
               * erfc((x0 + sigma**2.0 / tau - x) / (2.0**0.5 * sigma)) \
               + offset

    def approx(self):
        """ estimate func pars (assumes positive amplitude)"""
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.mean(self.ydata[:20])
        amp = np.max(self.ydata) - offset
        dx = np.mean(np.diff(self.xdata))
        sigma = 0.2 * ((self.ydata - offset) >= 0.5 * amp).sum() * dx
        tau = 2 * sigma
        return x0, amp, sigma, tau, offset


class DoubleEMG(_1D):
    """ Fit the sum of two exponentially modified Gaussians to 1D trace data
    """
    def func(self, x, x0, a0, s0, t0, x1, a1, s1, t1, offset):
        """ The sum of two Gaussians convolved with an exponential decay curve"""
        assert x0 < x1
        assert s0 >= 0.0
        assert s1 >= 0.0
        assert t0 >= 0.0
        assert t1 >= 0.0
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
    """ Fit a trapezoidal pulse to 1D trace data
    """
    def func(self, t, t0, width, edge, amp, offset):
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
                          [0,
                           lambda t: (t - t0 + 0.5 * edge) / edge,
                           1,
                           lambda t: 1 - (t - t0 + 0.5*edge - width) / edge,
                           0])
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


class Charge(_1D):
    """ Fit a capactivie charging curve to 1D trace data
    """
    def func(self, t, t0, tau, amp, offset):
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
    """ Fit a capactivie charge-decay curve to 1D trace data
    """
    def func(self, t, t0, width, tau, amp, offset):
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
        else:
            self.Z = data[0]
            ny, nx = self.Z.shape
            self.xvals = np.arange(nx)
            self.yvals = np.arange(ny)
            self.X, self.Y = np.meshgrid(self.xvals, self.yvals)
        self.dx = np.mean(np.diff(self.xvals))
        self.dy = np.mean(np.diff(self.yvals))
        self.popt = None
        self.sig = signature(self.func)
        self.variables = list(self.sig.parameters.keys())[2:]

    def fit(self, p0=None, maxfev=100):
        """ least-squares fit of func() to data """
        if p0 is None:
            p0 = self.approx()
        popt, success = leastsq(lambda p: np.ravel(self.func(self.X,
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

    @abstractmethod
    def func(self):
        pass

    @abstractmethod
    def approx(self):
        pass


class Gauss2D(_2D):
    """ Fit a 2D Gaussian to image data
    """
    def func(self, x, y, x0, y0, amp, width, offset):
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
                 * (self.dx + self.dy) / 2.0)
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
    def func(self, x, y, x0, y0, amp, width, epsilon, angle, offset):
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
                 * (self.dx + self.dy) / 2.0)
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
