# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:06:21 2018

@author: Adam

    fitting tools

"""
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
        self.xdata = xdata
        self.ydata = ydata
        self.sigma = sigma
        self.popt = None
        self.pcov = None
        self.perr = None

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

    @property
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def func(self, *vars):
        pass

    @abstractmethod
    def approx(self):
        pass


class Gaussian(_1D):
    """ Fit a 1D Gaussian to trace data
    """
    @property
    def variables(self):
        return ["x0", "amp", "sigma", "offset"]

    def func(self, x, x0, amp, sigma, offset):
        """ 1D Gaussian function with offset"""
        assert sigma >= 0.0
        return amp * np.exp(-0.5 * ((x - x0) / sigma)**2.0) + offset

    def approx(self):
        """ estimate func pars (assumes positive amplitude)"""
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.min(self.ydata)
        amp = np.max(self.ydata) - offset
        dx = np.mean(np.diff(self.xdata))
        sigma = ((self.ydata - offset) >= 0.5 * amp).sum() * dx
        return x0, amp, sigma, offset


class Lorentzian(_1D):
    """ Fit a 1D Lorentzian to trace data
    """
    @property
    def variables(self):
        return ["x0", "amp", "gamma", "offset"]

    def func(self, x, x0, amp, gamma, offset):
        """ 1D Lorentzian function """
        assert gamma >= 0.0
        return amp * gamma**2 / ((x - x0)**2 + gamma**2) + offset

    def approx(self):
        """ estimate func pars (assumes positive amplitude)"""
        x0 = self.xdata[np.argmax(self.ydata)]
        offset = np.min(self.ydata)
        amp = np.max(self.ydata) - offset
        dx = np.mean(np.diff(self.xdata))
        gamma = ((self.ydata - offset) >= 0.5 * amp).sum() * dx
        return x0, amp, gamma, offset


class EMG(_1D):
    """ Fit an exponentially modified Gaussian to 1D trace data
    """
    @property
    def variables(self):
        return ["x0", "amp", "sigma", "tau", "offset"]

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
        offset = np.min(self.ydata)
        amp = np.max(self.ydata) - offset
        dx = np.mean(np.diff(self.xdata))
        sigma = 0.2 * ((self.ydata - offset) >= 0.5 * amp).sum() * dx
        tau = 2 * sigma
        return x0, amp, sigma, tau, offset


class DoubleEMG(_1D):
    """ Fit the sum of two exponentially modified Gaussians to 1D trace data
    """
    @property
    def variables(self):
        return ["x0", "a0", "s0", "t0", "x1", "a1", "s1", "t1", "offset"]

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
        else:
            self.Z = data[0]
            nx, ny = self.Z.shape
            xvals = np.arange(nx)
            yvals = np.arange(ny)
            self.X, self.Y = np.meshgrid(xvals, yvals, indexing="ij")
        self.dx = np.mean(np.diff(self.X[:, 0]))
        self.dy = np.mean(np.diff(self.Y[0, :]))
        self.popt = None

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

    @property
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def func(self):
        pass

    @abstractmethod
    def approx(self):
        pass


class Gauss2D(_2D):
    """ Fit a 2D Gaussian to image data
    """
    @property
    def variables(self):
        return ["x0", "y0", "amp", "width", "offset"]

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
    @property
    def variables(self):
        return ["x0", "y0", "amp", "width", "epsilon", "angle", "offset"]

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
