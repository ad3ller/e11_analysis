# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:06:21 2018

@author: Adam

    fitting tools

"""
import numpy as np
from scipy.optimize import leastsq


class Gauss2D(object):
    """ Fit a 2D Gaussian to image data
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


class Gauss2DAngle(object):
    """ Fit an asymmetric 2D Gaussian to image data
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
        angle = - 1.0
        # position
        i, j = np.median(np.argwhere(self.Z - offset >= 0.95 * amp),
                         axis=0).astype(int)
        x0 = self.X[i, j]
        y0 = self.Y[i, j]
        return x0, y0, amp, width, epsilon, angle, offset

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

    def text(self, pars=None):
        """ fit result """
        if pars is None:
            if self.popt is None:
                raise Exception("popt is None. Try .fit() method.")
            else:
                pars = self.popt
        w1 = pars[3]
        w2 = pars[3] * pars[4]
        w_av = (w1 + w2) / 2.0
        return f"xy = ({pars[0]:.2f}, {pars[1]:.2f}) \n" + \
               f"amplitude = {pars[2]:.2f} \n" + \
               f"FWHM = {2.35482 * w_av:.2f} \n" + \
               f"epsilon = {pars[4]:.2f} \n" + \
               f"angle = {pars[5]:.2f} \n" + \
               f"offset = {pars[6]:.2f}"
