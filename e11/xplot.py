# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:14:09 2018

@author: Adam

    xarray plotting tools

"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from .plot import step_plot, autorange_xy

def ximshow_xy(axes, coord, data,
               xy_limits=None, xy_pad=0.02, add_lines=False,
               step_kw=None, line_kw=None, **kwargs):
    """ Show data as an image in ax, with slices at x and y in
    the subplots axy and axx.

            img = ax.imshow(data)
            step_x = axx.step(data[coord[1], :])
            step_y = axy.step(data[:, coord[0]])

        args:
            axes             # (ax, axx, axy)
            coord            # position (x, y)
            data             # xarray.DataArray
            xy_limits=None   # None or "auto" / (min, max) / "data" / "match" / "clim"
            xy_pad=0.02      # padding of the xy vertical range
                             # (active for xy_limits="data" or "clim")
            add_lines=False  # mark coords with lines
            line_kw=None     # dict() of kwargs for ax.axhline()
            step_kw=None     # dict() of kwargs for axx.step()

        kwargs:
            passed to xarray.plot.imshow()

        returns:
            img, step_x, step_y
    """
    ax, axx, axy = axes
    x, y = coord
    # xy slices
    if isinstance(data, xr.DataArray):
        xdim = kwargs.pop('x', data.dims[1])
        ydim = kwargs.pop('y', data.dims[0])
        if xdim in data.coords:
            xvals = data.coords[xdim]
            ydata = data.sel({xdim:x}, method="nearest")
        else:
            xvals = np.arange(data.sizes[xdim])
            ydata = data.isel({xdim:int(np.round(x))})
        if ydim in data.coords:
            yvals = data.coords[ydim]
            xdata = data.sel({ydim:y}, method="nearest")
        else:
            yvals = np.arange(data.sizes[ydim])
            xdata = data.isel({ydim:int(np.round(y))})
    else:
        raise TypeError("data must be an xarray.DataArray")
    # xy step plots
    if step_kw is None:
        step_kw = {}
    step_x = step_plot(axx, xvals, xdata,
                       orientation="horizontal",
                       **step_kw)
    step_y = step_plot(axy, yvals, ydata,
                       orientation="vertical",
                       **step_kw)
    # plot image
    kwargs = {**{"add_colorbar":False}, **kwargs}
    img = data.plot.imshow(x=xdim, y=ydim, ax=ax, **kwargs)
    # lines
    if add_lines:
        if line_kw is None:
            line_kw = {}
        ax.axvline(coord[0], **line_kw)
        ax.axhline(coord[1], **line_kw)
    # autorange
    axx, axy = autorange_xy(img, axx, axy, data, xy_limits, xy_pad)
    return img, step_x, step_y


def xcontour_xy(axes, coord, data,
                fill=False, xy_limits=None, xy_pad=0.02, add_lines=False,
                plot_kw=None, line_kw=None, **kwargs):
    """ Show data as an contour plot in ax, with slices at x and y in
    the subplots axy and axx.

            cnt = ax.contour(data)
            plot_x = axx.plot(data[coord[1], :])
            plot_y = axy.plot(data[:, coord[0]])

        args:
            axes             # (ax, axx, axy)
            coord            # position (x, y)
            data             # xarray.DataArray
            xy_limits=None   # None or "auto" / (min, max) / "data" / "match" / "clim"
            xy_pad=0.02      # padding of the xy vertical range
                             # (active for xy_limits="data" or "clim")
            add_lines=False  # mark coords with lines
            line_kw=None     # dict() of kwargs for ax.axhline()
            plot_kw=None     # dict() of kwargs for axx.plot()

        kwargs:
            passed to xarray.plot.imshow()

        returns:
            cnt, plot_x, plot_y
    """
    ax, axx, axy = axes
    x, y = coord
    if plot_kw is None:
        plot_kw = {}
    # Z
    if isinstance(data, xr.DataArray):
        xdim = kwargs.pop('x', data.dims[1])
        ydim = kwargs.pop('y', data.dims[0])
        if xdim in data.coords:
            xvals = data.coords[xdim]
            ydata = data.sel({xdim:x}, method="nearest")
        else:
            xvals = np.arange(data.sizes[xdim])
            ydata = data.isel({xdim:int(np.round(x))})
        if ydim in data.coords:
            yvals = data.coords[ydim]
            xdata = data.sel({ydim:y}, method="nearest")
        else:
            yvals = np.arange(data.sizes[ydim])
            xdata = data.isel({ydim:int(np.round(y))})
        # xy plot
        plot_x = axx.plot(xvals, xdata, **plot_kw)
        plot_y = axy.plot(ydata, yvals, **plot_kw)
        kwargs = {**{"add_colorbar":False}, **kwargs}
        if fill:
            cnt = data.plot.contourf(x=xdim, y=ydim, ax=ax, **kwargs)
        else:
            cnt = data.plot.contour(x=xdim, y=ydim, ax=ax, **kwargs)
    else:
        raise TypeError("data must an xarray.DataArray")
    # lines
    if add_lines:
        if line_kw is None:
            line_kw = {}
        ax.axvline(coord[0], **line_kw)
        ax.axhline(coord[1], **line_kw)
    # autorange
    axx, axy = autorange_xy(cnt, axx, axy, data, xy_limits, xy_pad)
    return cnt, plot_x, plot_y
