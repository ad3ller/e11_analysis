# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:14:09 2018

@author: Adam

    misc. plotting tools

"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_step_edges(xdata, ydata):
    """ Convert xdata and ydata into step edges (xs ys) with
    interleaved values at the midpoints in x.

        args:
            xdata
            ydata

        returns:
            xs, xy
    """
    assert len(xdata) == len(ydata), "length of xdata and ydata must match"
    # initialise
    xs = np.empty(3 * len(xdata) - 2)
    ys = np.empty(len(xs))
    # xdata
    xs[0::3] = xdata
    xs[1::3] = xs[2::3] = xdata[:-1] + 0.5 * np.diff(xdata)
    # ydata
    ys[0::3] = ydata
    ys[1::3] = ydata[:-1]
    ys[2::3] = ydata[1:]
    return xs, ys


def step_plot(ax, xdata, ydata, orientation='horizontal', **kwargs):
    """ Step plot.

        args:
            ax                (matplotlib.axes._subplots.AxesSubplot)
            xdata             (numpy.ndarray)
            ydata             (numpy.ndarray)
            orientation       'horizontal' / 'vertical' (str)

        kwargs:
            passed to matplotlib.pyplot.plot()

        returns:
            matplotlib.pyplot.plot()

        notes:
            Simular to matplotlib.pyplot.step() but with the option
            of plotting with vertical orientation.

    """
    xs, ys = get_step_edges(xdata, ydata)
    # defaults
    kwargs = {**{"linewidth": 2,
                 "linestyle": "-",
                 "marker": "",
                 "color": "black"}, **kwargs}
    if orientation == "horizontal":
        lines = ax.plot(xs, ys, **kwargs)
    elif orientation == "vertical":
        lines = ax.plot(ys, xs, **kwargs)
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    return lines


def floating_xticks(ax, xticks, y_pos, length, **kwargs):
    """ Draw ticks inside ax at y=pos.

        args:
            ax          (matplotlib.axes._subplots.AxesSubplot)
            xticks      (list / np.ndarray)
            y_pos       (float)
            length      (float)

        kwargs:
            passed to matplotlib.pyplot.plot

        returns:
            ax
    """
    if "color" not in kwargs:
        kwargs["color"] = "k"
    if "lw" not in kwargs and "linewidth" not in kwargs:
        kwargs["lw"] = 1.0
    ax.plot([np.min(xticks), np.max(xticks)], [y_pos, y_pos], **kwargs)
    for tx in xticks:
        ax.plot([tx, tx], [y_pos, y_pos + length], **kwargs)
    return ax


def floating_xlabels(ax, xticks, labels, y_pos, **kwargs):
    """ Write labels at x=xticks inside ax at y=y_pos.

        args:
            ax          (matplotlib.axes._subplots.AxesSubplot)
            xticks      (list / np.ndarray)
            labels      (list / np.ndarray)
            y_pos       (float)

        kwargs:
            passed to matplotlib.pyplot.text

        returns:
            ax
    """
    if "ha" not in kwargs:
        kwargs["ha"] = "center"
    for tx, lbl in zip(xticks, labels):
        ax.text(tx, y_pos, str(lbl), **kwargs)
    return ax


def top_xticks(ax, minor=None, major=None, labels=None):
    """
    Add top axis to ax with custom major / minor ticks.
       
       args:
           minor      # list() or np.ndarray() minor xtick locations
           major      # None, list() or np.ndarray() major xtick locations
           labels     # None, list() or np.ndarray() major xtick labels
    
        returns:
            ax_top
            
        example:
        
            import numpy as np
            import matplotlib.pyplot as plt
            from positronium import Bohr

            # minor ticks
            nvals = np.arange(10, 200)
            minor = Bohr.energy(2, nvals, unit="nm")

            # major ticks
            nvals = [10, 12, 15, 19, 24, 32, np.inf]
            labels = map(lambda x: r"$\infty$" if x == np.inf else f"{x:d}", nvals)
            major = Bohr.energy(2, nvals, unit="nm")

            # plot
            fig, ax = plt.subplots()

            ax_top = top_xticks(ax, minor, major, labels)
            ax_top.grid(which="both", axis="x", zorder=-20, alpha=0.2)
            
            ax.set_xlim(728, 762)
            plt.show()
                
    """
    ax_top = ax.twiny()
    ax.get_shared_x_axes().join(ax, ax_top)
    
    # major
    if major is not None:
        ax_top.set_xticks(major)
    if labels is not None:
        ax_top.xaxis.set_ticklabels(labels)
    
    # minor 
    if minor is not None:
        ax_top.set_xticks(minor, minor=True)
    
    ax_top.set_xbound(ax.get_xbound())
    return ax_top


def subplots_xy(size=0.7, pad=0.15, aspect="auto", 
                xaxis="off", yaxis="off", **kwargs):
    """ Create a figure with three subplots, ax, axx, and, axy.

        The subplots axx and axy are positioned at the top and
        right of the main subplot, ax.

        args:
            size=0.7         # size of the subplots
            pad=0.15         # subplot padding
            xaxis="off"      # toggle axx axis
            yaxis="off"      # toggle axy axis

        kwargs:
            passed to matplotlib.pyplot.subplot()

        returns:
            ax, axx, axy
    """
    fig, ax = plt.subplots(**kwargs)
    divider = make_axes_locatable(ax)
    axx = divider.append_axes("top", size=size, pad=pad, sharex=ax)
    axy = divider.append_axes("right", size=size, pad=pad, sharey=ax)
    axx.axis(xaxis)
    axy.axis(yaxis)
    ax.set_aspect(aspect)
    return fig, (ax, axx, axy)

def xy_autorange(img, axx, axy, data, xy_limits, xy_pad):
    """ Adjust axx and axy vertical range.

        xy_limits:
            None or "auto"   # matplotlib default range
            "data"           # vrange to min and max of data -/+ xy_pad
            "match"          # vrange to min and max of default ranges of each
            "clim"           # vrange to img.clim() -/+ xy_pad

        args:
            axx              # horizontal axis
            axy              # vertical axis
            data             # 2D numpy.ndarray
            xy_limits        # None or "auto" / "data" / "match" / "clim"
            xy_pad           # padding of the xy vertical range
                             # (active for xy_limits="data" or "clim")
        
        returns:
            axx, axy
    """
        # axx vertical range
    if xy_limits is None or xy_limits == "auto":
        pass
    elif xy_limits == "data":
        # edge plots range to match padded data range
        rng = data.max() - data.min()
        limits = (data.min() - xy_pad * rng,
                  data.max() + xy_pad * rng)
        axx.set_ylim(*limits)
        axy.set_xlim(*limits)
    elif xy_limits == "match":
        # edge plots range to match each other
        limits = (min(axx.get_ylim()[0], axy.get_xlim()[0]),
                  max(axx.get_ylim()[1], axy.get_xlim()[1]))
        axx.set_ylim(*limits)
        axy.set_xlim(*limits)
    elif xy_limits == "clim":
        # edge plots range to match padded image clim
        clim = img.get_clim()
        rng = clim[1] - clim[0]
        limits = (clim[0] - xy_pad * rng,
                  clim[1] + xy_pad * rng)
        axx.set_ylim(*limits)
        axy.set_xlim(*limits)
    else:
        raise ValueError(f"Invalid value for `xy_limits`={xy_limits}")
    return axx, axy


def imshow_xy(axes, coord, data,
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
            data             # 2D numpy.ndarray or xarray.DataArray
            xy_limits=None   # None or "auto" / "data" / "match" / "clim"
            xy_pad=0.02      # padding of the xy vertical range
                             # (active for xy_limits="data" or "clim")
            add_lines=False  # mark coords with lines
            line_kw=None     # dict() of kwargs for ax.axhline()
            step_kw=None     # dict() of kwargs for axx.step()

        kwargs:
            passed to matplotlib.pyplot.imshow() or xarray.plot.imshow()

        returns:
            img, step_x, step_y
    """
    ax, axx, axy = axes
    x, y = coord
    # xy slices
    if isinstance(data, np.ndarray):
        assert data.ndim == 2
        ny, nx = np.shape(data)
        if "extent" in kwargs:
            xmin, xmax, ymin, ymax = kwargs["extent"]
            dx = (xmax - xmin) / nx
            xmin += dx / 2.0
            xmax -= dx / 2.0
            dy = (ymax - ymin) / ny
            ymin += dy / 2.0
            ymax -= dy / 2.0
            xi = int(np.round((x - xmin) / dx))
            yi = int(np.round((y - ymin) / dy))
        else:
            xmin, xmax, ymin, ymax = 0, nx - 1, 0, ny - 1
            dx = dy = 1
            xi, yi = int(np.round(x)), int(np.round(y))
        xvals = np.linspace(xmin, xmax, nx)
        xdata = data[yi, :]
        yvals = np.linspace(ymin, ymax, ny)
        ydata = data[:, xi]
    elif isinstance(data, xr.DataArray):
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
        raise TypeError("data must be a 2D numpy array or an xarray object")
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
    if isinstance(data, xr.DataArray):
        kwargs = {**{"add_colorbar":False}, **kwargs}
        img = data.plot.imshow(x=xdim, y=ydim, ax=ax, **kwargs)
    else:    
        img = ax.imshow(data, **kwargs)
    # lines
    if add_lines:
        if line_kw is None:
            line_kw = {}
        ax.axvline(coord[0], **line_kw)
        ax.axhline(coord[1], **line_kw)
    # autorange
    axx, axy = xy_autorange(img, axx, axy, data, xy_limits, xy_pad)
    return img, step_x, step_y


def contour_xy(axes, coord, *data,
               fill=False, xy_limits=None, xy_pad=0.02, add_lines=False,
               plot_kw=None, line_kw=None, **kwargs):
    """ Show data as an contour plot in ax, with slices at x and y in
    the subplots axy and axx.

            cnt = ax.contour(*data)
            plot_x = axx.plot(data[coord[1], :])
            plot_y = axy.plot(data[:, coord[0]])

        args:
            axes             # (ax, axx, axy)
            coord            # position (x, y)
            data             # [X, Y,] Z of 2D numpy.ndarrays
                             # or xarray.DataArray
            xy_limits=None   # None or "auto" / "data" / "match" / "clim"
            xy_pad=0.02      # padding of the xy vertical range
                             # (active for xy_limits="data" or "clim")
            add_lines=False  # mark coords with lines
            line_kw=None     # dict() of kwargs for ax.axhline()
            plot_kw=None     # dict() of kwargs for axx.plot()

        kwargs:
            passed to matplotlib.pyplot.imshow() or xarray.plot.imshow()

        returns:
            cnt, step_x, step_y
    """
    ax, axx, axy = axes
    x, y = coord
    if plot_kw is None:
        plot_kw = {}
    # Z
    if len(data) == 1 and isinstance(data[0], np.ndarray):
        data = data[0]
        assert data.ndim == 2
        ny, nx = np.shape(data)
        if "extent" in kwargs:
            xmin, xmax, ymin, ymax = kwargs["extent"]
            dx = (xmax - xmin) / nx
            xmin += dx / 2.0
            xmax -= dx / 2.0
            dy = (ymax - ymin) / ny
            ymin += dy / 2.0
            ymax -= dy / 2.0
            xi = int(np.round((x - xmin) / dx))
            yi = int(np.round((y - ymin) / dy))
        else:
            xmin, xmax, ymin, ymax = 0, nx - 1, 0, ny - 1
            dx = dy = 1
            xi, yi = int(np.round(x)), int(np.round(y))
        xvals = np.linspace(xmin, xmax, nx)
        xdata = data[yi, :]
        yvals = np.linspace(ymin, ymax, ny)
        ydata = data[:, xi]
        # xy plot
        plot_x = axx.plot(xvals, xdata, **plot_kw)
        plot_y = axy.plot(ydata, yvals, **plot_kw)
        # main plot
        if fill:
            cnt = ax.contourf(data, **kwargs)
        else:
            cnt = ax.contour(data, **kwargs)
    # X, Y, Z
    elif len(data) == 3 and all([isinstance(el, np.ndarray) for el in data]):
        # xy slices
        xvals = data[0][0, :]
        yvals = data[1][:, 0]
        xi = abs(x - xvals).argmin()
        yi = abs(y - yvals).argmin()
        # xy step plots
        plot_x = axx.plot(xvals, data[2][yi, :], **plot_kw)
        plot_y = axy.plot(data[2][:, xi], yvals, **plot_kw)
        # main plot
        if fill:
            cnt = ax.contourf(*data, **kwargs)
        else:
            cnt = ax.contour(*data, **kwargs)
        data = data[2]
    # xarray.DataArray
    elif len(data) == 1 and isinstance(data[0], xr.DataArray):
        data = data[0]
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
        raise TypeError("data must one or three 2D numpy arrays or an xarray object")
    # lines
    if add_lines:
        if line_kw is None:
            line_kw = {}
        ax.axvline(coord[0], **line_kw)
        ax.axhline(coord[1], **line_kw)
    # autorange
    axx, axy = xy_autorange(cnt, axx, axy, data, xy_limits, xy_pad)
    return cnt, plot_x, plot_y
