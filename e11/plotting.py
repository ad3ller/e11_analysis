# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:14:09 2018

@author: Adam

    misc. plotting tools

"""
import numpy as np
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
    kwargs = {**{"linewidth":2,
                 "linestyle":"-",
                 "marker":"",
                 "color":"black"}, **kwargs}
    if orientation == 'horizontal':
        lines = ax.plot(xs, ys, **kwargs)
    elif orientation == 'vertical':
        lines = ax.plot(ys, xs, **kwargs)
    else:
        raise ValueError("arg 'orientation' must be 'horizontal' or 'vertical'")
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


def subplots_xy(size=0.7, pad=0.15, xaxis="off", yaxis="off", **kwargs):
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
    ax.set_aspect(1.)
    divider = make_axes_locatable(ax)
    axx = divider.append_axes("top", size=size, pad=pad, sharex=ax)
    axy = divider.append_axes("right", size=size, pad=pad, sharey=ax)
    axx.axis(xaxis)
    axy.axis(yaxis)
    return fig, (ax, axx, axy)


def imshow_xy(axes, data, x, y, imshow=True, line=True, snap=False, step=True, 
              line_kw=None, step_kw=None, **kwargs):
    """ Show data as an image in ax, with slices at x and y in
    the subplots axy and axx.

            image = ax.imshow(data.T, origin="lower")
            step_x = axx.step(data[:, y])
            step_y = axy.step(data[x, :])        

        args:
            axes             # (ax, axx, axy)
            data             # 2D np.ndarray()
            x                # x position
            y                # y position
            line_kw=None     # dict() of kwargs for ax.axvline
            step_kw=None     # dict() of kwargs for axx.step()

        kwargs:
            passed to matplotlib.pyplot.imshow()

        returns:
            image, step_x, step_y
    """
    ax, axx, axy = axes
    nx, ny = np.shape(data)
    # xy slices
    if step:
        if step_kw is None:
            step_kw = {}
        # position
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
        yvals = np.linspace(ymin, ymax, ny)
        # xy step plots
        step_x = step_plot(axx, xvals, data[:, yi], orientation="horizontal", **step_kw)
        step_y = step_plot(axy, yvals, data[xi, :], orientation="vertical", **step_kw)
    # scale
    rng = np.max(data) - np.min(data)
    vmin = kwargs.get("vmin", np.min(data) - 0.05 * rng)
    vmax = kwargs.get("vmax", np.max(data) + 0.05 * rng)
    axx.set_ylim(vmin, vmax)
    axy.set_xlim(vmin, vmax)
    # plot image
    if imshow:
        if "origin" not in kwargs:
            kwargs["origin"] = "lower"
        image = ax.imshow(data.T, **kwargs)
    else:
        image = None
    # xy lines
    if line:
        if line_kw is None:
            line_kw = {}
        line_kw = {**{"alpha":0.8, "color":'black'}, **line_kw}
        if snap:
            x = np.round((x - xmin) / dx) * dx + xmin
            y = np.round((y - ymin) / dy) * dy + ymin
        ax.axvline(x, **line_kw)
        ax.axhline(y, **line_kw) 
    return image, step_x, step_y
