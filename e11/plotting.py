# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:14:09 2018

@author: Adam

    misc. plotting tools

"""
import numpy as np


def floating_xticks(ax, xticks, y_pos, length, **kwargs):
    """ draw ticks inside ax at y=pos.
    """
    if "color" not in kwargs:
        kwargs["color"] = "k"
    if "lw" not in kwargs:
        kwargs["lw"] = 1.0
    ax.plot([np.min(xticks), np.max(xticks)], [y_pos, y_pos], **kwargs)
    for tx in xticks:
        ax.plot([tx, tx], [y_pos, y_pos + length], **kwargs)
    return ax


def floating_xlabels(ax, xticks, labels, y_pos, **kwargs):
    """ draw labels at x=ticks inside ax at y=y_pos.
    """
    if "ha" not in kwargs:
        kwargs["ha"] = "center"
    for tx, lbl in zip(xticks, labels):
        ax.text(tx, y_pos, str(lbl), **kwargs)
    return ax
