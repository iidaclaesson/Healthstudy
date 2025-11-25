import matplotlib.pyplot as plt
import pandas as pd



def hist(ax, data, title, xlabel, ylabel, bins=30, edgecolor="black"):
    ax.hist(data, bins=bins, edgecolor=edgecolor)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def bar(ax, x, y, title, xlabel, ylabel, edgecolor="black", width=0.6):
    ax.bar(x, y, edgecolor=edgecolor, width=width)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


    