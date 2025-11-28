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


def boxplot(ax, data, title, ylabel, labels=None, patch_artist=True):
    boxplot = ax.boxplot(data, labels=labels, patch_artist=patch_artist)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


def scatter(ax, x, y, title, xlabel, ylabel, marker="o"):
    ax.scatter(x, y, marker=marker)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return ax