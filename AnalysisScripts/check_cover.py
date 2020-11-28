# Copyright (C) 2020 Sorbonne University
# Maintainer: Achkan Salehi (salehi@isir.upmc.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb

import scipy.stats as stats
import scipy.special
import math
from collections import namedtuple

sys.path.append("../NS/")
sys.path.append("..")
from NS import MiscUtils
from NS import Agents

import matplotlib
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def check_cover_deceptive_maze(root_dir,
                               num_gens_to_check=400,
                               G=6,
                               h=600,
                               w=600):
    """
    G   grid size, applies to both vertical and horizontal
    """

    grid = np.zeros([G, G])
    ratio_h = h // G
    ratio_w = w // G
    cover_hist = []
    stride = 1
    for i in range(0, num_gens_to_check, stride):
        if i % 10 == 0:
            print("i==", i)
        with open(root_dir + f"/population_gen_{i}", "rb") as fl:
            pop = pickle.load(fl)

            for x in pop:
                bd_y, bd_x = x._behavior_descr[0].tolist()
                bd_y = h - bd_y
                h_i = int(bd_x // ratio_h)
                v_i = int(bd_y // ratio_w)
                grid[h_i, v_i] = 1

            bds_i = [x._behavior_descr for x in pop]
            bds_i = np.concatenate(bds_i, 0)
            bds_i[:, 1] = 600 - bds_i[:, 1]

            plt.scatter(bds_i[:, 0],
                        bds_i[:, 1],
                        s=80,
                        facecolors="none",
                        edgecolors="r")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(0, 600)
            plt.ylim(0, 600)
            plt.grid("on", alpha=3)

            cover_i = (grid != 0).sum() / (G * G)
            cover_hist.append(cover_i)

    ax = plt.gca()
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    ax.xaxis.set_major_locator(MultipleLocator(int(600 // G)))
    ax.yaxis.set_major_locator(MultipleLocator(int(600 // G)))
    plt.grid("on")
    plt.show()

    return grid, cover_hist
