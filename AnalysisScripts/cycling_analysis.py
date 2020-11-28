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
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import os

from sklearn.neighbors import KDTree
import scipy.stats as stats
import scipy.special
import math
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

sys.path.append("../NS/")
sys.path.append("..")
from NS import MiscUtils


def analyse_cycling_behavior_archive_based(root_dir):

    #skip around 5 as otherwise it takes ages to compute
    generations_to_use = list(range(1, 1000, 10))
    populations = []  #those will actually store behavior descriptors
    archives = []
    unused = []
    for i in generations_to_use:
        with open(root_dir + "/" + f"archive_{i}", "rb") as fl:
            agents = pickle.load(fl)
            bds = [x._behavior_descr for x in agents]
            if not len(bds):
                unused.append(i)
                continue
            archives.append(np.concatenate(bds, 0))

        with open(root_dir + "/" + f"population_gen_{i}", "rb") as fl:
            agents = pickle.load(fl)
            bds = [x._behavior_descr for x in agents]
            populations.append(np.concatenate(bds, 0))

            task_solvers = [x for x in agents if x._solved_task]
            if len(task_solvers):
                print("solvers found at gen ", i)

    generations_to_use = [x for x in generations_to_use if x not in unused]

    Qmat = np.zeros([len(generations_to_use), len(generations_to_use)])

    for i in range(len(generations_to_use)):
        if i % 5 == 0:
            print("i==", i)
        for j in range(i, len(generations_to_use)):

            bds_ij = np.concatenate([populations[i], archives[j]], 0)
            kdt = KDTree(bds_ij, leaf_size=20, metric='euclidean')

            dists, ids = kdt.query(populations[i],
                                   min(15, bds_ij.shape[0]),
                                   return_distance=True)
            dists = dists[:, 1:]
            novs = dists.mean(1)
            Qmat[i, j] = novs.mean()

    return Qmat


def analyse_cycling_behavior_learnt_nov(root_dir, in_dim, out_dim):

    frozen_net_path = root_dir + "/frozen_net.model"
    frozen = MiscUtils.SmallEncoder1d(in_dim,
                                      out_dim,
                                      num_hidden=3,
                                      non_lin="leaky_relu",
                                      use_bn=False)
    frozen.load_state_dict(torch.load(frozen_net_path))
    frozen.eval()

    learned_model_generations = list(range(0, 2000, 1))
    learned_models_paths = [
        root_dir + f"/learnt_{i}.model" for i in learned_model_generations
    ]
    num_non_frozen = len(learned_models_paths)
    models = []
    for i in range(num_non_frozen):
        model = MiscUtils.SmallEncoder1d(in_dim,
                                         out_dim,
                                         num_hidden=5,
                                         non_lin="leaky_relu",
                                         use_bn=False)
        model.load_state_dict(torch.load(learned_models_paths[i]))
        model.eval()
        models.append(model)

    assert len(models) == len(
        learned_model_generations), "this shouldn't happen"

    populations = []  #those will actually store behavior descriptors
    for i in learned_model_generations:
        with open(root_dir + "/" + f"population_gen_{i}", "rb") as fl:
            #print(root_dir+"/"+f"population_gen_{i}")
            agents = pickle.load(fl)
            bds = [x._behavior_descr for x in agents]
            task_solvers = [x for x in agents if x._solved_task]
            if len(task_solvers):
                print(f"task solved at generation {i}")
            populations.append(
                np.concatenate(bds, 0)
            )  #so each matrix in populations will be of size pop_sz*bd_dim

    #this will store at each row i, the evolution of mean population novelty for population i through all generations
    Qmat = np.zeros(
        [len(learned_model_generations),
         len(learned_model_generations)])

    assert len(models) == len(populations), "something isn't right"
    thresh_u = 10
    with torch.no_grad():
        for i in range(len(learned_model_generations)):  #over populations
            if i % 10 == 0:
                print("i==", i)
            batch = torch.Tensor(
                populations[i]
            )  #no need to set a batch size, we consider the entier population to be the batch
            pred_f = frozen(batch)
            for j in range(i, len(learned_model_generations)):  #over models
                pred_j = models[j](batch)
                diff = (pred_f - pred_j)**2
                diff = diff.sum(1)
                diff = diff.sqrt()
                novelty = diff.mean().item()
                Qmat[i, j] = novelty
    return Qmat
