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

import gym, gym_fastsim
import time
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import os

import scipy.stats as stats
import scipy.special
import math
from collections import namedtuple

sys.path.append("../NS/")
sys.path.append("..")
from NS import MiscUtils

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import distrib_utils


def get_params_sum(model, trainable_only=False):

    with torch.no_grad():
        if trainable_only:
            model_parameters = filter(lambda p: p.requires_grad,
                                      model.parameters())
        else:
            model_parameters = model.parameters()

        u = sum([x.sum().item() for x in model_parameters])
        return u


def randomize_weights(net):
    u = [x for x in net.mds]
    with torch.no_grad():
        for m in u:
            m.weight.fill_(0.0)
            m.bias.fill_(0.0)


def see_evolution_of_learned_novelty_distribution_hardmaze(
        root_dir,
        bn_was_used=True,
        non_lin_type="leaky_relu",
        in_dim=2,
        out_dim=2,
        behavior_type="generic_2d",
        batch_producer=None):
    """
    root_dir      directory of an NS experiment (NS_log_{pid}), expected to contain frozen_net.model and learnt_{i}.model for i in range(200)
    """

    if bn_was_used:
        raise Exception(
            "bn doesn't make sense 1) for frozen, it will modify it's behavior 2) for the other one, not sure but seems like it will hinder convergence"
        )

    frozen_net_path = root_dir + "/frozen_net.model"
    #learned_model_generations=list(range(0,45,10))
    learned_model_generations = list(range(0, 20, 1))
    #learned_model_generations=list(range(0,20))
    learned_models_paths = [
        root_dir + f"/learnt_{i}.model" for i in learned_model_generations
    ]
    #print(learned_models_paths)

    display = True

    env = gym.make('FastsimSimpleNavigation-v0')
    _ = env.reset()

    width = int(env.map.get_real_w())
    height = int(env.map.get_real_h())

    #sys.argv[1]
    frozen = MiscUtils.SmallEncoder1d(in_dim,
                                      out_dim,
                                      num_hidden=3,
                                      non_lin=non_lin_type,
                                      use_bn=bn_was_used)
    frozen.load_state_dict(torch.load(frozen_net_path))
    frozen.eval()

    num_non_frozen = len(learned_models_paths)
    models = []
    results = []
    for i in range(num_non_frozen):
        model = MiscUtils.SmallEncoder1d(in_dim,
                                         out_dim,
                                         num_hidden=5,
                                         non_lin=non_lin_type,
                                         use_bn=bn_was_used)
        model.load_state_dict(torch.load(learned_models_paths[i]))
        model.eval()
        models.append(model)
        results.append(np.zeros([height, width]))

    if behavior_type == "generic_2d":
        with torch.no_grad():
            for i in range(height):
                if i % 10 == 0:
                    print("i==", i)
                batch = torch.cat([
                    torch.ones(width, 1) * i,
                    torch.arange(width).float().unsqueeze(1)
                ], 1)
                z1 = frozen(batch)
                for j in range(num_non_frozen):
                    z2 = models[j](batch)
                    diff = (z2 - z1)**2
                    diff = diff.sum(1)
                    if torch.isnan(diff).any():
                        print("j==", j)
                        pdb.set_trace()

                    results[j][i, :] = np.sqrt(diff.cpu().numpy())

    elif behavior_type == "from_encoder":
        if batch_producer is None:
            raise Exception(
                "you must provide a batch_producer if using from_encoder")

    for i in range(len(results)):
        results[i] = np.flip(results[i], 0)

    results_np = np.concatenate(results, 1)
    #plt.imshow(results_np)
    #plt.show()

    uniform_distrib = distrib_utils.uniform_like(results[0])
    jensen_shanon_dists = []
    for i in range(num_non_frozen):
        jensen_shanon_dists.append(
            distrib_utils.jensen_shannon(results[i], uniform_distrib))

    env.close()

    return jensen_shanon_dists


def evolution_of_age_and_parent_child_distances(root_dir):

    ages = []
    dists = []
    for gen in range(0, 100, 5):
        if gen % 100 == 0:
            print("gen==", gen)
        fn = root_dir + f"/population_gen_{gen}"
        with open(fn, "rb") as f:
            pop = pickle.load(f)

        ages.append(np.mean([gen - indv._created_at_gen for indv in pop]))
        dists.append(np.mean([indv._bd_dist_to_parent_bd for indv in pop]))

    return ages, dists
