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

from abc import ABC, abstractmethod
from sklearn.neighbors import KDTree
import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb
import random

import MiscUtils


class NoveltyEstimator(ABC):
    """
    Interface for estimating Novelty
    """
    @abstractmethod
    def __call__(self):
        """
        estimate novelty of entire current population w.r.t istelf+archive
        """
        pass

    @abstractmethod
    def update(self, pop, archive=None):
        pass


class ArchiveBasedNoveltyEstimator(NoveltyEstimator):
    """
    For now parallelising this is just premature optimisation
    """
    def __init__(self, k):
        self.k = k
        self.archive = None
        self.pop = None
        self.log_dir = "/tmp/"

    def update(self, pop, archive):
        self.archive = archive
        self.pop = pop

        self.pop_bds = [x._behavior_descr for x in self.pop]
        self.pop_bds = np.concatenate(self.pop_bds, 0)
        self.archive_bds = [x._behavior_descr for x in self.archive]

        if len(self.archive_bds):
            self.archive_bds = np.concatenate(self.archive_bds, 0)

        self.kdt_bds = np.concatenate([self.archive_bds, self.pop_bds],
                                      0) if len(
                                          self.archive_bds) else self.pop_bds
        self.kdt = KDTree(self.kdt_bds, leaf_size=20, metric='euclidean')

    def __call__(self):
        """
        estimate novelty of entire current population w.r.t istelf+archive

        returns novelties as unsorted list
        """
        dists, ids = self.kdt.query(self.pop_bds, self.k, return_distance=True)
        dists = dists[:, 1:]
        ids = ids[:, 1:]

        novs = dists.mean(1)
        return novs.tolist()


class LearnedNovelty1d(NoveltyEstimator):
    def __init__(self,
                 in_dim,
                 emb_dim,
                 pb_limits=None,
                 batch_sz=128,
                 log_dir="/tmp/"):

        self.frozen = MiscUtils.SmallEncoder1d(in_dim,
                                               emb_dim,
                                               num_hidden=3,
                                               non_lin="leaky_relu",
                                               use_bn=False)

        self.frozen.eval()

        self.learnt = MiscUtils.SmallEncoder1d(in_dim,
                                               emb_dim,
                                               num_hidden=5,
                                               non_lin="leaky_relu",
                                               use_bn=False)

        self.optimizer = torch.optim.Adam(self.learnt.parameters(), lr=1e-2)
        self.archive = None
        self.pop = None
        self.batch_sz = batch_sz

        self.epoch = 0

        self.log_dir = log_dir

        if pb_limits is not None:
            MiscUtils.make_networks_divergent(self.frozen,
                                              self.learnt,
                                              pb_limits,
                                              iters=50)

    def update(self, pop, archive=None):

        self.pop = pop

        self.pop_bds = [x._behavior_descr for x in self.pop]
        self.pop_bds = np.concatenate(self.pop_bds, 0)

    def __call__(self):

        pop_novs = []
        for i in range(0, self.pop_bds.shape[0], self.batch_sz):
            batch = torch.Tensor(self.pop_bds[i:i + self.batch_sz])
            with torch.no_grad():
                e_frozen = self.frozen(batch)
                self.learnt.eval()
                e_pred = self.learnt(batch)
                diff = (e_pred - e_frozen)**2
                diff = diff.sum(1)
                pop_novs += diff.cpu().detach().tolist()

        assert len(pop_novs) == self.pop_bds.shape[0], "that shouldn't happen"

        return pop_novs

    def train(self, pop):

        if self.epoch == 0:
            torch.save(self.frozen.state_dict(),
                       self.log_dir + "/frozen_net.model")

        torch.save(self.learnt.state_dict(),
                   self.log_dir + f"/learnt_{self.epoch}.model")

        pop_bds = [x._behavior_descr for x in pop]
        pop_bds = np.concatenate(pop_bds, 0)
        for _ in range(3):
            for i in range(0, pop_bds.shape[0], self.batch_sz):
                batch = torch.Tensor(pop_bds[i:i + self.batch_sz])
                with torch.no_grad():
                    e_frozen = self.frozen(batch)

                self.learnt.train()
                self.optimizer.zero_grad()
                e_l = self.learnt(batch)
                ll = (e_l - e_frozen)**2
                ll = ll.sum(1)
                weights = torch.Tensor([1.0 for x in range(batch.shape[0])])
                loss = ll * weights
                loss = loss.mean().clone()
                if torch.isnan(loss).any():
                    raise Exception(
                        "loss is Nan. Maybe tray reducing the learning rate")
                loss.backward()
                self.optimizer.step()

        self.epoch += 1
