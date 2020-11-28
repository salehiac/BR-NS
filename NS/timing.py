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
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import argparse

from sklearn.neighbors import KDTree
import MiscUtils


def compare_execution_time(min_d_e, max_d_e, pop_sz=50, offspring_sz=100, archive_size=8000,num_tests=50):

    global_time_networks = []
    global_time_archive = []

    bd_size_range = [2**x for x in range(min_d_e,max_d_e+1,1)]
    print(bd_size_range)
    for descr_dim in bd_size_range:

        num_gens = num_tests
        emb_dim = descr_dim * 2
        pop_bds = torch.rand(offspring_sz + pop_sz, descr_dim)

        frozen = MiscUtils.SmallEncoder1d(descr_dim,
                                          emb_dim,
                                          num_hidden=3,
                                          non_lin="leaky_relu",
                                          use_bn=False)
        frozen.eval()

        learnt = MiscUtils.SmallEncoder1d(descr_dim,
                                          emb_dim,
                                          num_hidden=5,
                                          non_lin="leaky_relu",
                                          use_bn=False)

        time_hist = []
        frozen.eval()
        optimizer = torch.optim.Adam(
            [x for x in learnt.parameters() if x.requires_grad], lr=1e-2)
        batch_sz = 256

        for i in range(num_gens):
            t1_n = time.time()

            learnt.eval()
            for batch_i in range(0, pop_bds.shape[0], batch_sz):
                batch = pop_bds[batch_i:batch_i + batch_sz]

                with torch.no_grad():
                    e_frozen = frozen(batch)
                    e_pred = learnt(batch)
                    nov = (e_pred - e_frozen).norm(dim=1)

            learnt.train()
            #this is how training is done, note that we can further reduce runtime by removing the extra frozen forward passes that we've made before when computing novelty
            for _ in range(5):#In the experiments presented in the paper, usually this was set to 3 so BR-NS in the paper should be slightly faster
                for batch_i in range(0, pop_bds.shape[0], batch_sz):
                    batch = pop_bds[batch_i:batch_i + batch_sz]
                    with torch.no_grad():
                        e_frozen = frozen(batch)

                    optimizer.zero_grad()
                    e_l = learnt(batch)
                    loss = (e_l - e_frozen).norm()**2
                    loss /= batch_sz
                    loss.backward()
                    optimizer.step()

            t2_n = time.time()
            time_hist.append(t2_n - t1_n)

        mean_t_nets = np.array(time_hist).mean()
        print(descr_dim, mean_t_nets)
        global_time_networks.append(mean_t_nets)

        if 1:
            knn_k = 15
            kdt_bds = np.random.rand(archive_size, descr_dim)

            times = []
            for i in range(num_gens):
                t1_a = time.time()
                #note that the kdtree has to be created everytime as after adding elements, you can't just reuse the same kdtree  (some cells might have become much more dense)
                kdt = KDTree(kdt_bds, leaf_size=20, metric='euclidean')
                dists, ids = kdt.query(pop_bds, knn_k, return_distance=True)
                t2_a = time.time()
                times.append(t2_a - t1_a)

            mean_t_arch = np.array(times).mean()
            global_time_archive.append(mean_t_arch)
            print(descr_dim, mean_t_arch)

    if 1:
        gt_arc_ms = [x * 1000 for x in global_time_archive]
        gt_net_ms = [x * 1000 for x in global_time_networks]
        plt.plot(bd_size_range,
                 gt_arc_ms,
                 "r",
                 label=f"Archive-based NS (size=={archive_size})",
                 linewidth=5)
        plt.plot(bd_size_range, gt_net_ms, "b", label="BR-NS", linewidth=5)
        plt.grid("on")
        plt.legend(fontsize=28)
        plt.xlabel("behavior descriptor dimensionality", fontsize=28)
        plt.ylabel("time (ms)", fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.xlim(0, 2**max_d_e)

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Timing experiments.')
    parser.add_argument('--min_d_e',
                        type=int,
                        help="the minimum behavior descriptor dimensionality that will be considerd will be 2**min_d_e")
    parser.add_argument('--max_d_e',
                        type=int,
                        help="the maximum behavior descriptor dimensionality that will be considerd will be 2**max_d_e")
    parser.add_argument('--pop_sz',
                        type=int,
                        help="population size",
                        default=50)
    parser.add_argument('--off_sz',
                        type=int,
                        help="offspring size",
                        default=100)
    parser.add_argument('--archive_size',
                        type=int,
                        help="archive size",
                        default=8000)

    args = parser.parse_args()

    assert args.max_d_e is not None and args.min_d_e is not None

    compare_execution_time(args.min_d_e, args.max_d_e, args.pop_sz, args.off_sz, args.archive_size)

