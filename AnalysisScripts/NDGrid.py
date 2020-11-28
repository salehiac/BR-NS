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

import numpy as np
import matplotlib.pyplot as plt


class NDGridUniformNoPrealloc:
    def __init__(self, Gs, dims, lower_bounds, higher_bounds):
        """
        Gs is a list and is the number of grids per dimension
        lower_bounds and higher_bounds should both be lists of length dims, giving the bounds for each dimension
        """

        self.Gs = Gs
        self.dims = dims
        self.num_cells = 1
        for x in Gs:
            self.num_cells *= x
        self.lb = lower_bounds
        self.hb = higher_bounds

        self.ranges = []
        for i in range(dims):
            self.ranges.append(
                list(np.linspace(self.lb[i], self.hb[i], self.Gs[i] + 1)))

        self.visited_cells = {}

    def compute_current_coverage(self):

        return len(self.visited_cells) / self.num_cells

    def visit_cell(self, pt):
        """
        pt should be of size 1*self.dims
        """

        cell = []
        for i in range(self.dims):
            z = pt[i]
            if z > self.hb[i] or z < self.lb[i]:
                raise Exception("input point is outside of grid boundaries")

            for ii in range(1, self.Gs[i] + 1):
                if self.ranges[i][ii] >= z:
                    cell.append(ii - 1)
                    break

        assert len(cell) == self.dims, "this should'nt happen"

        cell = tuple(cell)
        print("cell==\n", cell)

        if cell not in self.visited_cells:
            self.visited_cells[cell] = 1
        else:
            self.visited_cells[cell] += 1
