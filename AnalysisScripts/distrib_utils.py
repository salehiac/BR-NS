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
import scipy.special
import pdb


def KLdiv(P, Q):
    """
    P, Q 2d distributions
    """
    eps = 1e-9
    C = np.log2(P / (Q + eps))
    return (P * C).sum()


def jensen_shannon(P, Q):
    M = 0.5 * (P + Q)

    return 0.5 * KLdiv(P, M) + 0.5 * KLdiv(Q, M)


def uniform_like(A):

    u = np.ones_like(A)
    return scipy.special.softmax(u)
