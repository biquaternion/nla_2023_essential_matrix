#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from algo.coords import translation, diff_by_euler


def calc_true_essential(loc1, rot1, loc2, rot2):
    t = translation(loc1=loc1, loc2=loc2)
    t_hat = np.array([[0, -t[2], t[1]],
                      [t[2], 0, -t[0]],
                      [-t[1], t[0], 0]])
    rot_mat = diff_by_euler(rot1, rot2)
    return np.matmul(t_hat, rot_mat)


def solve_essential():
    pass


if __name__ == '__main__':
    pass
