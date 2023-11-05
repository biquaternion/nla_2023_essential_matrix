#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from coords import translation, diff_by_euler
from adapters import svd


def calc_true_essential(loc1, rot1, loc2, rot2):
    t = translation(loc1=loc1, loc2=loc2)
    t_hat = np.array([[0, -t[2], t[1]],
                      [t[2], 0, -t[0]],
                      [-t[1], t[0], 0]])
    rot_mat = diff_by_euler(rot1, rot2)
    return np.matmul(t_hat, rot_mat)


def solve_essential(points1, points2, camera_matrix):
    assert points1.shape == points2.shape
    num_pts = points1.shape[0]
    rays_1 = np.dot(np.linalg.inv(camera_matrix),
                    np.hstack([points1, np.ones((num_pts, 1))]).T).T
    rays_2 = np.dot(np.linalg.inv(camera_matrix),
                    np.hstack([points2, np.ones((num_pts, 1))]).T).T
    khi = np.hstack([rays_1 * rays_2[:, 0].reshape((8, 1)),
                     rays_1 * rays_2[:, 1].reshape((8, 1)),
                     rays_1 * rays_2[:, 2].reshape((8, 1))])
    _, _, vT_khi = svd(khi)
    E_est = np.reshape(vT_khi.T[:, 8], (3, 3))

    u_E, s_E, vh_E = np.linalg.svd(E_est, full_matrices=True)
    s_E[-1] = 0
    sigma = (s_E[0] + s_E[1]) / 2
    E = u_E @ np.diag([sigma, sigma, 0]) @ vh_E
    return E


if __name__ == '__main__':
    pass
