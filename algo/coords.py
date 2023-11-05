#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Iterable, List

import numpy as np
from scipy.spatial.transform import Rotation

deg2rad = np.pi / 180
rad2deg = 180 / np.pi


def euler_to_rotation(euler: List,
                      seq: str = 'xyz') -> np.ndarray:
    '''
    calculates rotation matrix from euler angles
    with given rotation sequence
    :param euler: 3 angles
    :param seq:   sequence of rotations (default - xyz)
    :return:      rotation matrix
    '''
    rot = Rotation.from_euler(seq, angles=euler)
    return rot.as_matrix()


def translation(loc1: np.ndarray,
                loc2: np.ndarray) -> np.ndarray:
    '''
    calculates shift between 2 locations
    :param loc1:
    :param loc2:
    :return:
    '''
    return loc2 - loc1


def diff_by_euler(euler1: List,
                  euler2: List,
                  seq: str = 'xyz'):# -> np.ndarray:
    '''
    calculate rotation matrix between 2 orientations, given in euler
    :param euler1:
    :param euler2:
    :param seq:
    :return:
    '''
    r1 = Rotation.from_euler(seq, angles=euler1)
    r2 = Rotation.from_euler(seq, angles=euler2)
    # return r1.as_matrix() @ r2.inv().as_matrix()
    # return r1.as_quat() / r2.as_quat()
    return Rotation.concatenate([r1, r2.inv()])


if __name__ == '__main__':
    euler = [50, 15, 25]
    rot_mat = euler_to_rotation(euler)
    print(rot_mat)
    print(np.linalg.det(rot_mat))
    euler = [45, 20, 25]
    rot_mat = euler_to_rotation(euler)
    print(rot_mat)
    print(np.linalg.det(rot_mat))

    rot_mat = diff_by_euler([0, 0, 15 * deg2rad], [0, 0, 375 * deg2rad])
    print(rot_mat)
    print(np.linalg.det(rot_mat))
