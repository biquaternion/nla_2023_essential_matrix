#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from csv import reader
from typing import Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from coords import diff_by_euler, translation
from essential import calc_true_essential, solve_essential

from config import Config


def load_points(camera_id: int = 0) -> np.ndarray:
    filename = f'data/points_cam{camera_id}.csv'
    with open(filename) as csvfile:
        points = reader(csvfile.readlines(), delimiter=',')
        points_arr = np.array(list(map(list, [map(float, line) for line in points])))
        return points_arr


def get_cam_position(config, cam_id) -> Tuple:
    cam_pose = config.cameras[cam_id]
    loc = cam_pose['pos']
    loc = np.array([loc['x'], loc['y'], loc['z']])
    ori = cam_pose['euler']
    ori = [ori['x'], ori['y'], ori['z']]
    return loc, ori


if __name__ == '__main__':
    config = yaml.load(open('data/cam_poses.yaml'), Loader=yaml.FullLoader)

    intrinsic = config.intrinsic
    focal = intrinsic['focal'] if 'focal' in intrinsic else 50
    width = intrinsic['width'] if 'width' in intrinsic else 36
    height = intrinsic['height'] if 'height' in intrinsic else 24

    # cam_mat = np.array([[focal, 0, width / 2],
    #                     [0, focal, height / 2],
    #                     [0, 0, 1]])
    focal *= 1920 / 36
    width = 1920
    height = 1080
    # focal *= 1000
    # width *= 1000
    # height *= 1000
    cam_mat = np.array([[focal, 0, width / 2],
                        [0, focal, height / 2],
                        [0, 0, 1]])

    cam_a = 0
    cam_b = 3

    points_a = load_points(cam_a)
    points_b = load_points(cam_b)

    points_a[:, 1] = 1080 - points_a[:, 1]
    points_b[:, 1] = 1080 - points_b[:, 1]
    points_a[:, 0] = 1920 - points_a[:, 0]
    points_b[:, 0] = 1920 - points_b[:, 0]



    loc_a, ori_a = get_cam_position(config, cam_a)
    loc_b, ori_b = get_cam_position(config, cam_b)

    E0 = calc_true_essential(loc_a, ori_a, loc_b, ori_b)
    R0 = diff_by_euler(ori_a, ori_b)
    t_mat = Rotation.from_euler(seq='xyz',
                                angles=ori_b,
                                degrees=True).inv()
    t0 = translation(loc_a, loc_b)
    t0 = t_mat.apply(t0)

    E, mask = cv2.findEssentialMat(points_a,
                                   points_b,
                                   cameraMatrix=cam_mat,
                                   method=cv2.RANSAC)

    E_1 = solve_essential(points_a, points_b, camera_matrix=cam_mat)

    T = cv2.recoverPose(E, points_a, points_b, cameraMatrix=cam_mat)
    T_1 = cv2.recoverPose(E_1,
                          points_a,
                          points_b,
                          cameraMatrix=cam_mat)

    R_calc = T[1]
    t_calc = T[2]

    R_1 = T_1[1]
    t_1 = T_1[2]

    print("GT")
    print("R")
    print(R0)
    print("t")
    norm0 = np.linalg.norm(t0)
    print((t0 / norm0 if norm0 != 0 else np.array([0, 0, 0])).T)

    print("OpenCV")
    print("R")
    print(R_calc)
    print("t_calc")
    print(t_calc.T / np.linalg.norm(t_calc))


    print('solution')
    print('R')
    print(R_1)
    print('t')
    print(t_1.T / np.linalg.norm(t_1))
