#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from csv import reader
from typing import Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from coords import diff_by_euler, translation
from essential import calc_true_essential

def load_points(filename: str) -> np.ndarray:
    with open(filename) as csvfile:
        points = reader(csvfile.readlines(), delimiter=',')
        points_arr = np.array(list(map(list, [map(float, line) for line in points])))
        return points_arr


def get_cam_position(cam_poses, cam_id) -> Tuple:
    cam_a_pose = cam_poses[f'cam{cam_id}']
    loc = cam_a_pose['pos']
    loc = np.array([loc['x'], loc['y'], loc['z']])
    ori = cam_a_pose['euler']
    ori = [ori['x'], ori['y'], ori['z']]
    return loc, ori


if __name__ == '__main__':

    cam_poses = yaml.safe_load(open('data/cam_poses.yaml'))
    intrinsic = cam_poses['intrinsic']
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

    # points = reader(open('data/points.csv').readlines(),
    #                 delimiter=',')
    # header = next(points)
    # cols = {i: name for i, name in enumerate(header)}
    # points_arr = np.array(list(map(list, [map(float, line) for line in points])))
    #
    # points3d = points_arr[:, :3]
    points1 = load_points('data/points_cam0.csv')
    points2 = load_points('data/points_cam2.csv')


    # points1[:, 1] = 1080 - points1[:, 1]
    # points2[:, 1] = 1080 - points2[:, 1]

    loc_a, ori_a = get_cam_position(cam_poses, 0)
    loc_b, ori_b = get_cam_position(cam_poses, 2)

    # E0 = calc_true_essential(loc_a, ori_a, loc_b, ori_b)
    R0 = diff_by_euler(ori_a, ori_b).as_matrix()
    t_mat = Rotation.from_euler(seq='xyz', angles=ori_a).inv().as_matrix()
    t0 = translation(loc_a, loc_b)

    E, mask = cv2.findEssentialMat(points1,
                                   points2,
                                   cameraMatrix=cam_mat,
                                   method=cv2.RANSAC)

    T = cv2.recoverPose(E, points1, points2, cameraMatrix=cam_mat)

    R_calc = T[1]
    t_calc = t_mat @ np.reshape(T[2], t0.shape)

    print("R_gt")
    print(R0)
    print("R_cals")
    print(R_calc)

    print("t_gt")
    print(t0 / np.linalg.norm(t0))
    print("t_calc")
    print(t_calc / np.linalg.norm(t_calc))
