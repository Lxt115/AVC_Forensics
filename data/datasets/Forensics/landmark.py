#!/usr/bin/env python
# coding=utf-8

"""
@author: Richard Huang
@license: WHU
@contact: 2539444133@qq.com
@file: landmark.py
@date: 22/04/27 14:19
@desc: 
"""
import face_alignment
from skimage import io
import os
from os.path import join
import argparse
import subprocess
import dlib
from tqdm import tqdm
import numpy as np

DATASET_PATHS = {
    'original': 'RealFF',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def extract_frames(data_path, output_path):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    for name in tqdm(os.listdir(data_path)):
        ext = os.path.splitext(name)
        land_name = ext[0] + '.npy'
        image = join(data_path, name)
        input_img = io.imread(image)
        faces = detector(input_img)
        if len(faces) == 0:
            face_detector = 'sfd'
            face_detector_kwargs = {
                "filter_threshold": 0.8
            }

            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True,
                                              face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)
            if fa.get_landmarks(input_img) is None:
                continue
            preds = fa.get_landmarks(input_img)[-1]
            np.save(os.path.join(output_path, land_name), preds)
        else:
            preds = np.matrix([[p.x, p.y] for p in predictor(input_img, faces[0]).parts()])
            np.save(os.path.join(output_path, land_name), preds)


def extract_method_videos(data_path, dataset, compression):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    image_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    landmark_path = join(data_path, DATASET_PATHS[dataset], compression, 'landmarks')
    i = 0
    for video in os.listdir(image_path):
        if i >= 640:
            print("number {}".format(i))
            landmark_folder = video.split('.')[0]
            extract_frames(join(image_path, video),
                           join(landmark_path, landmark_folder))
        i += 1



if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str, default='./')
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='original')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c23')
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))