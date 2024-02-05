"""Evaluate pre-trained LipForensics model on various face forgery datasets"""

import librosa
import torch
import os
from models.spatiotemporal_net import get_model
from utils import get_save_folder, get_logger, load_model
import argparse
import torch.nn.functional as F
from preprocessing.utils import cut_patch, warp_img
import cv2
from torchvision import transforms
from PIL import Image as pil_image
import dlib
import numpy as np
import random

STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("preprocessing/shape_predictor_68_face_landmarks.dat")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFake detector evaluation")
    parser.add_argument(
        "--dataset",  # 数据集类型
        help="Dataset to evaluate on",
        type=str,
        default="DFDC",
    )
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")  # 灰度图
    parser.add_argument("--rgb", dest="grayscale", action="store_false")  # 彩图
    parser.set_defaults(grayscale=True)
    parser.add_argument('--model-path', type=str, default="./train_logs/Both/2022-05-17T16-20-50/ckpt.best.pth.tar",
                        help='Pretrained model pathname')
    parser.add_argument("--frames_per_clip", default=29, type=int)  # 每个clip的帧数
    parser.add_argument("--batch_size", default=4, type=int)  # batch_size
    parser.add_argument("--device", help="Device to put tensors on", type=str, default="cuda:0")
    parser.add_argument("--num_workers", default=4, type=int)  # 线程数
    parser.add_argument(
        "--weights_forgery_path",
        help="Path to pretrained weights for forgery detection",
        type=str
    )
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    parser.add_argument('--logging-dir', type=str, default='./train_logs',
                        help='path to the directory in which to save the log file')
    parser.add_argument('--training-mode', default='Both', help='visual, audio, Both')
    parser.add_argument('--video_path', type=str, help='test video path', default='./hierggamuo.mp4')
    parser.add_argument('--mean_face', type=str, default='./preprocessing/20words_mean_face.npy')
    parser.add_argument("--crop-width", default=96, type=int, help="Width of mouth ROIs")
    parser.add_argument("--crop-height", default=96, type=int, help="Height of mouth ROIs")
    parser.add_argument("--start-idx", default=48, type=int, help="Start of landmark index for mouth")
    parser.add_argument("--stop-idx", default=68, type=int, help="End of landmark index for mouth")

    args = parser.parse_args()
    return args


def validate_video_level(model, data, args):
    """ "
    Evaluate model using video-level AUC score.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance
    loader : torch.utils.data.DataLoader
        Loader for forgery data
    args
        Options for evaluation
    """
    model.eval()

    images = images.to(args.device)
    audios = audios.to(args.device)
    # Forward
    logits = model(images, lengths=[args.frames_per_clip] * images.shape[0])
    _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)


def extract_frames(data_path, args):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    mean_face_landmarks = np.load(args.mean_face)
    reader = cv2.VideoCapture(data_path)
    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector(image)
        preds = np.matrix([[point.x, point.y] for point in predictor(image, faces[0]).parts()])
        landmark = preds.A
        trans_frame, trans = warp_img(
            landmark[STABLE_POINTS, :], mean_face_landmarks[STABLE_POINTS, :], image, STD_SIZE
        )
        trans_landmarks = trans(landmark)
        cropped_frame = cut_patch(
            trans_frame,
            trans_landmarks[args.start_idx: args.stop_idx],
            args.crop_height // 2,
            args.crop_width // 2,
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((88, 88)),
            transforms.Normalize((0.421,), (0.165,))
        ])
        frame = transform(pil_image.fromarray(cropped_frame))

        frame_num += 1
        if frame_num >= 150:
            break
    reader.release()


def main():
    args = parse_args()
    save_path = get_save_folder(args)
    logger = get_logger(args, save_path)
    model = get_model(weights_forgery_path=args.weights_forgery_path, device=args.device, mode=args.training_mode)
    if args.model_path:
        assert args.model_path.endswith('.tar') and os.path.isfile(args.model_path), \
            "'.tar' model path does not exist. Path input: {}".format(args.model_path)
        # resume from checkpoint
        model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
        logger.info('Model has been successfully loaded from {}'.format(args.model_path))

    # print(model)
    audio = librosa.load(args.video_path, sr=16000)[0][-101000:]
    extract_frames(args.video_path, args)
    data = 0


if __name__ == "__main__":
    main()
