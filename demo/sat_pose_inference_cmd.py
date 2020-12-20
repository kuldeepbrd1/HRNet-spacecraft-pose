from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil
import json


from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import glob
import numpy as np


import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform


INPUT_DATA={
    'video':0,
    'image':1,
    'images':2
}

def delete_if_exists(target_dir_list):
    delete_all = True
    for path in target_dir_list:
        if os.path.exists(path) and os.path.isdir(path):
            if not delete_all:
              print(f'{path} exists. It will be removed to avoid data conflict. Do you want to continue? [y/n]')
              response = input()
              if response.lower() == 'y':
                  delete_all = True

            if delete_all:
                shutil.rmtree(path)
                print(f"{path} Removed")

def prepare_output_dirs(args, cfg, prefix='output/'):
    pose_dir = os.path.join(prefix,cfg.DATASET.DATASET, args.data_name, 'poses')
    box_dir = os.path.join(prefix,cfg.DATASET.DATASET, args.data_name, 'boxes')
    hm_dir = os.path.join(prefix,cfg.DATASET.DATASET, args.data_name,'heatmaps')

    delete_if_exists([pose_dir,box_dir,hm_dir])
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
    return pose_dir, box_dir, hm_dir

def load_data(args):
    data_path = args.data_path
    data_type = INPUT_DATA[args.data_type.lower()]
    img_idxs = None
    frames = None

    if data_type ==0:
        frames = []
        vidcap = cv2.VideoCapture(args.data_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps < args.inferenceFps:
            print('desired inference fps is '+str(args.inferenceFps)+' but video fps is '+str(fps))
            exit()
        every_nth_frame = round(fps/args.inferenceFps)

        success, image_bgr = vidcap.read()
        count = 0

        while success:
            if count % every_nth_frame != 0:
                success, image_bgr = vidcap.read()
                count += 1
                continue

            image = image_bgr[:, :, [2, 1, 0]]
            frames.append(image)

        frames = np.array(frames)


    elif data_type ==2:
        valid_extensions = ['jp*g','png']
        img_paths=[]
        for ext in valid_extensions:
            img_paths.extend(glob.glob(os.path.join(data_path,f'*.{ext}')))

        frames = []
        imgs_map = {}
        for idx,img_path in enumerate(img_paths):
            img_bgr = cv2.imread(img_path)
            img_rgb = img_bgr[:, :, [2, 1, 0]]

            img_name = (img_path.split('/')[-1]).split('.')[0]
            frames.append(img_rgb)
            imgs_map[img_name]= idx
        frames = np.array(frames)
        img_idxs = imgs_map

    else:
        frames = np.array([cv2.imread(data_path)])
        img_path = data_path
        img_name = (img_path.split('/')[-1]).split('.')[0]
        img_idxs = {img_name:0}

    return frames, img_idxs

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_x = box[0]
    bottom_left_y = box[1]
    box_width = box[2]
    box_height = box[3]

    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale




def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--data_name', type=str, default='output/')
    parser.add_argument('--data_type', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--bbox', type=str)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    print(args)
    update_config(cfg, args)
    pose_dir, box_dir, hm_dir = prepare_output_dirs(args= args, cfg = cfg, prefix= args.outputDir)
    csv_output_filename = args.outputDir+'pose-data.csv'
    csv_output_rows = []

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS).cuda()

    # Loading an video
    frames, idx_map = load_data(args)

    with open(args.bbox,'r') as jfile:
        bboxes_all_imgs = json.load(jfile)

    for idx,frame in enumerate(frames):
        img_name = list(idx_map)[idx]
        boxes = bboxes_all_imgs[img_name]
        boxes = [boxes] if np.ndim(boxes)==1 else boxes
        print(np.ndim(boxes))
        if args.writeBoxFrames:
            image_bgr_box = frame.copy()
            for box in boxes:
                cv2.rectangle(image_bgr_box, box[0], box[1], color=(0, 255, 0),
                              thickness=3)  # Draw Rectangle with the coordinates

            cv2.imwrite(os.path.join (box_dir,f"{img_name}_box_{idx}.jpg"))
        if not boxes:
            continue

        # pose estimation
        box = boxes[0]  # assume there is only 1 person
        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        #image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
        image_pose = frame.copy()
        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)

        new_csv_row = []
        for idx, mat in enumerate(pose_preds[0]):
            x_coord, y_coord = int(mat[0]), int(mat[1])
            cv2.circle(image_pose, (x_coord, y_coord), 2, (150, 0, 255), 2)
            new_csv_row.extend([x_coord, y_coord])
            print(f"{idx}, {[x_coord,y_coord]}")

        csv_output_rows.append(new_csv_row)
        cv2.imwrite(pose_dir+f'{img_name}_pose.jpg', image_pose)
        print(os.path.abspath(pose_dir))
    # write csv
    csv_headers = ['frame']
    for keypoint in range(cfg.MODEL.NUM_JOINTS):
        csv_headers.extend([f"{keypoint}_x", f"{keypoint}_y"])

    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)
    '''
    os.system("ffmpeg -y -r "
              + str(args.inferenceFps)
              + " -pattern_type glob -i '"
              + pose_dir
              + "/*.jpg' -c:v libx264 -vf fps="
              + str(args.inferenceFps)+" -pix_fmt yuv420p /output/movie.mp4")
    '''

if __name__ == '__main__':
    main()