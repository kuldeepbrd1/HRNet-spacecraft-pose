# Extra utils for processing images/heatmaps for satellite poe estimation
# Kuldeep Barad (TU Delft)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
import os

from core.inference import get_max_preds

def save_all_heatmaps(batch_image, batch_heatmaps, output_dir, meta, normalize=True,):
    '''
    Need test batch size =1
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    heatmaps_dir = os.path.join(output_dir,'heatmaps')
    if not os.path.exists(heatmaps_dir):
        os.makedirs(heatmaps_dir, exist_ok=True)
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)


    heatmap_image = np.zeros((heatmap_height,heatmap_width,3))
    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        img_name = ((meta['image'][i]).split("/")[-1]).split(".")[0]
        print(img_name)
        for j in range(num_joints):

            img_dir = os.path.join(heatmaps_dir,img_name)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            heatmap_filepath = os.path.join(img_dir,(img_name + '_heatmap_'+ str(j)+".jpg"))

            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            cv2.imwrite(heatmap_filepath, colored_heatmap)
            #grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #    masked_image
        #grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

        #cv2.imwrite((img_dir+'/'+'grid.jpg'), grid_image)



def save_all_val_heatmaps(config, input, meta, target, joints_pred, output, output_dir):
    if not config.DEBUG.DEBUG:
        return
    set_dir = os.path.join(output_dir,config.DATASET.TEST_SET)
    if config.DEBUG.SAVE_HEATMAPS_TEST_ALL:
        if not os.path.exists(set_dir):
            os.mkdir(set_dir)
        save_all_heatmaps(input, output, set_dir, meta)


