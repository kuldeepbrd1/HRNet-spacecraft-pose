# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:39:15 2020

@author: apoca
"""

import json
import numpy as np
import os

def get_id_map(images_dicts):
    ids = []
    id_map = {}
    name_map={}
    for idx,image in enumerate(images_dicts):
        img_id = image['id']
        ids.append(img_id)
        id_map[str(img_id)] = idx
        name_map[str(img_id)] = image['file_name']
    return ids,id_map, name_map

def get_kps_coco(coco_kps_list, num_kps):
    kps_dict = {}
    kpts_score = np.zeros((num_kps,3))
    for idx in range(0,num_kps):
        x_kp = coco_kps_list[3*idx]
        y_kp = coco_kps_list[3*idx+1]
        score = coco_kps_list[3*idx+2]
        kps_dict[str(idx)] = [x_kp,y_kp,score]  #for gt score==visibility
        kpts_score[idx,:] =  [x_kp,y_kp,score]
    return kps_dict

def get_invalid_info(invalid_dicts):
    errs = []
    for img in invalid_dicts:
       for feat in img['invalid_scores']:
           error =np.array(feat['error'])
           if np.any(error>1):
               print(feat)
               errs.append(feat)

    print(f"Length:{len(errs)}")

def calc_error(gt, pred,scale,debug=False):
    # Return  [x_gt,x_pred,y_gt,y_pred,dx,dy,dr,score]
    n = len(list(pred))
    data_w_err = {}
    total_dr = 0
    total_norm_dr = 0

    invalid_scores= []

    for i in range(0,n):
        x_gt = gt[str(i)][0]
        y_gt = gt[str(i)][1]
        visibility = gt[str(i)][2]
        x_pred = pred[str(i)][0]
        y_pred = pred[str(i)][1]
        score = pred[str(i)][2]

        dx = abs(x_pred-x_gt)
        dy = abs(y_pred - y_gt)
        dr = float(np.sqrt(dx**2+dy**2))

        dx_normalized = dx/scale[0]
        dy_normalized = dy/scale[1]
        dr_normalized = float(np.sqrt(dx_normalized**2 + dy_normalized**2))

        total_dr += dr
        total_norm_dr += dr_normalized

        if score>1:
            invalid_scores.append(
                {
                    'kpt':i,
                    'score': score,
                    'gt': [x_gt,y_gt],
                    'pred': [x_pred,y_pred],
                    'error': [dx,dy]
                }
            )

        if visibility ==0:
            data_w_err[str(i)]= list(np.zeros((11,)))
        else:
            data_w_err[str(i)] = [x_gt,x_pred,y_gt,y_pred,score,dx,dy,dr,dx_normalized, dy_normalized, dr_normalized]

    avg_err = total_dr/n
    avg_norm_err=  total_norm_dr/n

    if debug:
        return invalid_scores, None, None
    else:
        return data_w_err, avg_err, avg_norm_err

#image set and config file to make folderss for each result
HRNet_dir = os.path.abspath(os.path.join(os.getcwd(),'..','..'))
gt_dir = os.path.abspath(os.path.join(HRNet_dir, '..','data','Envisat_Set1','val.json'))
pred_dir = os.path.join(HRNet_dir,'Results','Envisat_1','keypoints_val_results_0.json')

size_input = (256,256)
size_default = (512,512)

# change this accounting for bbox
#scale = ((size_default[0]/size_input[0]),(size_default[1]/size_input[1]))

with open(gt_dir,'r') as gtf:
    gt_data = json.load(gtf)

with open(pred_dir,'r') as predf:
    pred_data = json.load(predf)

gt_images_data = gt_data['images']
gt_ann_data = gt_data['annotations']
pred_ann_data = pred_data #Output is a list with size 1

all_ids, id_map, filename_map = get_id_map(gt_images_data)
num_img = len(all_ids)
num_kps = gt_ann_data[0]['num_keypoints']

error_lists = []
norm_avg_errs = []
avg_errs = []
info_dict = {}

invalid_dicts = []

for img_id in all_ids:
    gt_kps = get_kps_coco(gt_ann_data[id_map[str(img_id)]]['keypoints'],gt_ann_data[id_map[str(img_id)]]['num_keypoints'])
    pred_kps = get_kps_coco(pred_ann_data[id_map[str(img_id)]]['keypoints'], int(len(pred_ann_data[id_map[str(img_id)]]['keypoints'])/3))

    ## To do: get scale for each bbox in gt and use that
    scale_bbox = pred_ann_data[ id_map[str(img_id)]]['scale']
    data_err,avg_err, avg_norm_err = calc_error(gt_kps, pred_kps,scale_bbox)
    invalid_scores,_,_ = calc_error(gt_kps, pred_kps,scale_bbox,debug=True)
    if len(invalid_scores)>0:
        invalid_dicts.append(
            {
                'id' : img_id,
                'invalid_scores' : invalid_scores
            }
        )

    avg_errs.append(avg_err)
    norm_avg_errs.append(avg_norm_err)
    error_lists.append(data_err)
    info_dict[str(img_id)]= {'filename': filename_map[str(img_id)], 'avg_error':avg_err,'avg_normalized_error':avg_norm_err, 'error_data': data_err, 'score' : pred_ann_data[id_map[str(img_id)]]['score']}

avg_val_error = np.average(np.array(avg_errs))
norm_val_error = np.average(np.array(norm_avg_errs))
get_invalid_info(invalid_dicts)
#with open(os.path.join(os.getcwd(),'error_analysis.json'),'w') as fj:
    #json.dump(info_dict,fj,indent=4)


print(f" average error = {avg_val_error} \n normalized error = {norm_val_error}")