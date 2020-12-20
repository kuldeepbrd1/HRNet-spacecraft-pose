# Deep High-Resolution Representation Learning for Spacecraft Pose Estimation
## Introduction
This is an unofficial pytorch implementation of [*Deep High-Resolution Representation Learning for Human Pose Estimation*](https://arxiv.org/abs/1902.09212). 
In this work, we are interested in the human pose estimation problem with a focus on learning reliable high-resolution representations. Most existing methods **recover high-resolution representations from low-resolution representations** produced by a high-to-low resolution network. Instead, our proposed network **maintains high-resolution representations** through the whole process.

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

6. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- hrnet_w32-36af842e.pth
            |   |-- hrnet_w48-8ef0771d.pth
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet101-5d3b4d8f.pth
            |   `-- resnet152-b121ed2d.pth
            |-- pose_coco
            |   |-- pose_hrnet_w32_256x192.pth
            |   |-- pose_hrnet_w32_384x288.pth
            |   |-- pose_hrnet_w48_256x192.pth
            |   |-- pose_hrnet_w48_384x288.pth
            |   |-- pose_resnet_101_256x192.pth
            |   |-- pose_resnet_101_384x288.pth
            |   |-- pose_resnet_152_256x192.pth
            |   |-- pose_resnet_152_384x288.pth
            |   |-- pose_resnet_50_256x192.pth
            |   `-- pose_resnet_50_384x288.pth
            `-- pose_mpii
                |-- pose_hrnet_w32_256x256.pth
                |-- pose_hrnet_w48_256x256.pth
                |-- pose_resnet_101_256x256.pth
                |-- pose_resnet_152_256x256.pth
                `-- pose_resnet_50_256x256.pth

   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing

#### Testing on MPII dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
```

#### Training on MPII dataset

```
python tools/train.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml
```

#### Testing on COCO val2017 dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    TEST.USE_GT_BBOX False
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
```

### Visualization

#### Visualizing predictions on COCO val

```
python visualization/plot_coco.py \
    --prediction output/coco/w48_384x288_adam_lr1e-3/results/keypoints_val2017_results_0.json \
    --save-path visualization/results

```


<img src="figures\visualization\coco\score_610_id_2685_000000002685.png" height="215"><img src="figures\visualization\coco\score_710_id_153229_000000153229.png" height="215"><img src="figures\visualization\coco\score_755_id_343561_000000343561.png" height="215">

<img src="figures\visualization\coco\score_755_id_559842_000000559842.png" height="209"><img src="figures\visualization\coco\score_770_id_6954_000000006954.png" height="209"><img src="figures\visualization\coco\score_919_id_53626_000000053626.png" height="209">

### Other applications
Many other dense prediction tasks, such as segmentation, face alignment and object detection, etc. have been benefited by HRNet. More information can be found at [High-Resolution Networks](https://github.com/HRNet).

### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {CoRR},
  volume    = {abs/1908.07919},
  year={2019}
}
```
