# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:51:39 2020

@author: Kuldeep Barad
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:51:39 2020
@author: Kuldeep Barad
"""

import json
import os
import glob
import string
from pycocotools.coco import COCO


class COCO_SatPose:
    ''' -----------------Format----------------------------
        {
         "info": info,
         "images": [image],
         "annotations": [annotation],
         "licenses": [license],
         }
    -------------------------------------------------------'''

    def __init__(self, **kwargs):
        self.dataset = kwargs["dataset"]
        self.dataset_objects = {"SPEED": "Tango",
                                "SPEED+": "Tango", "ENVISAT-1": "Envisat"}
        self.object = self.dataset_objects[self.dataset]
        kwargs["object"] = self.object

        self.keys = ["info", "licenses", "images",
                     "annotations", "categories", "segment_info"]

        self.info = info().getDatasetInfo(self.dataset)

        self.categories = [Category(**kwargs).get_dict()]

        self.license = self.get_license()

        kwargs["category"] = self.getCategoryinfo()
        kwargs["license_id"] = self.license

        self.images, self.annotations = Images(
            **kwargs).get_images_annotations()
        self.num_examples = len(self.images)

        self.json_dict = self.compose()

    def compose(self):
        return {"info": self.info, "licenses": self.license, "images": self.images, "annotations": self.annotations, "categories": self.categories}

    def write2File(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.json_dict, f, ensure_ascii=False, indent=4)

    def get_license(self):
        license_obj = Licenses()
        return license_obj.getLicense(self.dataset)

    def getCategoryinfo(self):
        for category in self.categories:
            if category["name"] == self.object:
                return category
            else:
                return None


class info:
    def __init__(self):

        self.description = ""

    def getDatasetInfo(self, dataset_name):
        # Dataset Name: SPEED , ...
        speed = {
            "description": "Spacecraft Pose Estimation Dataset (SPEED)",
            "url": "https://purl.stanford.edu/dz692fn7184",
            "version": "N/A",
            "year": 2019,
            "contributor": "Stanford Rendevous Lab (SLAB)",
            "date_created": "2019/02/01"
        }

        speedplus = {
            "description": "Next Generation Spacecraft Pose Estimation Dataset (SPEED+)",
            "url": "https://purl.stanford.edu/wv398fc4383",
            "version": "N/A",
            "year": 2021,
            "contributor": "Tae Ha Park, Marcus Märtens, Gurvan Lecuyer, Dario Izzo, & Simone D'Amico.",
            "date_created": "2021/10/25"
        }
        # Enter Envisat data here
        envisat_1 = {
            "description": "Envisat Set 1",
            "url": "https://tudelft.nl",
            "version": "1",
            "year": 2020,
            "contributor": "TU Delft Space Engineering",
            "date_created": "2019/06/01"
        }

        if dataset_name == 'SPEED':
            return speed
        elif dataset_name == 'SPEED+':
            return speedplus
        elif dataset_name == 'ENVISAT-1':
            return envisat_1
        else:
            return None


class Licenses:
    '''-----------------Format----------------------------
    licenses:[{
      "id": int,
      "name": str,
      "url": str,
      }]
    ---------------------------------------------------'---'''

    def __init__(self):
        self.dataset_license = {"SPEED": "cc-by-nc-sa-3.0",
                                "SPEED+": "cc-by-nc-sa-4.0", "ENVISAT-1": "None"}
        self.License_dict = self.supportedLicenses()

    def supportedLicenses(self):
        by_na_sa3 = {"id": 1, "url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
                     "name": "Attribution-NonCommercial-ShareAlike 3.0"}
        by_na_sa4 = {"id": 2, "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                     "name": "Attribution-NonCommercial-ShareAlike 4.0"}
        no_license = {"id": 0, "url": "N/A", "name": "N/A"}

        Licenses = {"None": no_license,
                    "cc-by-nc-sa-3.0": by_na_sa3, "cc-by-nc-sa-4.0": by_na_sa4}
        return Licenses

    def getLicense(self, dataset):
        return self.License_dict[self.dataset_license[dataset]]


class Category:
    # For each class of object
    # provide arguments with object name in kwargs
    def __init__(self, **kwargs):
        self.objects = {"Tango": 1, "Envisat": 2}
        self.supercategory = "Target"
        self.object = kwargs["object"] if kwargs["object"] in list(
            self.objects) else None
        self.id = self.objects[self.object]
        self.keypoints = [str(i) for i in range(0, kwargs['n_keypoints'])]
        self.skeleton = kwargs['skeleton']

    def get_dict(self):
        category = {"supercategory": self.supercategory, "id": self.id,
                    "name": self.object, "keypoints": self.keypoints, "skeleton": self.skeleton}
        return category


class Images:
    # Creates Images and Annotations
    '''------------------------FORMAT ----------------------------
    images: [{
        "id": int,
        "width": int,
        "height": int,
        "file_name": str,
        "license": int,
        "flickr_url": str,
        "coco_url": str,
        "date_captured": datetime,
        }]
    annotations: [{
                    annotation
                }]
    --------------------------------------------------------------'''

    def __init__(self, **kwargs):
        #size - 2-tuple (w,h)
        self.extension = kwargs["extension"]
        self.size = kwargs["size"]
        self.license = kwargs["license_id"]
        self.height = self.size[0]  # px
        self.width = self.size[1]  # px
        self.url = kwargs["url"]  # "https://purl.stanford.edu/dz692fn7184"
        self.captured = "N/A"
        #self.flickr_url = None
        self.image_path_list = []
        self.data = kwargs["data"]
        self.index_map = self.get_data_index_map(kwargs["data"])
        self.img_dir = os.path.abspath(kwargs["image_path"])
        self.annotations, self.ImageList = self.processImages(
            **kwargs)  # Self.List is the dict object

    def getCocoDict(self, filename, ID):
        image = {"license": self.license, "file_name": filename, "coco_url": "",
                 "height": self.height, "width": self.width, "date_captured": "", "url": self.url, "id": ID}
        return image

    def get_data_index_map(self, ann_data):
        index_map = {}
        for idx, entry in enumerate(ann_data):
            index_map[entry["filename"]] = idx
        return index_map

    def processImages(self, **kwargs):
        print('Processing Images and Annotations')
        img_path = self.img_dir
        self.image_path_list = glob.glob(
            os.path.join(img_path, f"*.{self.extension}"))
        image_obj_list = []
        annotations = []
        n = len(self.image_path_list)
        for idx, path in enumerate(self.image_path_list):
            if (idx/n*100) % 10 == 0:
                print(f"{idx/n*100+1} %")
            filename = os.path.basename(path)
            if filename in list(self.index_map):
                img_name, _ = os.path.splitext(filename)
                img_id = int(img_name.split(kwargs["img_name_prefix"])[-1])

                img_dict = self.getCocoDict(filename, img_id)
                img_data = self.data[self.index_map[filename]]

                annotation = Annotation(
                    idx+1, img_id, filename, img_data, **kwargs)
                ann_dict = annotation.getAnnotationDict()

                image_obj_list.append(img_dict)
                annotations.append(ann_dict)
        print('Processed images and annotations!')
        return annotations, image_obj_list

    def get_images_annotations(self):
        return self.ImageList, self.annotations


class Annotation:
    ''' -------------------------FORMAT -------------------------------
    annotation:{
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
        }
        --------------------------------------------------------------'''

    def __init__(self, annotation_id, ID, name, img_data, **kwargs):
        self.id = annotation_id
        self.segmentation_available = False
        if not self.segmentation_available:
            self.segmentation = []
            self.area = 0
        else:
            # implement segmentation info inclusion
            pass
        self.num_keypoints = len(kwargs["category"]["keypoints"])
        self.category_id = kwargs["category"]["id"]
        self.image_id = ID
        self.data = img_data
        self.name = name
        self.iscrowd = 0
        self.category_id = kwargs["category"]["id"]
        self.keypoints, self.bbox, self.area = self.getImgInfofromData()

    def getImgInfofromData(self):
        if self.data["filename"] == self.name:
            # bbox key should be label or bbox #changed for envisat
            bbox = self.data["label"] if 'label' in list(
                self.data) else self.data["bbox"]
            bbox, area = self.bbox_XY2WH(bbox)
            features = self.data["features"]
            kps_coco = []
            for feature in features:
                coordinates = list(feature["Coordinates"])
                visibility = feature["Visibility"]
                kps_coco = kps_coco + coordinates+[visibility]
        return kps_coco, bbox, area

    def bbox_XY2WH(self, bbox):
        # [x_min, x_max, y_min, y_max] to [x,y,w,h]
        bbox_new = [bbox[0], bbox[2], (bbox[1]-bbox[0]), (bbox[3]-bbox[2])]
        area = bbox_new[2]*bbox_new[3]
        return bbox_new, area

    def getAnnotationDict(self):
        annotation = {"segmentation": self.segmentation, "num_keypoints": self.num_keypoints, "area": self.area, "iscrowd": self.iscrowd,
                      "keypoints": self.keypoints, "image_id": self.image_id, "bbox": self.bbox, "category_id": self.category_id, "id": self.id}
        return annotation


"""
Customize the spacecraft wireframe model here
"""


def dataset_kp_info(dataset):
    '''
    dataset: Supported Dataset Names:
                SPEED : 'SPEED'
                Envisat_Set1: ENVISAT-1
    Returns
    -------
    dictionary with n_keypoints and skeleton for the dataset
    '''
    tango_KPs_n = 11
    tango_skeleton = [[1, 2], [2, 3], [3, 4], [4, 1], [1, 5], [2, 6], [3, 7], [
        4, 8], [5, 6], [6, 7], [7, 8], [8, 5], [1, 9], [2, 10], [4, 11]]

    Envisat_KPs_n = 16
    Envisat_skeleton = [[1, 2], [2, 4], [4, 3], [3, 1],
                        [5, 6], [6, 7], [7, 8], [8, 5],
                        [9, 10], [10, 12], [11, 12], [11, 9],
                        [13, 14], [14, 16], [16, 15], [15, 13],
                        [1, 11], [2, 12], [3, 9], [4, 10],
                        [10, 13], [12, 14]]
    all_dict = {
        "Tango": {"keypoints": tango_KPs_n, "skeleton": tango_skeleton},
        "ENVISAT-1": {"keypoints": Envisat_KPs_n, "skeleton": Envisat_skeleton}
    }

    return all_dict[dataset]['keypoints'], all_dict[dataset]['skeleton']
# ''' Add Dataset name:
#     SPEED : 'SPEED'
#     Envisat_Set1: 'ENVISAT-1'
# '''

# dataset_name= 'ENVISAT-1'


# n_kps, skeleton = dataset_kp_info(dataset_name)
# base_args = {
#         "dataset":dataset_name,
#         "size":(512,512),
#         "url": "",
#         "n_keypoints":n_kps,
#         "skeleton": skeleton
#         }


# #TRAIN
# train_args = base_args
# train_args["image_path"] = 'Envisat_Set1/train'
# train_args['datafile'] = 'Envisat_Set1/json_data/bbox/train_bbox.json'
# train_json_path = 'Envisat_Set1/json_data/COCO/train.json' #to write to

# import time
# start = time.time()
# train_COCO = COCO_SatPose(**train_args)
# train_COCO.write2File(train_json_path)
# print(time.time()-start)
# #coco = COCO('images/train.json')

# #VAL
# val_args = base_args
# val_args["image_path"] = 'Envisat_Set1/val'
# val_args['datafile'] = 'Envisat_Set1/json_data/bbox/val_bbox.json'
# val_json_path = 'Envisat_Set1/json_data/COCO/val.json'
# val_COCO = COCO_SatPose(**val_args)
# val_COCO.write2File(val_json_path)
