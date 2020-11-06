# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This batch prediction script is shared by my labmate Shengyi Qian (https://jasonqsy.github.io/)
"""

import numpy as np
import logging
import os
from collections import OrderedDict
from tqdm import tqdm
import torch
import cv2
import random
import pdb

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.data.datasets.pascal_voc import register_pascal_voc

_datasets_root = "datasets"
for d in ["trainval", "test"]:
    register_pascal_voc(name=f'100DOH_hand_{d}', dirname=_datasets_root, split=d, year=2007, class_names=["hand"])
    MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='pascal_voc')


class BatchPredictor:
    """
    The batch version of detectron2 DefaultPredictor
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, imgs):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            inputs = []
            for original_image in imgs:
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                entry = {"image": image, "height": height, "width": width}
                inputs.append(entry)
            
            # inference
            predictions = self.model(inputs)
            
            return predictions


def main():
    # load cfg and model
    cfg = get_cfg()
    cfg.merge_from_file("faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
    cfg.MODEL.WEIGHTS = 'models/model_0529999.pth' # add model weight here
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 0.5 , set the testing threshold for this model

    # predict
    predictor = BatchPredictor(cfg)

    # data path
    test_img = './viz/input.jpg'
    save_dir = './viz'
    os.makedirs(save_dir, exist_ok=True)

    # prepare input
    im = cv2.imread(test_img)
    imgs = []
    for _ in range(8):
        imgs.append(im)

    outputs = predictor(imgs)
    for i in range(8):
        print(i)
        print(outputs[i])


if __name__ == '__main__':
    main()