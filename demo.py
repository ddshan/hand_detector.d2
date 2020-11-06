# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
from collections import OrderedDict
from tqdm import tqdm
import torch
import cv2
import random

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.pascal_voc import register_pascal_voc

_datasets_root = "datasets"
for d in ["trainval", "test"]:
    register_pascal_voc(name=f'100DOH_hand_{d}', dirname=_datasets_root, split=d, year=2007, class_names=["hand"])
    MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='pascal_voc')


def main():
    # load cfg and model
    cfg = get_cfg()
    cfg.merge_from_file("faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
    cfg.MODEL.WEIGHTS = 'models/model_0529999.pth' # add model weight here
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 0.5 , set the testing threshold for this model

    # data path
    test_img = './viz/input.jpg'
    save_dir = './viz'
    os.makedirs(save_dir, exist_ok=True)

    # predict
    predictor = DefaultPredictor(cfg)

    # output
    im = cv2.imread(test_img)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('./viz/output_100Kego.jpg', v.get_image()[:, :, ::-1])

    # print
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    

if __name__ == '__main__':
    main()