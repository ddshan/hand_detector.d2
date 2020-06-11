# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""


import logging
import os
from collections import OrderedDict
from tqdm import tqdm
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
import cv2
import random



if __name__ == '__main__':

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
    