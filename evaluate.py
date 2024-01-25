# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-8-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from lib.metrics import *
from lib.dataset import MLDataset
from models.factory import create_model

import numpy as np
from thop import profile

torch.backends.cudnn.benchmark = True


# correct_pred = np.zeros([80],dtype=np.int32)
# total_pred = np.zeros([80],dtype=np.int32)

# again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1


# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

class Evaluator(object):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        dataset = MLDataset(cfg.test_path, cfg.label_path, cfg.img_size, is_train=False)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        self.labels = dataset.labels
        
        # cfg.model = convnext_small
        self.model = create_model(cfg.model, pretrained=False, cfg=cfg)
        self.model.cuda()

        self.cfg = cfg
        self.voc07_mAP = VOCmAP(cfg.num_classes, year='2007', ignore_path=cfg.ignore_path)
        self.voc12_mAP = VOCmAP(cfg.num_classes, year='2007', ignore_path=cfg.ignore_path)

        self.meter1 = TopkMeter(cfg.num_classes, ignore_path=cfg.ignore_path)
        self.meter2 = ThresholdMeter(cfg.num_classes, ignore_path=cfg.ignore_path)

        # self.voc07_mAP1 = VOCmAP(1, year='2007', ignore_path=cfg.ignore_path)

    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_best_path)
        self.model.load_state_dict(model_dict, strict=True)
        print(f'loading best checkpoint success')
        
        self.model.eval()
        # print(self.model)
        self.voc07_mAP.reset()
        self.voc12_mAP.reset()

        self.meter1.reset()
        self.meter2.reset()

        # print(self.dataloader)
        
        for batch in tqdm(self.dataloader):
            # print(batch['img_path'])
            
            img = batch['img'].cuda()
            targets = batch['target'].numpy()
            logits, _ = self.model(img)
            # print size of model
            # flops, params = profile(self.model, (img,))
            # print('flops: ', flops, 'params: ', params)

            scores = torch.sigmoid(logits).cpu().numpy()
            self.voc07_mAP.update(scores, targets)
            self.voc12_mAP.update(scores, targets)

            self.meter1.update(scores, targets)
            self.meter2.update(scores, targets)

            # print(scores.shape, targets.shape, self.voc12_mAP)
            # for target, score in zip(targets, scores):
            #     # print(score.shape, target.shape)
            #     for j in range(80):
            #         if score[j] == target[j]:
            #             correct_pred[j] += 1
            #         total_pred[j] += 1

            # self.voc07_mAP.update(scores[:,1], targets[:,1])
            
        aps1, mAP_07 = self.voc07_mAP.compute()
        aps2, mAP_12 = self.voc12_mAP.compute()
        print('model {} data {} mAP_07: {:.4f} mAP_12: {:.4f}'.format(self.cfg.model, self.cfg.data, mAP_07, mAP_12))
        print(aps1)

        # accuracy = 100 * float(correct_pred[0]) / total_pred[0]
        # print("Accuracy for class 0 is: {:.1f} %".format(accuracy))

        _op, _or, _of1, _cp, _cr, _cf1 = self.meter1.compute()
        print("top3 op: {:.4f}, or: {:.4f}, of1: {:.4f}, cp: {:.4f}, cr: {:.4f}, cf1: {:.4f},"
                       .format(_op, _or, _of1, _cp, _cr, _cf1))
        _op, _or, _of1, _cp, _cr, _cf1 = self.meter2.compute()
        print("all op: {:.4f}, or: {:.4f}, of1: {:.4f}, cp: {:.4f}, cr: {:.4f}, cf1: {:.4f},"
                       .format(_op, _or, _of1, _cp, _cr, _cf1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='experiments/convnext_small_mscoco/exp1')
    args = parser.parse_args()
    cfg_path = os.path.join(args.exp, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    cfg = Namespace(**cfg)
    print(cfg)
    
    evaluator = Evaluator(cfg)
    evaluator.run()