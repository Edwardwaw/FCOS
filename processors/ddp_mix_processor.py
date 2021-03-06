import os
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.distributed as dist
from torch.cuda import amp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import interpolate
from torch.utils.tensorboard import SummaryWriter

from datasets.coco import COCODataSets
from nets.fcos import FCOS
from losses.fcos_loss import FCOSLoss
from utils.fcos import non_max_suppression
from commons.model_utils import rand_seed, is_parallel, ModelEMA, freeze_bn
from metrics.map import coco_map
from commons.optims_utils import WarmUpCosineDecayMultiStepLRAdjust, split_optimizer


rand_seed(1024)


class DDPMixProcessor(object):
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as rf:
            self.cfg = yaml.safe_load(rf)
        self.data_cfg = self.cfg['data']
        self.model_cfg = self.cfg['model']
        self.optim_cfg = self.cfg['optim']
        self.hyper_params = self.cfg['hyper_params']
        self.val_cfg = self.cfg['val']
        print(self.data_cfg)
        print(self.model_cfg)
        print(self.optim_cfg)
        print(self.hyper_params)
        print(self.val_cfg)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['gpus']
        dist.init_process_group(backend='nccl',init_method='env://')

        self.tdata = COCODataSets(img_root=self.data_cfg['train_img_root'],
                                  annotation_path=self.data_cfg['train_annotation_path'],
                                  img_size=self.data_cfg['img_size'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=True,
                                  remove_blank=self.data_cfg['remove_blank']
                                  )
        self.tloader = DataLoader(dataset=self.tdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.tdata.collate_fn,
                                  sampler=DistributedSampler(dataset=self.tdata, shuffle=True))
        self.vdata = COCODataSets(img_root=self.data_cfg['val_img_root'],
                                  annotation_path=self.data_cfg['val_annotation_path'],
                                  img_size=self.data_cfg['img_size'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=False,
                                  remove_blank=False
                                  )
        self.vloader = DataLoader(dataset=self.vdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.vdata.collate_fn,
                                  sampler=DistributedSampler(dataset=self.vdata, shuffle=False))
        print("train_data: ", len(self.tdata), " | ",
              "val_data: ", len(self.vdata), " | ",
              "empty_data: ", self.tdata.empty_images_len)
        print("train_iter: ", len(self.tloader), " | ",
              "val_iter: ", len(self.vloader))

        local_rank = dist.get_rank()
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)

        model = FCOS(num_cls=self.model_cfg['num_cls'],
                     strides=self.model_cfg['strides'],
                     backbone=self.model_cfg['backbone'],
                     )
        optimizer = split_optimizer(model, self.optim_cfg)
        model.to(self.device)
        pretrain=self.model_cfg.get('pretrain',None)
        if pretrain is not None:
            pretrained_weights = torch.load(pretrain, map_location=self.device)
            load_info = model.load_state_dict(pretrained_weights['ema'], strict=False)
            print('load info ', load_info)

        self.scaler = amp.GradScaler(enabled=True)
        if self.optim_cfg['sync_bn']:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = nn.parallel.distributed.DistributedDataParallel(model,
                                                                     device_ids=[local_rank],
                                                                     output_device=local_rank)
        self.optimizer = optimizer
        self.ema = ModelEMA(self.model)

        self.creterion = FCOSLoss(
            alpha=self.hyper_params['alpha'],
            gamma=self.hyper_params['gamma'],
            radius=self.hyper_params['radius'],
            layer_limits=self.hyper_params['layer_limits'],
            strides=self.model_cfg['strides'],
            iou_type=self.hyper_params['iou_type']
        )
        self.lr_adjuster = WarmUpCosineDecayMultiStepLRAdjust(init_lr=self.optim_cfg['lr'],
                                                              milestones=self.optim_cfg['milestones'],
                                                              warm_up_epoch=self.optim_cfg['warm_up_epoch'],
                                                              iter_per_epoch=len(self.tloader),
                                                              epochs=self.optim_cfg['epochs'],
                                                              cosine_weights=self.optim_cfg['cosine_weights']
                                                              )

        self.tb_writer=None
        if self.local_rank==0:
            log_dir = 'runs/'
            print('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % log_dir)
            self.tb_writer = SummaryWriter(log_dir=log_dir)

        self.best_map = 0.
        self.best_map50 = 0.


    def train(self, epoch):
        self.model.train()
        if self.model_cfg['freeze_bn']:
            self.model.apply(freeze_bn)
        if self.local_rank == 0:
            pbar = tqdm(self.tloader)
        else:
            pbar = self.tloader
        loss_list = [list(), list(), list(), list()]
        lr = 0
        match_num = 0
        for i, (img_tensor, targets_tensor, _) in enumerate(pbar):
            if len(self.hyper_params['multi_scale']) > 2:
                target_size = np.random.choice(self.hyper_params['multi_scale'])
                img_tensor = interpolate(img_tensor, mode='bilinear', size=target_size, align_corners=False)
            _, _, h, w = img_tensor.shape
            with torch.no_grad():
                targets_tensor[:, 3:] = targets_tensor[:, 3:] * torch.tensor(data=[w, h, w, h])
                img_tensor = img_tensor.to(self.device)
                targets_tensor = targets_tensor.to(self.device)
            self.optimizer.zero_grad()
            # Forward
            # 混合精度
            with amp.autocast(enabled=True):
                cls_outputs, reg_outputs, center_outputs, grids = self.model(img_tensor)
                total_loss, detail_loss, total_num = self.creterion(cls_outputs,
                                                                    reg_outputs,
                                                                    center_outputs,
                                                                    grids,
                                                                    targets_tensor)
            match_num += total_num
            self.scaler.scale(total_loss).backward()
            self.lr_adjuster(self.optimizer, i, epoch)
            lr0 = self.optimizer.param_groups[0]['lr']
            lr1 = self.optimizer.param_groups[1]['lr']
            lr2 = self.optimizer.param_groups[2]['lr']
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)

            loss_cls, loss_reg, loss_center = detail_loss
            loss_list[0].append(total_loss.item())
            loss_list[1].append(loss_cls.item())
            loss_list[2].append(loss_reg.item())
            loss_list[3].append(loss_center.item())
            if self.local_rank == 0:
                pbar.set_description(
                    "epoch:{:3d}|match_num:{:4d}|size:{:3d}|target_loss:{:6.4f}|loss_cls:{:6.4f}|loss_reg:{:6.4f}|loss_cen:{:6.4f}|lr:{:8.6f}".format(
                        epoch + 1,
                        int(total_num),
                        h,
                        total_loss.item(),
                        loss_cls.item(),
                        loss_reg.item(),
                        loss_center.item(),
                        lr0
                    ))
            if self.tb_writer:
                tags = ['train/total_loss', 'train/cls_loss', 'train/reg_loss', 'train/center_loss',  # train loss
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
                for x, tag in zip([total_loss.item(), loss_cls.item(), loss_reg.item(), loss_center.item(), lr0, lr1, lr2],
                                  tags):
                    self.tb_writer.add_scalar(tag, x, i + 1 + epoch * len(self.tloader))


        self.ema.update_attr(self.model)
        mean_loss_list = [np.array(item).mean() for item in loss_list]
        print(
            "epoch:{:3d}|match_num:{:4d}|local:{:3d}|target_loss:{:6.4f}|loss_cls:{:6.4f}|loss_reg:{:6.4f}|loss_cen:{:6.4f}|lr:{:8.6f}"
                .format(epoch + 1,
                        int(match_num),
                        self.local_rank,
                        mean_loss_list[0],
                        mean_loss_list[1],
                        mean_loss_list[2],
                        mean_loss_list[3],
                        lr))

    @torch.no_grad()
    def val(self, epoch):
        predict_list = list()
        target_list = list()
        self.model.eval()
        if self.local_rank == 0:
            pbar = tqdm(self.vloader)
        else:
            pbar = self.vloader
        for img_tensor, targets_tensor, _ in pbar:
            _, _, h, w = img_tensor.shape
            targets_tensor[:, 3:] = targets_tensor[:, 3:] * torch.tensor(data=[w, h, w, h])
            img_tensor = img_tensor.to(self.device)
            targets_tensor = targets_tensor.to(self.device)
            predicts = self.ema.ema(img_tensor)

            for i in range(len(predicts)):
                predicts[i][:, [0, 2]] = predicts[i][:, [0, 2]].clamp(min=0, max=w)
                predicts[i][:, [1, 3]] = predicts[i][:, [1, 3]].clamp(min=0, max=h)
            predicts = non_max_suppression(predicts,
                                           conf_thresh=self.val_cfg['conf_thresh'],
                                           iou_thresh=self.val_cfg['iou_thresh'],
                                           max_det=self.val_cfg['max_det'],
                                           )
            for i, predict in enumerate(predicts):
                predict_list.append(predict)
                targets_sample = targets_tensor[targets_tensor[:, 0] == i][:, 2:]
                target_list.append(targets_sample)
        mp, mr, map50, map = coco_map(predict_list, target_list)
        print("epoch: {:2d}|local:{:d}|mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}"
              .format(epoch + 1,
                      self.local_rank,
                      mp * 100,
                      mr * 100,
                      map50 * 100,
                      map * 100))
        last_weight_path = os.path.join(self.val_cfg['weight_path'],
                                        "{:s}_last.pth"
                                        .format(self.cfg['model_name']))
        best_map_weight_path = os.path.join(self.val_cfg['weight_path'],
                                            "{:s}_best_map.pth"
                                            .format(self.cfg['model_name']))
        best_map50_weight_path = os.path.join(self.val_cfg['weight_path'],
                                              "{:s}_best_map50.pth"
                                              .format(self.cfg['model_name']))


        ema_static = self.ema.ema.state_dict()
        cpkt = {
            "ema": ema_static,
            "map": map * 100,
            "epoch": epoch,
            "map50": map50 * 100
        }
        if self.local_rank != 0:
            return

        if self.tb_writer:
            tags = ['metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']
            for x, tag in zip([mp, mr, map50, map], tags):
                self.tb_writer.add_scalar(tag, x, epoch + 1)

        torch.save(cpkt, last_weight_path)
        if map > self.best_map:
            torch.save(cpkt, best_map_weight_path)
            self.best_map = map
        if map50 > self.best_map50:
            torch.save(cpkt, best_map50_weight_path)
            self.best_map50 = map50

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
        dist.destroy_process_group()
        torch.cuda.empty_cache()


