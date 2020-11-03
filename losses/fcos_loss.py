import torch
import numpy as np
from torch import nn
from utils.fcos import BoxCoder
from losses.common import IOULoss



INF=1e8



def iou(predicts,targets,weights=None,iou_type='giou'):
    '''
    params
    :param predicts:
    :param targets:
    :param weights:
    :param iou_type:
    :return:
    '''
    assert len(predicts)==len(targets)
    pred_left=predicts[:,0]
    pred_top=predicts[:,1]
    pred_right=predicts[:,2]
    pred_bottom=predicts[:,3]

    target_left = targets[:, 0]
    target_top = targets[:, 1]
    target_right = targets[:, 2]
    target_bottom = targets[:, 3]

    target_area=(target_left+target_right)*(target_top+target_bottom)
    pred_area=(pred_left+pred_right)*(pred_top+pred_bottom)

    w_intersect=torch.min(pred_left,target_left)+torch.min(pred_right,target_right)
    h_intersect=torch.min(pred_top,target_top)+torch.min(pred_bottom,target_bottom)

    w_outer=torch.max(pred_left,target_left)+torch.max(pred_right,target_right)
    h_outer=torch.max(pred_top,target_top)+torch.max(pred_bottom,target_bottom)

    intersect_area=w_intersect*h_intersect
    outer_area=w_outer*h_outer+1e-6
    union_area=target_area+pred_area-intersect_area

    ious = (intersect_area+1.0)/(union_area+1.0)  # for compute stablity?
    gious = ious-(outer_area-union_area)/outer_area

    if iou_type == 'iou':
        losses = -ious.log()
    elif iou_type == 'giou':
        losses = 1-gious
    else:
        raise NotImplementedError
    if weights is not None and weights.sum()>0:
        return (losses*weights).sum()
    else:
        assert losses.numel() != 0
        return losses.sum()






class FCOSLossBuilder(object):
    # target build class
    def __init__(self,radius=1,strides=None,layer_limits=None):
        self.radius=radius   # decide matching method
        self.box_coder=BoxCoder()
        if strides is None:
            strides=[8,16,32,64,128]
        self.strides=torch.tensor(strides)
        if layer_limits is None:
            layer_limits=[64,128,256,512]
        expand_limits=np.array(layer_limits)[None].repeat(2).tolist() # list, len=8
        '''
        FCOS通过规定每一层预测的尺度范围来避免一个sample匹配到多个target的情况
        self.layer_limits=[ [-1,64],         shape=[5,2]
                            [64,128],
                            [128,256],
                            [256,512],
                            [512,inf]]
        '''
        self.layer_limits=torch.tensor([-1.]+expand_limits+[INF]).view(-1,2)

    @torch.no_grad()
    def __call__(self, bs, grids, targets):
        '''
        params
        :param bs: batch_size
        :param grids (list, len=num_layer, num_layer=5) : its element shape = [h,w,2] 2==>(x,y) 原图尺度
        :param targets : [gts,7] (batch_id,weights,label_id,x1,y1,x2,y2)   x1,y1,x2,y2==>原图尺度
        :return:
        batch_reg_targets (list, len=bs): its element shape = [num_grids,4]   4==>(l*,t*,r*,b*)
                                          注意，即使是副样本也可能有对应的为正的取值，因此要和label_target结合来甄别出副样本
        batch_label_targets (list, len=bs): its element shape = [num_grids,1]  1==>label id (匹配到某个gt_box的 class id)  1==>-1 (neg_sample)
        '''
        device=grids[0].device
        self.layer_limits=self.layer_limits.to(device)
        self.strides=self.strides.to(device)

        # [num_grids,5]  5==>(xc,yc,min_limit,max_limit,stride)   num_grids = num of grid points among all featuremaps
        expand_grid=torch.cat(
            [torch.cat([grid,layer_limit.expand_as(grid),stride.expand_as(grid[...,[0]])],
                       dim=-1).view(-1,5)
            for grid,layer_limit,stride in zip(grids,self.layer_limits,self.strides)
            ],dim=0)

        # build targets for each image
        batch_reg_targets=list()
        batch_label_targets=list()

        for bi in range(bs):
            batch_targets=targets[targets[:,0]==bi,1:]   # [num_gts,6]==>(weights,label_id,x1,y1,x2,y2)
            # no target in the image
            if len(batch_targets)==0:
                batch_reg_targets.append(torch.Tensor())
                batch_label_targets.append(torch.ones(size=(len(expand_grid),),
                                                      device=device,
                                                      dtype=torch.float32)*-1)
                continue

            # encode grid point with all targets
            reg_target_per_img=self.box_coder.encoder(expand_grid,batch_targets[:,2:])  # shape=[num_grids,num_gts,4] (l,t,r,b)==>原图尺度

            # 筛选条件1:
            '''
            if self.radius==0: 删除掉那些不再gt_box区域内部的点/样本
            else: 删除掉那些不在以gt_box中心为依据生成的坐标为
                    (gt_xc-radius*stride,gt_yc-radius*stride,gt_xc+radius*stride,gt_yc-radius*stride)且不在gt_box区域内部的点/样本
            '''
            if self.radius==0:
                valid_in_box=reg_target_per_img.min(dim=2)[0]>0
            else:
                limit_gt_xy = (batch_targets[:,[2,3]]+batch_targets[:,[4,5]])/2.0  # shape=[num_gts,2], 2==>(xc,yc)
                limit_gt_min_xy = limit_gt_xy[None,:,:]-expand_grid[:,None,[4,4]]*self.radius   # [1,num_gts,2]-[num_grids,1,2]=[num_grids,num_gts,2]
                limit_gt_max_xy = limit_gt_xy[None,:,:]+expand_grid[:,None,[4,4]]*self.radius   # [1,num_gts,2]+[num_grids,1,2]=[num_grids,num_gts,2]
                limit_gt_min_xy = torch.where(limit_gt_min_xy>batch_targets[None,:,[2,3]],
                                              limit_gt_min_xy,batch_targets[None,:,[2,3]])
                limit_gt_max_xy = torch.where(limit_gt_max_xy<batch_targets[None,:,[4,5]],
                                              limit_gt_max_xy,batch_targets[None,:,[4,5]])

                left_top=expand_grid[:,None,[0,1]]-limit_gt_min_xy
                right_bottom=limit_gt_max_xy-expand_grid[:,None,[0,1]]
                valid_in_box=torch.cat([left_top,right_bottom],dim=2).min(dim=2)[0]>0

            # 筛选条件2:
            '''
            删除掉那些所对应的gt_box不符合尺度限制条件的样本
            '''
            max_reg_targets_per_img=reg_target_per_img.max(dim=2)[0]  # shape=[num_grids,num_gts]
            is_card_in_level=(max_reg_targets_per_img>=expand_grid[:,[2]])&(
                max_reg_targets_per_img<=expand_grid[:,[3]])

            #
            gt_area = (batch_targets[:, 4] - batch_targets[:, 2]) * (batch_targets[:, 5] - batch_targets[:, 3])  # shape=[num_gts]
            locations_to_gt_area=gt_area[None,:].repeat(len(expand_grid),1)   # shape=[num_grid,num_gts]
            # 筛选掉不符合条件的样本
            locations_to_gt_area[~valid_in_box]=INF
            locations_to_gt_area[~is_card_in_level]=INF
            # 筛选条件3:
            '''
            当某个grid/sample匹配到多个gt_box时,选择面积最小的gt_box作为其匹配到的target
            '''
            min_area,gt_idx=locations_to_gt_area.min(dim=1)  # shape=[num_grids], calculate the minest gt_box area coorsponding to grid/sample
            reg_target_per_img=reg_target_per_img[range(len(reg_target_per_img)),gt_idx]  # shape=[num_grids,4]  (l,t,r,b)==>原图尺度

            labels_per_img=batch_targets[:,1][gt_idx]
            labels_per_img[min_area==INF]=-1
            batch_reg_targets.append(reg_target_per_img)
            batch_label_targets.append(labels_per_img)

        return batch_reg_targets,batch_label_targets














class FCOSLoss(object):

    def __init__(self,radius=0,alpha=0.25,gamma=2.0,iou_type='giou',strides=None,layer_limits=None):
        super(FCOSLoss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.iou_loss=IOULoss(iou_type,coord_type='ltrb')
        self.builder=FCOSLossBuilder(radius=radius,strides=strides,layer_limits=layer_limits)
        self.centerness_loss=nn.BCEWithLogitsLoss(reduction='sum')

    @staticmethod
    def build_centerness(reg_targets):
        '''
        :param reg_targets (torch.Tensor):  shape=[num_reg,4] 4==>(l*,t*,r*,b*)
        :return:
        torch.sqrt(centerness): shape=[num_reg]
        '''
        left_right=reg_targets[:,[0,2]]
        top_bottom=reg_targets[:,[1,3]]
        centerness=(left_right.min(dim=-1)[0]/left_right.max(dim=-1)[0])*(
            top_bottom.min(dim=-1)[0]/top_bottom.max(dim=-1)[0]
            )
        return torch.sqrt(centerness)


    def __call__(self, cls_outputs, reg_outputs, center_outputs, grids, targets):
        '''
        params
        :param cls_outputs (list, len=num_layer, num_layer=5) :    its element shape = [bs,80,h,w]
        :param reg_outputs (list, len=num_layer, num_layer=5) : its element shape = [bs,4,h,w]
        :param center_outputs (list, len=num_layer, num_layer=5) : its element shape = [bs,1,h,w]
        :param grids (list, len=num_layer, num_layer=5) : its element shape = [h,w,2] 2==>(x,y) 原图尺度
        :param targets () : [gts,7] (batch_id,weights,label_id,x1,y1,x2,y2)   x1,y1,x2,y2==>原图尺度
        :return:
        '''
        device=cls_outputs[0].device
        bs=cls_outputs[0].shape[0]
        cls_num=cls_outputs[0].shape[1]


        cls_loss_list=list()
        reg_loss_list = list()
        center_loss_list = list()
        num_pos=0

        for i in range(len(reg_outputs)):
            if reg_outputs[i].dtype==torch.float16:
                reg_outputs[i]=reg_outputs[i].float()
        for i in range(len(center_outputs)):
            if center_outputs[i].dtype==torch.float16:
                center_outputs[i]=center_outputs[i].float()

        '''
        batch_reg_targets (list, len=bs): its element shape = [num_grids,4]   4==>(l*,t*,r*,b*)
                                          注意，即使是副样本也可能有对应的为正的取值，因此要和label_target结合来甄别出副样本
        batch_label_targets (list, len=bs): its element shape = [num_grids,1]  1==>label id (匹配到某个gt_box的 class id)  1==>-1 (neg_sample)
        '''
        reg_targets,labels_targets=self.builder(bs,grids,targets)
        for bi,batch_reg_targets,batch_label_targets in zip(range(bs),reg_targets,labels_targets):
            pos_idx=(batch_label_targets>=0).nonzero(as_tuple=False).squeeze(1)

            # batch_cls_predicts.shape=[num_grids,cls_num]    num_grids=num of grid point among all featuremaps
            batch_cls_predicts=torch.cat([cls_output_item[bi].permute(1,2,0).contiguous().view(-1,cls_num)
                                          for cls_output_item in cls_outputs],dim=0).sigmoid().clamp(1e-6,1-1e-6)

            if len(pos_idx)==0:
                cls_neg_loss = -(1-self.alpha)*(batch_cls_predicts**self.gamma)*(1-batch_cls_predicts).log()
                cls_loss_list.append(cls_neg_loss.sum())
                continue

            num_pos += len(pos_idx)

            # batch_reg_predicts.shape=[num_grids,4]
            batch_reg_predicts=torch.cat([reg_output_item[bi].permute(1,2,0).contiguous().view(-1,4)
                                          for reg_output_item in reg_outputs],dim=0)
            # batch_centerness_predicts.shape=[num_grids]
            batch_centerness_predicts=torch.cat([center_output_item[bi].permute(1, 2, 0).contiguous().view(-1)
                                                 for center_output_item in center_outputs],dim=0)

            pos_reg_predicts=batch_reg_predicts[pos_idx,:]
            pos_reg_targets=batch_reg_targets[pos_idx,:]
            pos_center_predicts=batch_centerness_predicts[pos_idx]
            pos_centerness_targets=self.build_centerness(pos_reg_targets)

            # loss 1 (reg loss): weighted iou_loss
            # iou_loss = iou(pos_reg_predicts, pos_reg_targets, weights=pos_centerness_targets)
            iou_loss = self.iou_loss(pos_reg_predicts, pos_reg_targets)
            iou_loss = (iou_loss * pos_centerness_targets).sum()

            # loss 2 (centerness loss): bce_loss
            centerness_loss=self.centerness_loss(pos_center_predicts,pos_centerness_targets)

            # loss 3 (class loss): focal loss
            cls_targets=torch.zeros_like(batch_cls_predicts)
            cls_idx=batch_label_targets[pos_idx]
            cls_targets[pos_idx,cls_idx.long()]=1.

            cls_pos_loss=-self.alpha*cls_targets*((1-batch_cls_predicts)**self.gamma)*batch_cls_predicts.log()
            cls_neg_loss=-(1-self.alpha)*(1-cls_targets)*(batch_cls_predicts**self.gamma)*(1-batch_cls_predicts).log()
            cls_loss=(cls_pos_loss+cls_neg_loss).sum()

            cls_loss_list.append(cls_loss)
            reg_loss_list.append(iou_loss)
            center_loss_list.append(centerness_loss)

        cls_loss_sum=torch.stack(cls_loss_list).sum()

        if num_pos==0:
            total_loss=cls_loss_sum/bs
            return total_loss,torch.stack([cls_loss_sum,
                                           torch.tensor(data=0.,device=device),
                                           torch.tensor(data=0.,device=device)]).detach(),num_pos

        cls_loss_sum=cls_loss_sum/num_pos
        reg_loss_sum = torch.stack(reg_loss_list).sum() / num_pos
        center_loss_sum = torch.stack(center_loss_list).sum() / num_pos
        return cls_loss_sum + reg_loss_sum + center_loss_sum, torch.stack([
                                                                        cls_loss_sum,
                                                                        reg_loss_sum,
                                                                        center_loss_sum]).detach(), num_pos




