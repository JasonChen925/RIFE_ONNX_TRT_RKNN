import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
# from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, local_rank=-1, arbitrary=False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
                                                                                                                        ## 以下为2560x1440分辨率
        # self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log_origin/model_1440x2560_fp16.trt', None)    ## 使用fp16的trt engine
        # self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log_origin/model_1440x2560_fp32.trt',None)              ##使用fp32的trt engine
        # self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log_HDv3/model_fp16_int8.trt', None) ## 使用fp16和int8混合精度
                                                                                                                        ##
                                                                                                                        ##以下为256x256的分辨率
        # self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log_HDv3/model_256x448_fp32.trt', None) ## 使用fp32的trt engine
        # self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log_HDv3/model_256x448_fp16.trt', None)  ## 使用fp16的trt engine
        # self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log_HDv3/model_256x448_TrtInt8.trt', None)  ## 使用int8的trt engine
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                # if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet_unetchange.pkl'.format(path))))
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet_unetchange.pkl'.format(path))

    # def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5): #训练的时候使用
    #     for i in range(3):
    #         scale_list[i] = scale_list[i] * 1.0 / scale
    #     imgs = torch.cat((img0, img1), 1)
    #     flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
    #     if TTA == False:###TTA属于是使用输入图像的水平翻转 + 垂直翻转（即上下 + 左右翻转）进行第二次推理：
    #         return merged[2]
    #     else:
    #         flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
    #         return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    # def inference(self,img0,img1,scale=1,scale_list=[4,2,1],timestep=0.5):  ##推理的时候使用
    #     imgs = torch.cat((img0, img1), 1)
    #     flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
    #     return merged[2]
    def inference(self,img0,img1,scale=1.0):
        imgs = torch.cat((img0,img1),1)
        output = self.tensorrt(dict(imgs=imgs.cuda()))
        # flow  = output['flow_list_0','flow_list_1',"flow_list_2"]
        # mask = output['mask_list_2']
        merged = output['merged_2']
        return merged
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01 # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            }
