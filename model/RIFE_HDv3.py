import onnxruntime
import torch
import sys
sys.path.append("..")
import os
from torch.optim import AdamW
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from .IFNet_HDv3 import *
from .loss import *
from Using_TensorRT_Engine import TRTWrapper
# from ..Using_TensorRT_Engine import TRTWrapper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        # if local_rank != -1:
        #     self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

        # self.ort_session = onnxruntime.InferenceSession('/ECCV2022-RIFE/train_log/IFNet.onnx',
        #                                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                                                                                                                        ## 以下为2560x1440分辨率
        # self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/model_fp16.trt', None)    ## 使用fp16的trt engine
        # self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/model_fp32.engine',None)              ##使用fp32的trt engine
        # self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/model_fp16_int8.trt', None) ## 使用fp16和int8混合精度
                                                                                                                        ##
                                                                                                                        ##以下为256x256的分辨率
        self.tensorrt = TRTWrapper(r'/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/model_256x256_fp32.engine', None)
    # def initialize_session(self):
    #     session_options = onnxruntime.SessionOptions()
    #     session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    #     ort_session = onnxruntime.InferenceSession('/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/IFNet.onnx',
    #                                                 session_options=session_options,
    #                                                 providers=["CUDAExecutionProvider"])
    #     return ort_session
    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
            else:
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location ='cpu')))
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))
########################################################原版
    # def inference(self, img0, img1, scale=1.0):  #原版
    #     imgs = torch.cat((img0, img1), 1)
    #     flow, mask, merged = self.flownet(imgs)
    #     return merged[2]

    ###################################################### ONNX版，效果并不理想
    # def inference(self,img0,img1,scale=1.0,):
    #     imgs = torch.cat((img0, img1), 1).cpu().numpy() #将输入张量img0和img1拼接为一个新的张量
    #     input_name = self.ort_session.get_inputs()[0].name
    #     ort_inputs = {input_name:imgs}
    #     ort_outputs = self.ort_session.run(None,ort_inputs) #执行推理
    #     # print("执行推理完毕")
    #     output_names = [output.name for output in self.ort_session.get_outputs()]
    #     # for idx, (name, output) in enumerate(zip(output_names, ort_outputs)):
    #     #     print(f"Output {idx} ({name}): shape = {output.shape}")
    #     #     print(f"Output {idx} sample values: {output.flatten()[:10]}")  # 打印前10个值
    #     # profile_result = self.ort_session.end_profiling()
    #     flow = ort_outputs[:3]  # 取 ort_outputs 的第 0, 1, 2 元素，赋值给 flow
    #     mask = ort_outputs[3]  # 取 ort_outputs 的第 3 个元素，赋值给 mask
    #     merged = ort_outputs[4:7]  # 取 ort_outputs 的第 4, 5, 6 元素，赋值给 merged
    #     #
    #     # print(f"flow_list shape:{np.array(flow).shape}")
    #     # print(f"mask shape:{np.array(mask).shape}")
    #     # print(f"merged shape: {np.array(merged).shape}")
    #     return merged[2]
######################################Tensorrt加速
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
        scale = [4, 2, 1]
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[2] - gt).abs().mean()
        loss_smooth = self.sobel(flow[2], flow[2]*0).mean()
        # loss_vgg = self.vgg(merged[2], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_cons + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[2], {
            'mask': mask,
            'flow': flow[2][:, :2],
            'loss_l1': loss_l1,
            'loss_cons': loss_cons,
            'loss_smooth': loss_smooth,
            }

