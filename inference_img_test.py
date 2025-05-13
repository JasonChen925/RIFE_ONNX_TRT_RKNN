import numpy as np
import torch
import time
import os
import cv2
import psutil  # 用于监控 CPU 内存占用
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
from model.RIFE import Model        #使用原版
model = Model()
model.load_model('train_log_origin',-1)  #####在此处修改模型权重文件
print("Loaded RIFE_origin model.")
model.eval()
model.device()

# 读取图像
img_0 = r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/test/images/im1.png"
img_1 = r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/test/images/im3.png"
img0 = cv2.imread(img_0, cv2.IMREAD_UNCHANGED)
img1 = cv2.imread(img_1, cv2.IMREAD_UNCHANGED)

img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

# 处理输入尺寸，让图像符合32的倍数
n, c, h, w = img0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)

# 插帧
exp = 1
img_list = [img0, img1]

Time_Start = time.time()
for i in range(exp):
    tmp = []
    for j in range(len(img_list) - 1):
        mid = model.inference(img_list[j], img_list[j + 1])
        tmp.append(img_list[j])
        tmp.append(mid)
    tmp.append(img1)
    img_list = tmp

Time_End = time.time()
print("插帧用时为: {:.4f} 秒".format(Time_End - Time_Start))

# if not os.path.exists('output'):
#     os.mkdir('output')
#
# for i in range(len(img_list)):
#     cv2.imwrite('output/img_{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

output_folder = r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/test/images/"
for i in range(len(img_list)):
    save_path = os.path.join(output_folder, f"img_output.png")
    cv2.imwrite(save_path, (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

