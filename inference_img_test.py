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
    save_path = os.path.join(output_folder, f"img_output{i}.png")
    cv2.imwrite(save_path, (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

    #####计算PSNR和SSIM

# 读取 Ground Truth 中间帧
gt_path = os.path.join(output_folder, "im2_origin.png")
gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

# 读取生成的插帧中间帧
pred_path = os.path.join(output_folder, "img_output1.png")  # 注意改成 output1
pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)

# 转为 float32
gt = gt.astype(np.float32) / 255.
pred = pred.astype(np.float32) / 255.

# 对齐尺寸（如果尺寸有微小不同，裁剪处理）
h_, w_, _ = gt.shape
pred = pred[:h_, :w_, :]

# 计算 PSNR
mse = np.mean((gt - pred) ** 2)
if mse == 0:
    psnr = 100
else:
    psnr = 10 * np.log10(1.0 / mse)

# 计算 SSIM
import skimage.metrics

ssim = skimage.metrics.structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)


print(f"插帧结果与 im2_origin.png 对比：PSNR={psnr:.4f} dB, SSIM={ssim:.4f}")
