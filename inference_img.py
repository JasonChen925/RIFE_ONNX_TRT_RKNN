import torch
import time
import os
import cv2
import psutil  # 用于监控 CPU 内存占用
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
from model.RIFE_HDv3 import Model
model = Model()
model.load_model('train_log', -1)
model.eval()
model.device()

# 读取图像
img_0 = r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/im1.png"
img_1 = r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/im3.png"
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

def get_memory_usage():
    """ 获取当前进程的 CPU 内存占用（单位 MB） """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2  # 转换为 MB

# 监控初始 GPU 和 CPU 内存占用
print("初始显存占用: {:.2f} MB, 初始 CPU 内存占用: {:.2f} MB".format(
    torch.cuda.memory_allocated() / 1024 ** 2, get_memory_usage()))

Time_Start = time.time()
for i in range(exp):
    tmp = []
    for j in range(len(img_list) - 1):
        # 监控 inference 前的 GPU 和 CPU 内存占用
        print("Inference 前 - GPU: {:.2f} MB, 预留显存: {:.2f} MB, CPU 内存: {:.2f} MB".format(
            torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.memory_reserved() / 1024 ** 2,
            get_memory_usage()))

        mid = model.inference(img_list[j], img_list[j + 1])

        # 监控 inference 后的 GPU 和 CPU 内存占用
        print("Inference 后 - GPU: {:.2f} MB, 预留显存: {:.2f} MB, CPU 内存: {:.2f} MB".format(
            torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.memory_reserved() / 1024 ** 2,
            get_memory_usage()))

        tmp.append(img_list[j])
        tmp.append(mid)
    tmp.append(img1)
    img_list = tmp

Time_End = time.time()
print("插帧用时为: {:.4f} 秒".format(Time_End - Time_Start))

# 监控最终 GPU 和 CPU 内存占用
print("最终显存占用: {:.2f} MB, 最终 CPU 内存占用: {:.2f} MB".format(
    torch.cuda.memory_allocated() / 1024 ** 2, get_memory_usage()))

if not os.path.exists('output'):
    os.mkdir('output')

for i in range(len(img_list)):
    cv2.imwrite('output/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
