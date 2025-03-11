import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import glob

# 指定图片所在的目录
image_dir = "/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/output/TrtEngine"

# 获取所有图片文件
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))  # 仅获取 PNG 文件
if len(image_paths) < 5:
    raise ValueError("文件夹中图片数量不足 5 张，请检查路径！")

# 指定基准图片（确保它在文件夹内）
ref_img_path = os.path.join(image_dir, "img1_origin.png")
if not os.path.exists(ref_img_path):
    raise FileNotFoundError(f"基准图片 {ref_img_path} 不存在，请检查文件夹内容！")

# 读取基准图片
ref_img = cv2.imread(ref_img_path, cv2.IMREAD_COLOR)
if ref_img is None:
    raise ValueError(f"基准图片 {ref_img_path} 读取失败！")

# 计算 PSNR 和 SSIM
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同，PSNR 无穷大
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

results = []
for img_path in image_paths:
    img_name = os.path.basename(img_path)  # 获取文件名
    if img_name == "img1_origin.png":
        continue  # 跳过基准图像

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"图片 {img_path} 读取失败，跳过...")
        continue

    # 计算 PSNR
    psnr_value = calculate_psnr(ref_img, img)

    # 计算 SSIM（需转换为灰度图）
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(ref_gray, img_gray)

    results.append((img_name, psnr_value, ssim_value))

# 打印结果
print("\nPSNR 和 SSIM 计算结果（基准: img1_origin.png）:")
for img_name, psnr_value, ssim_value in results:
    print(f"图片: {img_name}\n  PSNR: {psnr_value:.2f} dB\n  SSIM: {ssim_value:.4f}\n")
