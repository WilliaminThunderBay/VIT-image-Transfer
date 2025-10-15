from PIL import Image  # 确保正确导入 Image 模块
import torch
import time
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score
import os

# GPU设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练VGG19模型生成的输出图像
def load_generated_image_vgg19(output_path):
    image = Image.open(output_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

# 加载预训练ViT模型生成的输出图像
def load_generated_image_vit(output_path):
    image = Image.open(output_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

# 加载真实内容图像
def load_content_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

# SSIM计算（修复形状不匹配问题）
def calculate_ssim(image1, image2):
    # 确保图像形状一致
    _, _, height1, width1 = image1.shape
    _, _, height2, width2 = image2.shape
    
    # 如果尺寸不同，将 image2 调整为与 image1 相同的分辨率
    if height1 != height2 or width1 != width2:
        image2 = torch.nn.functional.interpolate(image2, size=(height1, width1), mode='bilinear', align_corners=False)
    
    # 转换为 NumPy 数组
    image1_np = image1.cpu().squeeze().numpy().transpose(1, 2, 0)
    image2_np = image2.cpu().squeeze().numpy().transpose(1, 2, 0)
    
    return ssim(image1_np, image2_np, multichannel=True)

# PSNR计算
def calculate_psnr(image1, image2):
    # 确保图像形状一致
    _, _, height1, width1 = image1.shape
    _, _, height2, width2 = image2.shape
    
    # 如果尺寸不同，将 image2 调整为与 image1 相同的分辨率
    if height1 != height2 or width1 != width2:
        image2 = torch.nn.functional.interpolate(image2, size=(height1, width1), mode='bilinear', align_corners=False)
    
    # 转换为 NumPy 数组
    image1_np = image1.cpu().squeeze().numpy().transpose(1, 2, 0)
    image2_np = image2.cpu().squeeze().numpy().transpose(1, 2, 0)
    
    return psnr(image1_np, image2_np)

# 运行FID评估
def calculate_fid(real_dir, fake_dir):
    return fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device=str(device))

# 模型评估
def evaluate_model(model_name, content_img_path, generated_img_path, real_dir, fake_dir):
    content_img = load_content_image(content_img_path)
    
    if model_name == "VGG19":
        generated_img = load_generated_image_vgg19(generated_img_path)
    elif model_name == "ViT":
        generated_img = load_generated_image_vit(generated_img_path)
    
    # 计算SSIM和PSNR
    start_time = time.time()
    ssim_value = calculate_ssim(content_img, generated_img)
    psnr_value = calculate_psnr(content_img, generated_img)
    end_time = time.time()
    
    # 计算FID
    fid_value = calculate_fid(real_dir, fake_dir)
    
    # 记录GPU资源消耗
    gpu_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)  # 转为MB

    return {
        "Model": model_name,
        "SSIM": ssim_value,
        "PSNR": psnr_value,
        "FID": fid_value,
        "Time (s)": end_time - start_time,
        "GPU Memory (MB)": gpu_memory
    }

# 文件路径
content_image_path = "images/real/content.jpg"
vgg19_generated_path = "vgg19.jpg"
vit_generated_path = "vit.jpg"
real_image_dir = "images/real"
fake_image_dir_vgg19 = "images/fake_vgg19"
fake_image_dir_vit = "images/fake_vit"

# 运行评估
vgg19_results = evaluate_model("VGG19", content_image_path, vgg19_generated_path, real_image_dir, fake_image_dir_vgg19)
vit_results = evaluate_model("ViT", content_image_path, vit_generated_path, real_image_dir, fake_image_dir_vit)

# 打印结果
print("VGG19 Evaluation:", vgg19_results)
print("ViT Evaluation:", vit_results)
