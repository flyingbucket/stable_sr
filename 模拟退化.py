import numpy as np
import cv2
import os
from PIL import Image
from scipy.ndimage import convolve
from bm3d import bm3d, BM3DProfile, BM3DStages  # pip install bm3d

def simulate_sar_degradation(image_path, degradation_case=1, scale_factor=4):
    """
    模拟SAR图像退化过程
    Args:
        image_path: 输入图像路径
        degradation_case: 退化情况 (1-4)
        scale_factor: 下采样因子 (默认4，生成1/4分辨率)
    """
    # 读取图像
    img = Image.open(image_path).convert("L")
    original_size = img.size
    
    # 先将图像调整到能被scale_factor整除的尺寸
    new_width = (img.width // scale_factor) * scale_factor
    new_height = (img.height // scale_factor) * scale_factor
    img = img.resize((new_width, new_height), Image.BICUBIC)
    
    # 保存高分辨率原图
    hr_img = np.array(img).astype(np.float32) / 255.0
    
    # 1. 先添加轻微的退化效果（模糊和噪声）
    degraded = hr_img.copy()
    
    if degradation_case == 1:
        # 情况1：轻微高斯模糊 + 弱乘性噪声
        # 高斯模糊（核大小和sigma都减小）
        degraded = cv2.GaussianBlur(degraded, (3, 3), 0.5)
        # 弱乘性噪声（SAR特有的斑点噪声）
        speckle = np.random.gamma(shape=1/0.005, scale=0.005, size=degraded.shape)  # 减弱噪声
        degraded = degraded * speckle
        
    elif degradation_case == 2:
        # 情况2：轻微运动模糊 + 弱乘性噪声
        # 运动模糊（长度减小）
        kernel = motion_kernel(length=2.0, angle=30)
        degraded = convolve(degraded, kernel, mode='reflect')
        # 弱乘性噪声
        speckle = np.random.gamma(shape=1/0.005, scale=0.005, size=degraded.shape)
        degraded = degraded * speckle
        
    elif degradation_case == 3:
        # 情况3：弱乘性噪声 + 轻微高斯模糊 + 极弱加性噪声
        # 先加乘性噪声
        speckle = np.random.gamma(shape=1/0.003, scale=0.003, size=degraded.shape)
        degraded = degraded * speckle
        # 轻微模糊
        degraded = cv2.GaussianBlur(degraded, (3, 3), 0.3)
        # 极弱加性噪声
        noise = np.random.normal(0, 0.01, degraded.shape)  # 大幅减小
        degraded = degraded + noise
        
    elif degradation_case == 4:
        # 情况4：混合退化（最轻微）
        # 极轻微模糊
        degraded = cv2.GaussianBlur(degraded, (3, 3), 0.2)
        # 极弱乘性噪声
        speckle = np.random.gamma(shape=1/0.002, scale=0.002, size=degraded.shape)
        degraded = degraded * speckle
        
    else:
        raise ValueError("退化编号应为 1 ~ 4")
    
    # 确保值在[0,1]范围内
    degraded = np.clip(degraded, 0, 1)
    
    # 2. 下采样到1/4分辨率
    lr_size = (new_width // scale_factor, new_height // scale_factor)
    lr_degraded = cv2.resize(degraded, lr_size, interpolation=cv2.INTER_CUBIC)
    
    # 3. 对低分辨率图像进行轻微去噪
    # 使用较小的sigma值进行去噪，保留更多细节
    lr_denoised = apply_bm3d_denoising(lr_degraded, sigma=0.03)  # 大幅减小去噪强度
    
    return hr_img, degraded, lr_degraded, lr_denoised, scale_factor

def motion_kernel(length, angle):
    """生成运动模糊核"""
    kernel_size = int(length * 2 + 1)
    kernel = np.zeros((kernel_size, kernel_size))
    angle_rad = np.deg2rad(angle)
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    
    for t in np.linspace(-length/2, length/2, int(length*3)):
        x = int(center[0] + t * dx)
        y = int(center[1] + t * dy)
        if 0 <= x < kernel.shape[0] and 0 <= y < kernel.shape[1]:
            kernel[x, y] = 1
    
    kernel /= kernel.sum()
    return kernel

def apply_bm3d_denoising(img, sigma=0.03):
    """应用BM3D去噪，使用更保守的参数"""
    # 使用HARD_THRESHOLDING以保留更多细节
    denoised = bm3d(img, sigma_psd=sigma, stage_arg=BM3DStages.HARD_THRESHOLDING)
    return np.clip(denoised, 0, 1)

def calculate_psnr(img1, img2):
    """计算PSNR值"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

if __name__ == "__main__":
    input_image = r"C:\Users\thoma\Desktop\测试\NH49E001013.tif_0_0.png"
    output_dir = "output_sar_degradation"
    os.makedirs(output_dir, exist_ok=True)
    
    print("图像路径：", input_image)
    print("输出路径：", os.path.abspath(output_dir))
    print("-" * 50)
    
    # 处理每种退化情况
    for case in range(1, 5):
        print(f"\n处理退化情况 {case}...")
        
        # 生成退化图像
        hr_img, degraded_hr, lr_degraded, lr_denoised, scale_factor = simulate_sar_degradation(
            input_image, degradation_case=case, scale_factor=4
        )
        
        # 计算PSNR值以量化退化程度
        psnr_degraded = calculate_psnr(hr_img, degraded_hr)
        print(f"  HR原图 vs HR退化图 PSNR: {psnr_degraded:.2f} dB")
        
        # 保存所有图像
        # 1. 保存高分辨率原图（只在第一次保存）
        if case == 1:
            hr_uint8 = (hr_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, "hr_original.tif"), hr_uint8)
        
        # 2. 保存高分辨率退化图
        degraded_hr_uint8 = (degraded_hr * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"hr_degraded_case{case}.tif"), degraded_hr_uint8)
        
        # 3. 保存低分辨率退化图（下采样后）
        lr_degraded_uint8 = (lr_degraded * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"lr_degraded_case{case}.tif"), lr_degraded_uint8)
        
        # 4. 保存低分辨率去噪图
        lr_denoised_uint8 = (lr_denoised * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"lr_denoised_case{case}.tif"), lr_denoised_uint8)
        
        print(f"  已保存所有图像")
        print(f"  LR尺寸: {lr_degraded.shape}")
        
    print("\n" + "-" * 50)
    print("处理完成！")
    print("\n生成的文件说明：")
    print("- hr_original.tif: 高分辨率原图")
    print("- hr_degraded_caseX.tif: 高分辨率退化图")
    print("- lr_degraded_caseX.tif: 低分辨率退化图（1/4分辨率）")
    print("- lr_denoised_caseX.tif: 低分辨率去噪图（用于超分辨率输入）")
    
    # 额外建议
    print("\n建议：")
    print("1. lr_denoised_caseX.tif 可作为StableSR的输入")
    print("2. hr_original.tif 可作为Ground Truth进行评估")
    print("3. 如果退化效果仍然过强，可以进一步减小噪声参数")