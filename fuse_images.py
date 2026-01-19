"""
独立的图像融合程序
使用拉普拉斯金字塔算法：
1. 先将同一架飞机不同时刻拍到的画面进行融合
2. 再将三架飞机拍到的画面进行拼接
3. 最终保证是像素级的呈现，像是一个摄像机运动30m拍摄的长镜头
"""

import cv2
import numpy as np
import os
from typing import List, Dict
from control.image_fusion import ImageFusion

def load_images_from_directories():
    """
    从图像目录加载所有图像
    :return: {drone_name: [image_info_list]}
    """
    images_by_drone = {}
    
    for drone_name in ["Drone1", "Drone2", "Drone3"]:
        image_dir = f"images_{drone_name}"
        if not os.path.exists(image_dir):
            print(f"[警告] 目录不存在: {image_dir}")
            continue
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        images = []
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 从文件名提取索引（用于排序）
                try:
                    idx = int(img_file.split('_')[1].split('.')[0])
                except:
                    idx = len(images)
                
                images.append({
                    'image': img_rgb,
                    'filename': img_path,
                    'drone_name': drone_name,
                    'index': idx
                })
        
        if images:
            # 按索引排序
            images.sort(key=lambda x: x['index'])
            images_by_drone[drone_name] = images
            print(f"[加载] {drone_name}: {len(images)} 张图像")
    
    return images_by_drone

def fuse_single_drone_images(fusion_processor, images: List[Dict]) -> np.ndarray:
    """
    融合同一架无人机不同时刻拍摄的图像（使用拉普拉斯金字塔）
    :param fusion_processor: 图像融合处理器
    :param images: 图像列表
    :return: 融合后的图像
    """
    if len(images) == 0:
        return None
    
    if len(images) == 1:
        return images[0]['image']
    
    print(f"[单机融合] 开始融合 {len(images)} 张图像...")
    
    # 逐步融合所有图像
    result = images[0]['image'].copy()
    
    for i in range(1, len(images)):
        img = images[i]['image']
        print(f"  [单机融合] 融合第 {i+1}/{len(images)} 张图像...")
        
        # 使用拉普拉斯金字塔融合
        result = fusion_processor.pyramid_blend(result, img, levels=5)
        
        if result is None:
            print(f"  [警告] 第 {i+1} 张图像融合失败，使用简单融合")
            result = fusion_processor.simple_blend(result, img)
    
    print(f"[单机融合] 完成，最终尺寸: {result.shape[1]}x{result.shape[0]}")
    return result

def horizontal_stitch_images(fusion_processor, drone_images: Dict[str, np.ndarray]) -> np.ndarray:
    """
    水平拼接三架无人机拍摄的图像（长镜头效果）
    由于无人机沿直线移动，图像应该水平拼接
    :param fusion_processor: 图像融合处理器
    :param drone_images: {drone_name: fused_image}
    :return: 最终融合图像
    """
    if len(drone_images) == 0:
        return None
    
    if len(drone_images) == 1:
        return list(drone_images.values())[0]
    
    print(f"\n[多机融合] 开始水平拼接 {len(drone_images)} 架无人机的图像...")
    
    # 按无人机顺序排列（Drone1, Drone2, Drone3）
    drone_order = ["Drone1", "Drone2", "Drone3"]
    ordered_images = []
    
    for drone_name in drone_order:
        if drone_name in drone_images:
            ordered_images.append(drone_images[drone_name])
            print(f"  [多机融合] {drone_name} 图像尺寸: {drone_images[drone_name].shape[1]}x{drone_images[drone_name].shape[0]}")
    
    if len(ordered_images) == 0:
        return None
    
    # 获取所有图像的高度（应该相同）
    heights = [img.shape[0] for img in ordered_images]
    target_height = max(heights)
    
    # 统一所有图像的高度
    resized_images = []
    for img in ordered_images:
        if img.shape[0] != target_height:
            # 调整高度
            scale = target_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            img_resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_images.append(img_resized)
        else:
            resized_images.append(img)
    
    # 计算总宽度（考虑重叠区域）
    total_width = sum(img.shape[1] for img in resized_images)
    # 由于有重叠，实际宽度会小于总和，但先创建足够大的画布
    panorama = np.zeros((target_height, total_width, 3), dtype=np.float32)
    weight_map = np.zeros((target_height, total_width), dtype=np.float32)
    
    current_x = 0
    overlap_width = int(resized_images[0].shape[1] * 0.25)  # 25%重叠
    
    for i, img in enumerate(resized_images):
        img_float = img.astype(np.float32)
        h, w = img.shape[:2]
        
        # 计算放置位置
        if i == 0:
            start_x = 0
        else:
            # 考虑重叠，从重叠区域开始
            start_x = current_x - overlap_width
        
        end_x = start_x + w
        
        # 像素级融合
        for x_img in range(w):
            x_pano = start_x + x_img
            
            if x_pano < 0 or x_pano >= total_width:
                continue
            
            # 计算权重（在重叠区域使用渐变）
            if i == 0:
                pixel_weight = 1.0
            else:
                dist_from_overlap_start = x_pano - (current_x - overlap_width)
                if dist_from_overlap_start < 0:
                    pixel_weight = 1.0
                elif dist_from_overlap_start < overlap_width:
                    pixel_weight = dist_from_overlap_start / overlap_width
                else:
                    pixel_weight = 1.0
            
            # 像素级加权融合
            old_weight = weight_map[:, x_pano]
            new_weight = pixel_weight
            
            if old_weight.max() > 0.01:
                total_weight = old_weight + new_weight
                panorama[:, x_pano] = (panorama[:, x_pano] * old_weight[:, np.newaxis] + 
                                      img_float[:, x_img] * new_weight) / total_weight[:, np.newaxis]
                weight_map[:, x_pano] = total_weight
            else:
                panorama[:, x_pano] = img_float[:, x_img]
                weight_map[:, x_pano] = new_weight
        
        current_x = end_x
        print(f"  [多机融合] 已拼接 {i+1}/{len(resized_images)} 张图像")
    
    # 裁剪掉全黑区域
    non_zero_cols = np.where(weight_map.sum(axis=0) > 0)[0]
    if len(non_zero_cols) > 0:
        left_col = non_zero_cols[0]
        right_col = non_zero_cols[-1] + 1
        panorama = panorama[:, left_col:right_col]
    
    result = np.clip(panorama, 0, 255).astype(np.uint8)
    print(f"[多机融合] 完成，最终尺寸: {result.shape[1]}x{result.shape[0]}")
    return result

def main():
    print("=" * 60)
    print("图像融合程序")
    print("=" * 60)
    
    # 1. 加载所有图像
    print("\n[步骤1] 加载图像...")
    images_by_drone = load_images_from_directories()
    
    if not images_by_drone:
        print("[错误] 没有找到图像文件")
        return
    
    total_images = sum(len(imgs) for imgs in images_by_drone.values())
    print(f"[加载] 总共加载 {total_images} 张图像")
    
    # 2. 初始化融合处理器
    print("\n[步骤2] 初始化图像融合处理器...")
    fusion_processor = ImageFusion()
    
    # 3. 对每架无人机的图像进行融合（拉普拉斯金字塔）
    print("\n[步骤3] 融合每架无人机的图像（拉普拉斯金字塔）...")
    fused_drone_images = {}
    
    for drone_name, images in images_by_drone.items():
        print(f"\n[处理] {drone_name}...")
        fused_image = fuse_single_drone_images(fusion_processor, images)
        if fused_image is not None:
            fused_drone_images[drone_name] = fused_image
            print(f"[完成] {drone_name} 融合完成，尺寸: {fused_image.shape[1]}x{fused_image.shape[0]}")
        else:
            print(f"[错误] {drone_name} 融合失败")
    
    if not fused_drone_images:
        print("[错误] 没有成功融合的图像")
        return
    
    # 4. 融合三架无人机的图像（水平拼接成长镜头）
    print("\n[步骤4] 融合三架无人机的图像（水平拼接成长镜头）...")
    final_fused_image = horizontal_stitch_images(fusion_processor, fused_drone_images)
    
    if final_fused_image is None:
        print("[错误] 最终融合失败")
        return
    
    # 5. 保存结果
    print("\n[步骤5] 保存融合结果...")
    output_dir = "fusion_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "fused_image_final.png")
    cv2.imwrite(output_path, cv2.cvtColor(final_fused_image, cv2.COLOR_RGB2BGR))
    
    print("\n" + "=" * 60)
    print("图像融合完成！")
    print("=" * 60)
    print(f"融合结果已保存到: {output_path}")
    print(f"最终图像尺寸: {final_fused_image.shape[1]} x {final_fused_image.shape[0]} 像素")
    print(f"融合了 {len(fused_drone_images)} 架无人机的图像")
    print(f"总共处理了 {total_images} 张原始图像")
    print(f"文件位置: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
