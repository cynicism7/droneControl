"""
图像融合模块
实现多视角图像的融合处理，包括特征匹配、配准和融合
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple

class ImageFusion:
    """图像融合类"""
    
    def __init__(self):
        """初始化图像融合器"""
        # 使用SIFT特征检测器
        self.sift = cv2.SIFT_create()
        # FLANN匹配器参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    def detect_and_match_features(self, img1, img2, ratio_threshold=0.7):
        """
        检测并匹配两张图像的特征点
        :param img1: 第一张图像
        :param img2: 第二张图像
        :param ratio_threshold: Lowe's ratio test阈值（降低以提高匹配数量）
        :return: 匹配点对
        """
        # 转换为灰度图
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = img2
        
        # 增强对比度以提高特征检测
        gray1 = cv2.equalizeHist(gray1)
        gray2 = cv2.equalizeHist(gray2)
        
        # 检测关键点和描述符（增加特征点数量）
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None, None, None, None
        
        # 特征匹配
        try:
            matches = self.flann.knnMatch(des1, des2, k=2)
        except:
            # 如果FLANN失败，使用BFMatcher
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
        
        # 应用Lowe's ratio test筛选好的匹配（降低阈值）
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return None, None, None, None
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return src_pts, dst_pts, kp1, kp2
    
    def find_homography(self, img1, img2):
        """
        计算两张图像之间的单应性矩阵
        :param img1: 第一张图像
        :param img2: 第二张图像
        :return: 单应性矩阵和匹配质量
        """
        src_pts, dst_pts, kp1, kp2 = self.detect_and_match_features(img1, img2)
        
        if src_pts is None:
            return None, 0
        
        # 使用RANSAC计算单应性矩阵（增加容错度）
        H, mask = cv2.findHomography(src_pts, dst_pts, 
                                     cv2.RANSAC, 
                                     ransacReprojThreshold=10.0,
                                     maxIters=3000,
                                     confidence=0.99)
        
        if H is None:
            return None, 0
        
        # 验证单应性矩阵的合理性
        # 检查H矩阵的值是否在合理范围内
        if np.any(np.abs(H) > 1000) or np.any(np.isnan(H)) or np.any(np.isinf(H)):
            return None, 0
        
        # 检查变换后的角点是否在合理范围内
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # 检查变换后的坐标是否在合理范围内（不超过原图尺寸的10倍）
        max_expected_size = max(w1, h1, w2, h2) * 10
        if np.any(np.abs(transformed_corners) > max_expected_size):
            return None, 0
        
        # 计算匹配质量（inlier数量）
        inlier_count = np.sum(mask)
        total_matches = len(src_pts)
        quality = inlier_count / total_matches if total_matches > 0 else 0
        
        return H, quality
    
    def align_images(self, img1, img2, H):
        """
        使用单应性矩阵对齐图像
        :param img1: 参考图像
        :param img2: 待对齐图像
        :param H: 单应性矩阵
        :return: 对齐后的图像和变换后的图像尺寸，如果失败返回None
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 检查单应性矩阵的有效性
        if H is None or np.any(np.isnan(H)) or np.any(np.isinf(H)):
            return None
        
        # 获取图像2的四个角点
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners2_transformed = cv2.perspectiveTransform(corners2, H)
        
        # 检查变换后的角点是否合理（不应有异常大的值）
        corners_flat = corners2_transformed.ravel()
        if np.any(np.isnan(corners_flat)) or np.any(np.isinf(corners_flat)):
            return None
        
        # 预先检查变换后的角点范围是否合理
        max_expected_size = max(w1, h1, w2, h2) * 5  # 允许5倍的最大尺寸
        if np.any(np.abs(corners2_transformed) > max_expected_size):
            return None
        
        # 计算输出图像尺寸
        all_corners = np.concatenate([
            np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
            corners2_transformed
        ], axis=0)
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # 安全检查：限制输出图像的最大尺寸（防止内存溢出）
        MAX_OUTPUT_SIZE = 5000  # 降低到5000像素，更严格
        
        # 如果计算出的尺寸过大，说明单应性矩阵可能有问题
        output_width = x_max - x_min
        output_height = y_max - y_min
        
        if output_width > MAX_OUTPUT_SIZE or output_height > MAX_OUTPUT_SIZE:
            return None
        
        # 检查输出尺寸是否过小（小于原图尺寸的10%可能是计算错误）
        MIN_OUTPUT_SIZE = min(w1, h1, w2, h2) * 0.1
        if output_width < MIN_OUTPUT_SIZE or output_height < MIN_OUTPUT_SIZE:
            print(f"[警告] 对齐后图像尺寸过小 ({output_width}x{output_height})，可能是对齐错误")
            return None
        
        if output_width <= 0 or output_height <= 0:
            return None
        
        # 平移矩阵，使所有像素为正
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]],
                                 [0, 1, translation_dist[1]],
                                 [0, 0, 1]])
        
        # 应用平移
        H_final = H_translation @ H
        
        # 变换图像
        try:
            img2_warped = cv2.warpPerspective(img2, H_final, (output_width, output_height))
        except Exception as e:
            print(f"[错误] 图像变换失败: {e}")
            return None
        
        # 创建参考图像的变换版本
        img1_warped = np.zeros((output_height, output_width, 3), dtype=img1.dtype)
        
        # 确保索引在有效范围内
        y_start = max(0, translation_dist[1])
        y_end = min(output_height, translation_dist[1] + h1)
        x_start = max(0, translation_dist[0])
        x_end = min(output_width, translation_dist[0] + w1)
        
        img_y_start = max(0, -translation_dist[1])
        img_y_end = img_y_start + (y_end - y_start)
        img_x_start = max(0, -translation_dist[0])
        img_x_end = img_x_start + (x_end - x_start)
        
        img1_warped[y_start:y_end, x_start:x_end] = img1[img_y_start:img_y_end, img_x_start:img_x_end]
        
        return img1_warped, img2_warped, H_final
    
    def simple_blend(self, img1, img2, alpha=0.5):
        """
        简单加权融合
        :param img1: 第一张图像
        :param img2: 第二张图像
        :param alpha: 融合权重
        :return: 融合后的图像
        """
        # 确保图像尺寸相同
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 使用较大的尺寸作为目标尺寸（而不是较小的）
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        # 如果尺寸差异很大，调整到目标尺寸
        if abs(h1 - h2) > max(h1, h2) * 0.1 or abs(w1 - w2) > max(w1, w2) * 0.1:
            img1 = cv2.resize(img1, (target_w, target_h))
            img2 = cv2.resize(img2, (target_w, target_h))
        
        h = target_h
        w = target_w
        img1_cropped = img1[:h, :w]
        img2_cropped = img2[:h, :w]
        
        # 创建掩码（非零区域）
        mask1 = np.sum(img1_cropped, axis=2) > 0
        mask2 = np.sum(img2_cropped, axis=2) > 0
        
        # 融合
        result = img1_cropped.copy().astype(np.float32)
        result[mask2] = alpha * img1_cropped[mask2].astype(np.float32) + (1 - alpha) * img2_cropped[mask2].astype(np.float32)
        result[~mask1 & mask2] = img2_cropped[~mask1 & mask2].astype(np.float32)
        
        return result.astype(np.uint8)
    
    def pyramid_blend(self, img1, img2, levels=5):
        """
        多分辨率金字塔融合
        :param img1: 第一张图像
        :param img2: 第二张图像
        :param levels: 金字塔层数
        :return: 融合后的图像
        """
        # 确保图像尺寸相同
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 使用较大的尺寸作为目标尺寸（保持最大尺寸）
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        # 如果尺寸差异很大，调整到目标尺寸
        if abs(h1 - h2) > max(h1, h2) * 0.1 or abs(w1 - w2) > max(w1, w2) * 0.1:
            img1 = cv2.resize(img1, (target_w, target_h)).astype(np.float32)
            img2 = cv2.resize(img2, (target_w, target_h)).astype(np.float32)
        else:
            # 如果尺寸接近，使用较大的尺寸，并填充较小的图像
            if h1 < target_h or w1 < target_w:
                img1_new = np.zeros((target_h, target_w, 3), dtype=np.float32)
                img1_new[:h1, :w1] = img1.astype(np.float32)
                img1 = img1_new
            else:
                img1 = img1.astype(np.float32)
            
            if h2 < target_h or w2 < target_w:
                img2_new = np.zeros((target_h, target_w, 3), dtype=np.float32)
                img2_new[:h2, :w2] = img2.astype(np.float32)
                img2 = img2_new
            else:
                img2 = img2.astype(np.float32)
        
        h = target_h
        w = target_w
        
        # 检查图像是否为空
        if img1.size == 0 or img2.size == 0 or h < 4 or w < 4:
            # 如果图像太小或为空，使用简单融合
            return self.simple_blend(img1.astype(np.uint8), img2.astype(np.uint8))
        
        # 确保尺寸是偶数（pyrUp要求）且足够大
        if h % 2 == 1:
            img1 = img1[:-1, :]
            img2 = img2[:-1, :]
            h -= 1
        if w % 2 == 1:
            img1 = img1[:, :-1]
            img2 = img2[:, :-1]
            w -= 1
        
        # 再次检查尺寸
        if h < 4 or w < 4:
            return self.simple_blend(img1.astype(np.uint8), img2.astype(np.uint8))
        
        # 创建掩码
        mask1 = np.sum(img1, axis=2) > 0
        mask2 = np.sum(img2, axis=2) > 0
        
        # 生成高斯金字塔
        g1 = [img1.copy()]
        g2 = [img2.copy()]
        for i in range(levels - 1):
            # 检查当前图像尺寸是否足够大
            if g1[-1].shape[0] < 4 or g1[-1].shape[1] < 4:
                break
            down1 = cv2.pyrDown(g1[-1])
            down2 = cv2.pyrDown(g2[-1])
            # 检查下采样后的图像是否为空
            if down1.size == 0 or down2.size == 0:
                break
            g1.append(down1)
            g2.append(down2)
        
        # 如果金字塔层数不足，使用简单融合
        if len(g1) < 2:
            return self.simple_blend(img1.astype(np.uint8), img2.astype(np.uint8))
        
        # 调整实际使用的层数
        actual_levels = len(g1)
        
        # 生成拉普拉斯金字塔
        l1 = [g1[actual_levels - 1]]
        l2 = [g2[actual_levels - 1]]
        for i in range(actual_levels - 1, 0, -1):
            # 使用pyrUp默认行为，然后调整尺寸
            up1 = cv2.pyrUp(g1[i])
            up2 = cv2.pyrUp(g2[i])
            # 裁剪到正确尺寸
            target_h, target_w = g1[i - 1].shape[:2]
            if up1.shape[0] != target_h or up1.shape[1] != target_w:
                up1 = cv2.resize(up1, (target_w, target_h))
            if up2.shape[0] != target_h or up2.shape[1] != target_w:
                up2 = cv2.resize(up2, (target_w, target_h))
            l1.append(g1[i - 1] - up1)
            l2.append(g2[i - 1] - up2)
        
        # 融合拉普拉斯金字塔
        ls = []
        for i in range(actual_levels):
            # 在重叠区域使用加权平均
            blended = 0.5 * l1[actual_levels - 1 - i] + 0.5 * l2[actual_levels - 1 - i]
            ls.append(blended)
        
        # 重建图像
        result = ls[0]
        for i in range(1, actual_levels):
            # 使用pyrUp默认行为，然后调整尺寸
            up_result = cv2.pyrUp(result)
            target_h, target_w = ls[i].shape[:2]
            if up_result.shape[0] != target_h or up_result.shape[1] != target_w:
                up_result = cv2.resize(up_result, (target_w, target_h))
            result = up_result + ls[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def fuse_multiple_images(self, images: List[Dict], method='pyramid'):
        """
        融合多张图像
        :param images: 图像列表，每个元素包含'image'键
        :param method: 融合方法 ('simple' 或 'pyramid')
        :return: 融合后的图像
        """
        if len(images) == 0:
            return None
        
        if len(images) == 1:
            if 'image' in images[0] and images[0]['image'] is not None:
                return images[0]['image']
            else:
                return None
        
        # 使用第一张有效图像作为参考
        reference_img = None
        for img_data in images:
            if 'image' in img_data and img_data['image'] is not None:
                if hasattr(img_data['image'], 'shape') or isinstance(img_data['image'], np.ndarray):
                    reference_img = img_data['image']
                    break
        
        if reference_img is None:
            print("[错误] 没有找到有效的参考图像")
            return None
        
        aligned_images = [reference_img]
        
        print(f"[图像融合] 开始融合 {len(images)} 张图像...")
        print(f"[图像融合] 参考图像尺寸: {reference_img.shape[1]}x{reference_img.shape[0]}")
        
        successful_alignments = 0
        failed_alignments = 0
        
        # 对齐所有图像到参考图像
        for i in range(1, len(images)):
            # 检查图像数据是否存在
            if 'image' not in images[i]:
                print(f"\n[图像融合] 跳过图像 {i+1}/{len(images)} (缺少'image'键)")
                failed_alignments += 1
                continue
            
            img = images[i]['image']
            
            # 检查图像是否有效
            if img is None:
                print(f"\n[图像融合] 跳过图像 {i+1}/{len(images)} (图像为None)")
                failed_alignments += 1
                continue
            
            if not hasattr(img, 'shape') or len(img.shape) < 2:
                print(f"\n[图像融合] 跳过图像 {i+1}/{len(images)} (无效的图像格式)")
                failed_alignments += 1
                continue
            
            print(f"\n[图像融合] 处理图像 {i+1}/{len(images)} (尺寸: {img.shape[1]}x{img.shape[0]})...")
            
            H, quality = self.find_homography(reference_img, img)
            
            if H is not None and quality > 0.15:  # 降低匹配质量阈值（从0.3降到0.15）
                align_result = self.align_images(reference_img, img, H)
                if align_result is not None:
                    img1_aligned, img2_aligned, _ = align_result
                    aligned_images.append(img2_aligned)
                    successful_alignments += 1
                    print(f"  ✓ 对齐成功，匹配质量: {quality:.2f}, 对齐后尺寸: {img2_aligned.shape[1]}x{img2_aligned.shape[0]}")
                else:
                    # 对齐过程失败（可能是尺寸问题），使用原图
                    aligned_images.append(img)
                    failed_alignments += 1
                    print(f"  ✗ 对齐失败（尺寸异常），使用原图")
            else:
                # 如果无法对齐，直接使用原图
                aligned_images.append(img)
                failed_alignments += 1
                quality_str = f"{quality:.2f}" if H is not None and isinstance(quality, (int, float)) else "0.00"
                print(f"  ✗ 对齐失败（匹配质量: {quality_str} < 0.15），使用原图")
        
        print(f"\n[图像融合] 对齐统计: 成功 {successful_alignments} 张, 失败 {failed_alignments} 张")
        
        # 逐步融合所有对齐后的图像
        print(f"\n[图像融合] 开始逐步融合 {len(aligned_images)} 张图像...")
        
        # 过滤掉异常小的图像（小于原始图像的10%）
        MIN_IMAGE_SIZE = 64  # 最小图像尺寸阈值
        filtered_images = []
        ref_size = reference_img.shape[0] * reference_img.shape[1]
        
        for idx, img in enumerate(aligned_images):
            if img is None or img.size == 0:
                print(f"  [跳过] 图像 {idx+1} 为空，跳过融合")
                continue
            
            if len(img.shape) < 2:
                print(f"  [跳过] 图像 {idx+1} 格式无效，跳过融合")
                continue
            
            h, w = img.shape[:2]
            if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
                print(f"  [跳过] 图像 {idx+1} 尺寸过小 ({w}x{h} < {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE})，跳过融合")
                continue
            
            # 检查图像尺寸是否异常小（小于原始图像的5%）
            img_size = h * w
            if img_size < ref_size * 0.05:
                print(f"  [跳过] 图像 {idx+1} 尺寸异常小 ({w}x{h}, 仅原图{img_size/ref_size*100:.1f}%)，跳过融合")
                continue
            
            filtered_images.append(img)
        
        if len(filtered_images) == 0:
            print("[错误] 没有有效的图像可以融合")
            return None
        
        print(f"[图像融合] 经过过滤，有效图像数量: {len(filtered_images)}/{len(aligned_images)}")
        result = filtered_images[0]
        print(f"  初始图像尺寸: {result.shape[1]}x{result.shape[0]}")
        
        for i in range(1, len(filtered_images)):
            try:
                prev_size = result.shape
                if method == 'pyramid':
                    result = self.pyramid_blend(result, filtered_images[i])
                else:
                    result = self.simple_blend(result, filtered_images[i])
                
                # 检查融合结果是否有效
                if result is None or result.size == 0:
                    print(f"  [警告] 图像 {i+1} 融合结果为空，跳过")
                    continue
                
                # 检查融合后的尺寸是否异常变小
                if result.shape[0] < MIN_IMAGE_SIZE or result.shape[1] < MIN_IMAGE_SIZE:
                    print(f"  [警告] 融合后尺寸异常小 ({result.shape[1]}x{result.shape[0]})，停止融合")
                    result = filtered_images[0]  # 恢复为第一张图像
                    break
                    
                print(f"  [{i+1}/{len(filtered_images)}] 融合完成，当前尺寸: {result.shape[1]}x{result.shape[0]}")
            except Exception as e:
                print(f"  [警告] 融合第 {i+1} 张图像时出错: {e}，尝试简单融合")
                # 如果金字塔融合失败，使用简单融合
                try:
                    if filtered_images[i] is not None and filtered_images[i].size > 0:
                        result = self.simple_blend(result, filtered_images[i])
                        if result is not None and result.size > 0:
                            if result.shape[0] >= MIN_IMAGE_SIZE and result.shape[1] >= MIN_IMAGE_SIZE:
                                print(f"  [{i+1}/{len(filtered_images)}] 简单融合完成")
                            else:
                                print(f"  [警告] 简单融合后尺寸异常小，跳过该图像")
                        else:
                            print(f"  [跳过] 图像 {i+1} 简单融合结果无效")
                    else:
                        print(f"  [跳过] 图像 {i+1} 无效，无法融合")
                except Exception as e2:
                    print(f"  [警告] 简单融合也失败: {e2}，跳过该图像")
        
        if result is None or result.size == 0:
            print("[错误] 融合结果无效")
            return None
        
        # 最终检查：确保结果尺寸合理
        if result.shape[0] < MIN_IMAGE_SIZE or result.shape[1] < MIN_IMAGE_SIZE:
            print(f"[警告] 最终融合图像尺寸过小 ({result.shape[1]}x{result.shape[0]})，使用第一张有效图像")
            return filtered_images[0] if len(filtered_images) > 0 else reference_img
        
        print(f"[图像融合] 最终融合图像尺寸: {result.shape[1]}x{result.shape[0]}")
        return result
