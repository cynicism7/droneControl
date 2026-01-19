"""
独立的点云融合程序
融合三架无人机采集的点云数据
"""

import numpy as np
import os
from typing import List, Dict
from control.pointcloud_fusion import PointCloudFusion

def load_pointclouds_from_directories():
    """
    从点云目录加载所有点云
    :return: {drone_name: [pointcloud_info_list]}
    """
    pointclouds_by_drone = {}
    
    for drone_name in ["Drone1", "Drone2", "Drone3"]:
        pc_dir = f"pointclouds_{drone_name}"
        if not os.path.exists(pc_dir):
            print(f"[警告] 目录不存在: {pc_dir}")
            continue
        
        pc_files = sorted([f for f in os.listdir(pc_dir) if f.endswith('.ply')])
        pointclouds = []
        
        for pc_file in pc_files:
            pc_path = os.path.join(pc_dir, pc_file)
            # 从文件名提取索引（用于排序）
            try:
                idx = int(pc_file.split('_')[1].split('.')[0])
            except:
                idx = len(pointclouds)
            
            pointclouds.append({
                'filename': pc_path,
                'drone_name': drone_name,
                'index': idx
            })
        
        if pointclouds:
            # 按索引排序
            pointclouds.sort(key=lambda x: x['index'])
            pointclouds_by_drone[drone_name] = pointclouds
            print(f"[加载] {drone_name}: {len(pointclouds)} 个点云文件")
    
    return pointclouds_by_drone

def main():
    print("=" * 60)
    print("点云融合程序")
    print("=" * 60)
    
    # 1. 加载所有点云
    print("\n[步骤1] 加载点云文件...")
    pointclouds_by_drone = load_pointclouds_from_directories()
    
    if not pointclouds_by_drone:
        print("[错误] 没有找到点云文件")
        return
    
    total_pointclouds = sum(len(pcs) for pcs in pointclouds_by_drone.values())
    print(f"[加载] 总共加载 {total_pointclouds} 个点云文件")
    
    # 2. 初始化点云融合处理器
    print("\n[步骤2] 初始化点云融合处理器...")
    pc_fusion_processor = PointCloudFusion(max_iterations=50, distance_threshold=0.05)
    
    # 3. 准备点云数据列表
    print("\n[步骤3] 准备点云数据...")
    all_pointclouds = []
    
    # 按无人机顺序和索引顺序添加点云
    drone_order = ["Drone1", "Drone2", "Drone3"]
    for drone_name in drone_order:
        if drone_name in pointclouds_by_drone:
            for pc_info in pointclouds_by_drone[drone_name]:
                # 加载点云
                points = pc_fusion_processor.load_pointcloud_from_ply(pc_info['filename'])
                if points is not None and len(points) > 0:
                    all_pointclouds.append({
                        'points': points,
                        'filename': pc_info['filename'],
                        'drone_name': drone_name,
                        'position': [0, 0, 0],  # 默认位置，实际应该从元数据获取
                        'orientation': [1, 0, 0, 0]  # 默认姿态
                    })
                    print(f"  [加载] {pc_info['filename']}: {len(points)} 个点")
    
    if len(all_pointclouds) == 0:
        print("[错误] 没有成功加载的点云")
        return
    
    print(f"[准备] 准备融合 {len(all_pointclouds)} 个点云...")
    
    # 4. 执行点云融合
    print("\n[步骤4] 执行点云融合（ICP算法）...")
    fused_pointcloud = pc_fusion_processor.fuse_pointclouds(
        all_pointclouds,
        method='icp'
    )
    
    if fused_pointcloud is None or len(fused_pointcloud) == 0:
        print("[错误] 点云融合失败")
        return
    
    # 5. 保存结果
    print("\n[步骤5] 保存融合结果...")
    output_dir = "fusion_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "fused_pointcloud_final.ply")
    pc_fusion_processor.save_pointcloud_ply(fused_pointcloud, output_path)
    
    # 计算点云统计信息
    x_range = [fused_pointcloud[:, 0].min(), fused_pointcloud[:, 0].max()]
    y_range = [fused_pointcloud[:, 1].min(), fused_pointcloud[:, 1].max()]
    z_range = [fused_pointcloud[:, 2].min(), fused_pointcloud[:, 2].max()]
    
    print("\n" + "=" * 60)
    print("点云融合完成！")
    print("=" * 60)
    print(f"融合结果已保存到: {output_path}")
    print(f"融合点云点数: {len(fused_pointcloud)}")
    print(f"点云范围:")
    print(f"  X: [{x_range[0]:.2f}, {x_range[1]:.2f}]")
    print(f"  Y: [{y_range[0]:.2f}, {y_range[1]:.2f}]")
    print(f"  Z: [{z_range[0]:.2f}, {z_range[1]:.2f}]")
    print(f"融合了 {len(all_pointclouds)} 个点云文件")
    print(f"文件位置: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
