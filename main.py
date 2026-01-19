import time
import numpy as np
import cv2
import os
import threading
from swarm.drone import Drone
from swarm.communication import DroneNetwork
from swarm.udp_communication import UDPCommunication
from control.path_planner import RectanglePathPlanner
from control.image_fusion import ImageFusion
from config.swarm_config import *

def main():
    print("=" * 60)
    print("无人机集群协同数据采集与图像融合系统")
    print("=" * 60)
    
    # 1. 初始化无人机（头机为Drone1）
    leader = Drone("Drone1", [0, 0, 0], is_leader=True)
    drone2 = Drone("Drone2", [15, 0, 0], is_leader=False)
    drone3 = Drone("Drone3", [-15, 0, 0], is_leader=False)
    
    drones = [leader, drone2, drone3]
    follower_drones = [drone2, drone3]
    
    # 2. 初始化通信网络
    network = DroneNetwork(leader)
    for drone in drones:
        network.register_drone(drone)
    
    # 2.5. 初始化UDP通信（可选，用于真实网络通信）
    # 头机使用9001端口，跟随者使用9002和9003
    udp_comms = {}
    try:
        udp_comms[leader.name] = UDPCommunication(leader.name, 9001)
        udp_comms[leader.name].start()
        
        udp_comms[drone2.name] = UDPCommunication(drone2.name, 9002, leader_port=9001)
        udp_comms[drone2.name].start()
        
        udp_comms[drone3.name] = UDPCommunication(drone3.name, 9003, leader_port=9001)
        udp_comms[drone3.name].start()
        
        print("\n[系统] UDP通信已启动")
    except Exception as e:
        print(f"\n[警告] UDP通信启动失败: {e}，将使用模拟通信")
        udp_comms = {}
    
    print("\n[系统] 无人机初始化完成")
    
    # 3. 起飞
    print("\n[系统] 无人机起飞中...")
    for d in drones:
        d.takeoff(TARGET_ALTITUDE)
        print(f"[系统] {d.name} 已起飞到高度 {TARGET_ALTITUDE}m")
    time.sleep(2)
    
    # 3.5. 设置摄像机朝向（向下倾斜45度，便于拍摄地面）
    # 注意：摄像机朝向需要在settings.json中配置，运行时无法动态修改
    print("\n[系统] 摄像机朝向配置检查...")
    print("[提示] 摄像机朝向已在settings.json中配置，如需修改请编辑配置文件")
    for d in drones:
        d.set_camera_orientation(pitch=-45, yaw=0, roll=0)  # 仅用于记录，不影响实际配置
    time.sleep(1)
    
    # 4. 为每架无人机规划矩形路径
    # 头机：中心在(0,0)，扫描区域 20m x 20m
    # 无人机2：中心在(15,0)，扫描区域 20m x 20m
    # 无人机3：中心在(-15,0)，扫描区域 20m x 20m
    planners = {}
    planners[leader.name] = RectanglePathPlanner(
        center_pos=[0, 0],
        width=20.0,
        height=20.0,
        altitude=-TARGET_ALTITUDE,
        velocity=2.0
    )
    planners[drone2.name] = RectanglePathPlanner(
        center_pos=[15, 0],
        width=20.0,
        height=20.0,
        altitude=-TARGET_ALTITUDE,
        velocity=2.0
    )
    planners[drone3.name] = RectanglePathPlanner(
        center_pos=[-15, 0],
        width=20.0,
        height=20.0,
        altitude=-TARGET_ALTITUDE,
        velocity=2.0
    )
    
    print("\n[系统] 路径规划完成")
    
    # 4.5. 所有无人机先飞到各自路径起点并悬停
    print("\n[系统] 无人机飞向路径起点...")
    start_positions = {}
    for drone in drones:
        planner = planners[drone.name]
        start_pos = planner.get_start_waypoint()
        start_positions[drone.name] = start_pos
        drone.move_to_position(start_pos, velocity=2.0)
        print(f"[系统] {drone.name} 飞向起点 ({start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f})")
    
    # 等待所有无人机到达起点
    print("\n[系统] 等待所有无人机到达起点并悬停...")
    time.sleep(5)  # 给足够时间到达起点
    
    # 检查是否到达起点
    for drone in drones:
        current_pos = drone.get_position()
        start_pos = start_positions[drone.name]
        distance = np.linalg.norm(current_pos - start_pos)
        if distance > 2.0:
            print(f"[警告] {drone.name} 尚未到达起点，距离: {distance:.2f}m，继续等待...")
            drone.move_to_position(start_pos, velocity=2.0)
            time.sleep(3)
    
    print("\n[系统] 所有无人机已到达起点，准备同时开始扫描...")
    time.sleep(2)  # 悬停2秒，确保稳定
    
    # 4.6. 时序同步：头机广播同步信号，所有无人机同时开始
    print("\n[系统] 发送同步信号，所有无人机同时开始扫描...")
    sync_timestamp = time.time()
    
    if udp_comms and leader.name in udp_comms:
        # 使用UDP广播同步信号
        udp_comms[leader.name].broadcast_sync(sync_timestamp)
        print(f"[同步] 头机广播同步信号，时间戳: {sync_timestamp}")
    else:
        # 如果没有UDP，使用简单的时间同步
        print(f"[同步] 使用时间同步，基准时间: {sync_timestamp}")
    
    # 等待所有无人机收到同步信号（UDP通信有延迟）
    time.sleep(0.5)
    
    # 5. 执行矩形路径扫描并采集图像（同时开始）
    image_capture_interval = 2.0  # 每2秒采集一次图像
    pointcloud_capture_interval = 3.0  # 每3秒采集一次点云
    last_capture_time = {d.name: 0.0 for d in drones}
    last_pc_capture_time = {d.name: 0.0 for d in drones}
    max_iterations = 2000  # 最大迭代次数，防止无限循环
    iteration = 0
    
    all_paths_completed = False
    start_time = time.time()  # 记录开始时间，用于同步
    
    while not all_paths_completed and iteration < max_iterations:
        iteration += 1
        current_time = time.time()
        # 使用相对于同步时间的时间戳
        sync_relative_time = current_time - sync_timestamp
        all_paths_completed = True
        
        for drone in drones:
            planner = planners[drone.name]
            
            if not planner.is_completed():
                all_paths_completed = False
                
                # 获取当前位置
                current_pos = drone.get_position()
                
                # 更新路径规划，获取目标速度
                target_velocity = planner.update(current_pos, distance_threshold=1.5)
                
                # 控制无人机移动
                drone.move(target_velocity, dt=DT)
                
                # 定期采集图像（同步采集）
                if current_time - last_capture_time[drone.name] >= image_capture_interval:
                    img_info = drone.capture_image()
                    if img_info:
                        print(f"[采集] {drone.name} 在位置 {current_pos[:2]} 采集图像 (同步时间: {sync_relative_time:.2f}s)")
                        last_capture_time[drone.name] = current_time
                        
                        # 使用UDP发送数据（如果可用）
                        if udp_comms and drone.name in udp_comms and not drone.is_leader:
                            udp_comms[drone.name].send_to_leader(img_info, 'image_data')
                        
                        # 同时使用模拟通信（作为备用）
                        if not drone.is_leader:
                            network.send_data_to_leader(drone, img_info)
                
                # 定期采集点云（独立于图像采集）
                if current_time - last_pc_capture_time[drone.name] >= pointcloud_capture_interval:
                    pc_info = drone.capture_pointcloud()
                    if pc_info:
                        print(f"[采集] {drone.name} 采集点云，点数: {pc_info['point_count']} (同步时间: {sync_relative_time:.2f}s)")
                        last_pc_capture_time[drone.name] = current_time
                        
                        # 使用UDP发送点云数据（如果可用）
                        if udp_comms and drone.name in udp_comms and not drone.is_leader:
                            udp_comms[drone.name].send_to_leader(pc_info, 'pointcloud_data')
                        
                        # 同时使用模拟通信（作为备用）
                        if not drone.is_leader:
                            network.send_data_to_leader(drone, pc_info)
        
        time.sleep(DT)
        
        # 每50次迭代打印一次进度
        if iteration % 50 == 0:
            progress_info = []
            for drone in drones:
                progress = planners[drone.name].get_progress()
                progress_info.append(f"{drone.name}: {progress*100:.1f}%")
            print(f"[进度] 迭代 {iteration}: {', '.join(progress_info)}")
    
    print("\n[系统] 矩形路径扫描完成")
    
    # 6. 确保所有跟随者将剩余图像和点云发送到头机
    print("\n[系统] 收集所有采集的数据...")
    for follower in follower_drones:
        network.send_all_images_to_leader(follower)
        # 收集点云数据（通过UDP或模拟通信）
        if udp_comms and follower.name in udp_comms:
            received_data_udp = udp_comms[follower.name].get_received_data()
            for item in received_data_udp:
                if item.get('data', {}).get('point_count'):
                    network.send_data_to_leader(follower, item['data'])
    
    # 头机自己的图像和点云也需要添加到网络
    for img_info in leader.get_captured_images():
        network.send_data_to_leader(leader, img_info)
    
    # 统计点云数据
    print("\n[系统] 数据采集统计:")
    for drone in drones:
        img_count = len(drone.get_captured_images())
        # 统计点云文件数量
        pc_dir = f"pointclouds_{drone.name}"
        pc_count = len([f for f in os.listdir(pc_dir) if f.endswith('.ply')]) if os.path.exists(pc_dir) else 0
        print(f"  {drone.name}: {img_count} 张图像, {pc_count} 个点云文件")
    
    # 7. 图像融合
    print("\n[系统] 开始图像融合处理...")
    fusion_processor = ImageFusion()
    
    # 按发送者分组图像
    all_images_for_fusion = []
    received_data = network.get_received_data()
    
    print(f"[系统] 头机共接收到 {len(received_data)} 张图像")
    
    # 提取所有图像
    for item in received_data:
        all_images_for_fusion.append(item['data'])
    
    if len(all_images_for_fusion) > 0:
        print(f"\n[图像融合] 准备融合 {len(all_images_for_fusion)} 张图像")
        print(f"[图像融合] 图像来源统计:")
        sender_count = {}
        for item in received_data:
            sender = item['sender']
            sender_count[sender] = sender_count.get(sender, 0) + 1
        for sender, count in sender_count.items():
            print(f"  {sender}: {count} 张")
        
        # 执行图像融合
        print(f"\n[图像融合] 开始执行融合算法（金字塔融合）...")
        fused_image = fusion_processor.fuse_multiple_images(
            all_images_for_fusion, 
            method='pyramid'
        )
        
        if fused_image is not None:
            # 保存融合结果
            output_dir = "fusion_results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "fused_result.png")
            cv2.imwrite(output_path, cv2.cvtColor(fused_image, cv2.COLOR_RGB2BGR))
            
            print(f"\n{'='*60}")
            print(f"[系统] 图像融合完成！")
            print(f"{'='*60}")
            print(f"融合结果已保存到: {output_path}")
            print(f"融合图像尺寸: {fused_image.shape[1]} x {fused_image.shape[0]} 像素")
            print(f"融合图像通道数: {fused_image.shape[2] if len(fused_image.shape) > 2 else 1}")
            print(f"\n融合后的图像特点:")
            print(f"  - 融合了 {len(all_images_for_fusion)} 张不同视角的图像")
            print(f"  - 使用多分辨率金字塔融合算法")
            print(f"  - 生成无缝的广角图像")
            print(f"  - 文件位置: {os.path.abspath(output_path)}")
        else:
            print("\n[错误] 图像融合失败，请检查图像质量和匹配情况")
    else:
        print("\n[警告] 没有可融合的图像")
    
    # 8. 返航（异步执行，避免阻塞）
    print("\n[系统] 无人机返航中...")
    
    def return_home_with_timeout(drone, timeout=30):
        """带超时的返航函数"""
        p = drone.start_pos
        future = drone.client.moveToPositionAsync(
            float(p[0]), float(p[1]), float(p[2]), 2.0,
            vehicle_name=drone.name
        )
        
        # 使用线程实现超时
        result = [False]
        def wait_for_completion():
            try:
                future.join()
                result[0] = True
            except:
                pass
        
        thread = threading.Thread(target=wait_for_completion)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        return result[0]
    
    # 所有无人机同时返航
    return_threads = []
    for d in drones:
        def return_task(drone=d):
            try:
                if return_home_with_timeout(drone, timeout=30):
                    print(f"[系统] {drone.name} 已返航")
                else:
                    print(f"[警告] {drone.name} 返航超时")
            except Exception as e:
                print(f"[警告] {drone.name} 返航失败: {e}")
        
        t = threading.Thread(target=return_task)
        t.start()
        return_threads.append(t)
        print(f"[系统] {d.name} 开始返航...")
    
    # 等待所有返航线程完成
    for t in return_threads:
        t.join(timeout=35)
    
    time.sleep(2)  # 等待稳定
    
    # 9. 降落（异步执行）
    print("\n[系统] 无人机降落中...")
    
    def land_with_timeout(drone, timeout=15):
        """带超时的降落函数"""
        future = drone.client.landAsync(vehicle_name=drone.name)
        
        result = [False]
        def wait_for_completion():
            try:
                future.join()
                result[0] = True
            except:
                pass
        
        thread = threading.Thread(target=wait_for_completion)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        return result[0]
    
    # 所有无人机同时降落
    land_threads = []
    for d in drones:
        def land_task(drone=d):
            try:
                if land_with_timeout(drone, timeout=15):
                    print(f"[系统] {drone.name} 已降落")
                else:
                    print(f"[警告] {drone.name} 降落超时")
            except Exception as e:
                print(f"[警告] {drone.name} 降落失败: {e}")
        
        t = threading.Thread(target=land_task)
        t.start()
        land_threads.append(t)
        print(f"[系统] {d.name} 开始降落...")
    
    # 等待所有降落线程完成
    for t in land_threads:
        t.join(timeout=20)
    
    print("\n" + "=" * 60)
    print("任务完成！")
    print("=" * 60)
    
    # 打印统计信息
    print("\n[统计] 最终统计:")
    for drone in drones:
        img_count = len(drone.get_captured_images())
        pc_dir = f"pointclouds_{drone.name}"
        pc_count = len([f for f in os.listdir(pc_dir) if f.endswith('.ply')]) if os.path.exists(pc_dir) else 0
        print(f"  {drone.name}: {img_count} 张图像, {pc_count} 个点云文件")
    print(f"  融合结果: 1 张")
    
    # 停止UDP通信
    if udp_comms:
        print("\n[系统] 停止UDP通信...")
        for comm in udp_comms.values():
            comm.stop()

if __name__ == "__main__":
    main()