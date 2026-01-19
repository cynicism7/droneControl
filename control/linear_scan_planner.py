"""
直线扫描路径规划器
实现三架无人机在一条直线上，每架负责20m扫描范围，有20-30%重叠
"""

import numpy as np
import math
from typing import List, Tuple

class LinearScanPlanner:
    """
    直线扫描路径规划器
    三架无人机在一条直线上，朝向垂直于直线，每架负责20m范围
    """
    
    def __init__(self, line_start: List[float], line_end: List[float],
                 drone_index: int, scan_length: float = 20.0,
                 overlap_ratio: float = 0.25, altitude: float = -20.0,
                 velocity: float = 3.0):
        """
        初始化直线扫描路径规划器
        :param line_start: 直线起点 [x, y]
        :param line_end: 直线终点 [x, y]
        :param drone_index: 无人机索引 (0, 1, 2)
        :param scan_length: 每架无人机负责的扫描长度（米），默认20m
        :param overlap_ratio: 重叠比例（0-1），默认0.25（25%）
        :param altitude: 飞行高度（米，负值）
        :param velocity: 飞行速度（米/秒）
        """
        self.line_start = np.array(line_start)
        self.line_end = np.array(line_end)
        self.drone_index = drone_index
        self.scan_length = scan_length
        self.overlap_ratio = overlap_ratio
        self.altitude = altitude
        self.velocity = velocity
        
        # 计算直线方向和长度
        self.line_direction = self.line_end - self.line_start
        self.line_length = np.linalg.norm(self.line_direction)
        self.line_unit = self.line_direction / self.line_length if self.line_length > 0 else np.array([1, 0])
        
        # 计算扫描区域中心点（所有无人机朝向的目标）
        # 所有无人机应该朝向同一个目标区域，方便后续拼接
        self.scan_center = (self.line_start + self.line_end) / 2.0
        
        # 计算每架无人机的扫描区域起点
        # 总长度 = 3 * scan_length - 2 * overlap（因为中间有重叠）
        effective_length = scan_length * (1 - overlap_ratio)  # 有效长度（去除重叠）
        total_coverage = scan_length + 2 * effective_length  # 三架无人机的总覆盖
        
        # 计算每架无人机的起点位置
        if drone_index == 0:
            # 第一架：从line_start开始
            self.scan_start = self.line_start.copy()
        elif drone_index == 1:
            # 第二架：第一架结束 - 重叠
            self.scan_start = self.line_start + self.line_unit * (scan_length - scan_length * overlap_ratio)
        else:  # drone_index == 2
            # 第三架：第二架结束 - 重叠
            self.scan_start = self.line_start + self.line_unit * (2 * scan_length - 2 * scan_length * overlap_ratio)
        
        # 计算扫描终点
        self.scan_end = self.scan_start + self.line_unit * scan_length
        
        # 生成航点（沿直线移动，每2米一个航点）
        self.waypoints = []
        self.camera_yaws = []  # 存储每个航点的摄像头朝向（垂直于直线）
        
        waypoint_spacing = 2.0  # 航点间距（米）
        num_waypoints = int(scan_length / waypoint_spacing) + 1
        
        for i in range(num_waypoints):
            t = i / max(1, num_waypoints - 1)  # 0到1
            waypoint_2d = self.scan_start + t * (self.scan_end - self.scan_start)
            waypoint = np.array([waypoint_2d[0], waypoint_2d[1], altitude])
            self.waypoints.append(waypoint)
            
            # 计算朝向（朝向扫描区域中心点）
            # 所有无人机朝向同一个目标区域，方便后续拼接
            direction_to_center = self.scan_center - waypoint_2d
            yaw_rad = math.atan2(direction_to_center[1], direction_to_center[0])
            yaw_deg = math.degrees(yaw_rad)
            # 确保角度在0-360度范围内
            if yaw_deg < 0:
                yaw_deg += 360
            
            self.camera_yaws.append(yaw_deg)
        
        self.current_waypoint_idx = 0
        self.path_completed = False
        
        print(f"[直线扫描] Drone{drone_index+1} 路径规划完成")
        print(f"[直线扫描] 扫描区域: 起点({self.scan_start[0]:.1f}, {self.scan_start[1]:.1f}) -> "
              f"终点({self.scan_end[0]:.1f}, {self.scan_end[1]:.1f})")
        print(f"[直线扫描] 扫描长度: {scan_length}m, 重叠比例: {overlap_ratio*100:.1f}%")
        print(f"[直线扫描] 航点数量: {len(self.waypoints)}, 朝向: {yaw_deg:.1f}°")
    
    def get_start_waypoint(self) -> np.ndarray:
        """获取路径起点"""
        if len(self.waypoints) > 0:
            return self.waypoints[0]
        return np.array([self.scan_start[0], self.scan_start[1], self.altitude])
    
    def get_current_waypoint(self) -> np.ndarray:
        """获取当前目标航点"""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return self.waypoints[-1] if self.waypoints else None
    
    def get_current_camera_yaw(self) -> float:
        """获取当前航点的摄像头朝向（度）"""
        if self.current_waypoint_idx < len(self.camera_yaws):
            return self.camera_yaws[self.current_waypoint_idx]
        return self.camera_yaws[-1] if self.camera_yaws else 0.0
    
    def update(self, current_pos: np.ndarray, distance_threshold: float = 2.0) -> np.ndarray:
        """
        更新路径规划状态
        :param current_pos: 当前位置 [x, y, z]
        :param distance_threshold: 到达航点的距离阈值（米）
        :return: 目标速度向量
        """
        if self.path_completed:
            return np.array([0.0, 0.0, 0.0])
        
        if len(self.waypoints) == 0:
            self.path_completed = True
            return np.array([0.0, 0.0, 0.0])
        
        current_pos = np.array(current_pos)
        
        # 如果已经到达最后一个航点，检查是否完成
        if self.current_waypoint_idx >= len(self.waypoints):
            self.path_completed = True
            return np.array([0.0, 0.0, 0.0])
        
        current_waypoint = self.get_current_waypoint()
        if current_waypoint is None:
            self.path_completed = True
            return np.array([0.0, 0.0, 0.0])
        
        # 计算到目标航点的向量（只考虑XY平面距离）
        direction_2d = current_waypoint[:2] - current_pos[:2]
        distance_2d = np.linalg.norm(direction_2d)
        
        # 如果到达当前航点（XY平面），切换到下一个
        # 但确保不会跳过最后一个航点
        if distance_2d < distance_threshold:
            # 如果这是最后一个航点，标记为完成
            if self.current_waypoint_idx >= len(self.waypoints) - 1:
                self.path_completed = True
                return np.array([0.0, 0.0, 0.0])
            # 否则切换到下一个航点
            self.current_waypoint_idx += 1
            current_waypoint = self.get_current_waypoint()
            if current_waypoint is None:
                self.path_completed = True
                return np.array([0.0, 0.0, 0.0])
            direction_2d = current_waypoint[:2] - current_pos[:2]
            distance_2d = np.linalg.norm(direction_2d)
        
        # 计算完整的方向向量（包括高度）
        direction = current_waypoint - current_pos
        distance = np.linalg.norm(direction)
        
        # 计算目标速度（归一化方向向量 * 速度）
        if distance > 0.1:
            direction_normalized = direction / distance
            target_velocity = direction_normalized * self.velocity
        else:
            # 如果距离非常小，检查是否完成
            if self.current_waypoint_idx >= len(self.waypoints) - 1:
                self.path_completed = True
            target_velocity = np.array([0.0, 0.0, 0.0])
        
        return target_velocity
    
    def is_completed(self) -> bool:
        """检查路径是否完成"""
        return self.path_completed
    
    def get_progress(self) -> float:
        """获取路径完成进度（0-1）"""
        if len(self.waypoints) == 0:
            return 1.0
        return min(1.0, self.current_waypoint_idx / len(self.waypoints))
    
    def get_path_length(self) -> float:
        """计算路径总长度"""
        if len(self.waypoints) < 2:
            return 0.0
        total_length = 0.0
        for i in range(len(self.waypoints) - 1):
            total_length += np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
        return total_length
