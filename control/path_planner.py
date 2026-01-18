"""
矩形路径规划模块
为每架无人机规划独立的矩形扫描路径
"""

import numpy as np
import time
from config.swarm_config import DT

class RectanglePathPlanner:
    """矩形路径规划器"""
    
    def __init__(self, center_pos, width, height, altitude, velocity=2.0):
        """
        初始化矩形路径规划器
        :param center_pos: 矩形中心位置 [x, y]
        :param width: 矩形宽度（米）
        :param height: 矩形高度（米）
        :param altitude: 飞行高度（米，负值）
        :param velocity: 飞行速度（米/秒）
        """
        self.center = np.array([center_pos[0], center_pos[1], altitude])
        self.width = width
        self.height = height
        self.velocity = velocity
        self.current_waypoint_idx = 0
        
        # 计算矩形的四个顶点（相对于中心）
        half_w = width / 2.0
        half_h = height / 2.0
        
        # 矩形路径：起点 -> 右上 -> 右下 -> 左下 -> 左上 -> 起点
        self.waypoints = [
            np.array([-half_w, -half_h, altitude]),  # 起点（左下）
            np.array([half_w, -half_h, altitude]),    # 右上
            np.array([half_w, half_h, altitude]),     # 右下
            np.array([-half_w, half_h, altitude]),    # 左下
            np.array([-half_w, -half_h, altitude]),   # 左上（回到起点）
        ]
        
        # 转换为全局坐标系
        self.waypoints = [wp + self.center for wp in self.waypoints]
        self.path_completed = False
        
    def get_current_waypoint(self):
        """获取当前目标航点"""
        if self.current_waypoint_idx >= len(self.waypoints):
            self.path_completed = True
            return self.waypoints[-1]
        return self.waypoints[self.current_waypoint_idx]
    
    def update(self, current_pos, distance_threshold=1.0):
        """
        更新路径规划状态
        :param current_pos: 当前位置
        :param distance_threshold: 到达航点的距离阈值（米）
        :return: 目标速度向量
        """
        if self.path_completed:
            return np.array([0.0, 0.0, 0.0])
        
        current_waypoint = self.get_current_waypoint()
        current_pos = np.array(current_pos)
        
        # 计算到目标航点的向量
        direction = current_waypoint - current_pos
        distance = np.linalg.norm(direction)
        
        # 如果到达当前航点，切换到下一个
        if distance < distance_threshold:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.waypoints):
                self.path_completed = True
                return np.array([0.0, 0.0, 0.0])
            current_waypoint = self.get_current_waypoint()
            direction = current_waypoint - current_pos
            distance = np.linalg.norm(direction)
        
        # 计算目标速度（归一化方向向量 * 速度）
        if distance > 0.1:
            direction_normalized = direction / distance
            target_velocity = direction_normalized * self.velocity
        else:
            target_velocity = np.array([0.0, 0.0, 0.0])
        
        return target_velocity
    
    def is_completed(self):
        """检查路径是否完成"""
        return self.path_completed
    
    def get_progress(self):
        """获取路径完成进度（0-1）"""
        if len(self.waypoints) == 0:
            return 1.0
        return min(1.0, self.current_waypoint_idx / len(self.waypoints))
    
    def get_start_waypoint(self):
        """获取路径起点"""
        if len(self.waypoints) > 0:
            return self.waypoints[0]
        return self.center
