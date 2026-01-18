import airsim
import numpy as np
import cv2
import os
from config.swarm_config import DT

class Drone:
    def __init__(self, name, start_pos, is_leader=False):
        self.name = name    # 无人机名称
        self.start_pos = np.array(start_pos)                        # 起飞位置
        self.is_leader = is_leader                                  # 是否为头机
        self.client = airsim.MultirotorClient()                     # 连接 AirSim
        self.client.confirmConnection()                             # 确认连接
        self.client.enableApiControl(True, self.name)      # 启用 API 控制
        self.client.armDisarm(True, self.name)                  # 解锁
        self.prev_velocity = np.zeros(3)                            # 上一时刻速度
        self.captured_images = []                                   # 采集的图像列表
        self.image_data_dir = f"images_{self.name}"                 # 图像存储目录
        os.makedirs(self.image_data_dir, exist_ok=True)

    # 起飞
    def takeoff(self, altitude):
        self.client.takeoffAsync(vehicle_name=self.name).join()
        self.client.moveToZAsync(-altitude, 1.5, vehicle_name=self.name).join()

    def get_position(self):
        s = self.client.getMultirotorState(vehicle_name=self.name)
        p = s.kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val])
    
    def get_orientation(self):
        """获取无人机姿态（四元数）"""
        s = self.client.getMultirotorState(vehicle_name=self.name)
        q = s.kinematics_estimated.orientation
        return np.array([q.w_val, q.x_val, q.y_val, q.z_val])

    # 速度控制
    def move(self, v, dt=DT):
        # 低通滤波（减少抖动）
        v = 0.6 * self.prev_velocity + 0.4 * v
        self.prev_velocity = v

        self.client.moveByVelocityAsync(
            v[0], v[1], v[2],
            duration=dt,
            vehicle_name=self.name
        )

    def move_to_position(self, target_pos, velocity=2.0):
        """移动到指定位置"""
        self.client.moveToPositionAsync(
            target_pos[0], target_pos[1], target_pos[2], 
            velocity, vehicle_name=self.name
        ).join()

    def set_camera_orientation(self, pitch=-45, yaw=0, roll=0, camera_name="front_rgb"):
        """
        设置摄像机朝向
        注意：AirSim中摄像机朝向需要在settings.json中配置
        此函数仅用于记录配置，实际朝向已在settings.json中设置
        :param pitch: 俯仰角（度，负值向下）
        :param yaw: 偏航角（度）
        :param roll: 翻滚角（度）
        """
        # AirSim的摄像机朝向需要在settings.json中配置
        # 运行时无法动态修改，所以这里只记录配置
        # 如果需要改变朝向，请修改settings.json中的Pitch、Yaw、Roll参数
        print(f"[信息] {self.name} 摄像机朝向配置: Pitch={pitch}°, Yaw={yaw}°, Roll={roll}°")
        print(f"[提示] 如需修改摄像机朝向，请编辑settings.json中的相机配置")

    def capture_image(self, camera_name="front_rgb"):
        """采集图像"""
        try:
            # 获取图像
            responses = self.client.simGetImages([
                airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
            ], vehicle_name=self.name)
            
            if responses and responses[0]:
                response = responses[0]
                # 将图像数据转换为numpy数组
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                
                # 获取当前位置和姿态
                pos = self.get_position()
                orientation = self.get_orientation()
                
                # 保存图像信息（将numpy类型转换为Python原生类型，避免序列化错误）
                image_info = {
                    'image': img_rgb,
                    'position': pos.tolist() if isinstance(pos, np.ndarray) else pos,
                    'orientation': orientation.tolist() if isinstance(orientation, np.ndarray) else orientation,
                    'timestamp': int(response.time_stamp) if hasattr(response.time_stamp, '__int__') else response.time_stamp,
                    'camera_name': camera_name
                }
                
                # 保存到本地
                filename = f"{self.image_data_dir}/img_{len(self.captured_images):04d}.png"
                cv2.imwrite(filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                image_info['filename'] = filename
                
                self.captured_images.append(image_info)
                return image_info
        except Exception as e:
            print(f"{self.name} 图像采集失败: {e}")
            return None
    
    def capture_pointcloud(self):
        """
        采集点云数据（LiDAR）
        :return: 点云数据字典
        """
        try:
            # 获取LiDAR数据（使用深度图像生成点云）
            # 注意：AirSim的LiDAR需要先在settings.json中配置
            # 这里我们使用深度图像来生成点云
            
            # 获取深度图像
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_depth", airsim.ImageType.DepthPerspective, True, False)
            ], vehicle_name=self.name)
            
            if not responses or not responses[0]:
                return None
            
            response = responses[0]
            depth_img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
            depth_img = np.array(depth_img)
            
            # 获取相机内参（从settings.json中的FOV计算）
            fov = 90  # 默认FOV
            h, w = depth_img.shape
            fx = fy = w / (2 * np.tan(np.radians(fov / 2)))
            cx, cy = w / 2, h / 2
            
            # 生成点云
            points = []
            for v in range(h):
                for u in range(w):
                    depth = depth_img[v, u]
                    if depth > 0 and depth < 100:  # 过滤无效深度
                        # 从像素坐标转换为3D坐标
                        x = (u - cx) * depth / fx
                        y = (v - cy) * depth / fy
                        z = depth
                        points.append([x, y, z])
            
            if len(points) == 0:
                return None
            
            points = np.array(points, dtype=np.float32)
            
            # 获取当前位置和姿态
            pos = self.get_position()
            orientation = self.get_orientation()
            
            pointcloud_info = {
                'points': points.tolist(),  # 转换为列表以便序列化
                'position': pos.tolist() if isinstance(pos, np.ndarray) else pos,
                'orientation': orientation.tolist() if isinstance(orientation, np.ndarray) else orientation,
                'timestamp': int(response.time_stamp) if hasattr(response.time_stamp, '__int__') else response.time_stamp,
                'point_count': len(points)
            }
            
            # 保存点云数据到文件（PLY格式）
            pointcloud_dir = f"pointclouds_{self.name}"
            os.makedirs(pointcloud_dir, exist_ok=True)
            filename = f"{pointcloud_dir}/pc_{len(self.captured_images):04d}.ply"
            self._save_pointcloud_ply(points, filename)
            pointcloud_info['filename'] = filename
            
            return pointcloud_info
        except Exception as e:
            print(f"{self.name} 点云采集失败: {e}")
            return None
    
    def _save_pointcloud_ply(self, points, filename):
        """保存点云为PLY格式"""
        try:
            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                for point in points:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
        except Exception as e:
            print(f"保存点云文件失败: {e}")

    def get_captured_images(self):
        """获取所有采集的图像"""
        return self.captured_images

    def go_home(self, timeout=30):
        """
        返航到起始位置
        :param timeout: 超时时间（秒）
        """
        p = self.start_pos
        # 确保坐标是Python原生类型，避免msgpack序列化错误
        try:
            future = self.client.moveToPositionAsync(
                float(p[0]), float(p[1]), float(p[2]), 2.0,
                vehicle_name=self.name
            )
            # 使用超时机制，避免无限等待
            future.join(timeout_sec=timeout)
        except Exception as e:
            print(f"[警告] {self.name} 返航时出错: {e}，尝试直接降落")
            # 如果返航失败，直接尝试降落
            try:
                self.client.landAsync(vehicle_name=self.name).join(timeout_sec=10)
            except:
                pass
