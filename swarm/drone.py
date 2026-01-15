import airsim
import numpy as np
from config.swarm_config import DT

class Drone:
    def __init__(self, name, start_pos):
        self.name = name    # 无人机名称
        self.start_pos = np.array(start_pos)                        # 起飞位置
        self.client = airsim.MultirotorClient()                     # 连接 AirSim
        self.client.confirmConnection()                             # 确认连接
        self.client.enableApiControl(True, self.name)      # 启用 API 控制
        self.client.armDisarm(True, self.name)                  # 解锁
        self.prev_velocity = np.zeros(3)                            # 上一时刻速度

    # 起飞
    def takeoff(self, altitude):
        self.client.takeoffAsync(vehicle_name=self.name).join()
        self.client.moveToZAsync(-altitude, 1.5, vehicle_name=self.name).join()

    def get_position(self):
        s = self.client.getMultirotorState(vehicle_name=self.name)
        p = s.kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val])

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

    def go_home(self):
        p = self.start_pos
        self.client.moveToPositionAsync(
            p[0], p[1], p[2], 2,
            vehicle_name=self.name
        ).join()
