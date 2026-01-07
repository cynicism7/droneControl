import airsim
import numpy as np
from config.swarm_config import TARGET_ALTITUDE, DT

class Drone:
    def __init__(self, name, start_pos):
        self.name = name
        self.start_pos = np.array(start_pos)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)
        self.prev_velocity = np.zeros(3)

    def takeoff(self, target_altitude=TARGET_ALTITUDE):
        self.client.armDisarm(True, self.name)
        self.client.enableApiControl(True, self.name)
        self.client.takeoffAsync(vehicle_name=self.name).join()
        self.client.moveToZAsync(-target_altitude, 2, vehicle_name=self.name).join()

    def get_position(self):
        state = self.client.getMultirotorState(vehicle_name=self.name)
        p = state.kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val])

    def move(self, velocity, dt=DT):
        # 平滑速度
        velocity = 0.7 * self.prev_velocity + 0.3 * velocity
        self.prev_velocity = velocity
        self.client.moveByVelocityAsync(
            velocity[0], velocity[1], velocity[2],
            duration=dt,
            vehicle_name=self.name
        )

    def go_home(self):
        pos = self.start_pos
        self.client.moveToPositionAsync(pos[0], pos[1], pos[2], 3, vehicle_name=self.name).join()

    # --------------------------
    # 雷达与摄像头
    # --------------------------
    def get_lidar_points(self, lidar_name="LidarFront"):
        try:
            data = self.client.getLidarData(lidar_name=lidar_name, vehicle_name=self.name)
        except Exception:
            return None
        if len(data.point_cloud) < 3:
            return None
        pts = np.array(data.point_cloud, dtype=np.float32)
        return pts.reshape(-1, 3)

    def get_front_depth_points(self):
        try:
            response = self.client.simGetImage("front_rgb", airsim.ImageType.DepthPerspective)
            if response is None:
                return None
            depth_img = np.frombuffer(response.image_data_float, dtype=np.float32)
            w, h = response.width, response.height
            depth_img = depth_img.reshape(h, w)
            xs, ys = np.meshgrid(np.arange(w), np.arange(h))
            zs = depth_img
            pts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
            return pts
        except Exception:
            return None
