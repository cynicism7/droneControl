import numpy as np
from swarm.boids import compute_boids_velocity, compute_avoidance_vector

class SwarmController:
    def __init__(self, drones, center, radius=20.0):
        self.drones = drones
        self.center = center
        self.radius = radius
        self.angle = 0.0

    def step(self):
        # 围绕起飞点飞行一圈（圆轨迹）
        self.angle += 0.02
        target_pos = self.center + np.array([
            self.radius * np.cos(self.angle),
            self.radius * np.sin(self.angle),
            0
        ])

        swarm_center = np.mean(
            [d.get_position() for d in self.drones], axis=0
        )
        target_velocity = target_pos - swarm_center

        # 集群级避障向量
        v_avoid = compute_avoidance_vector(self.drones)

        for drone in self.drones:
            v = compute_boids_velocity(
                drone,
                self.drones,
                target_velocity,
                v_avoid
            )
            drone.move(v)
