import numpy as np
from config.swarm_config import TARGET_ALTITUDE, ALTITUDE_MARGIN, MAX_SPEED, MAX_Z_SPEED
from config.swarm_config import DESIRED_DISTANCE, MIN_DISTANCE, AVOID_DISTANCE
from config.swarm_config import K_TARGET, K_COHESION, K_SEPARATION, K_AVOID, CIRCLE_SPEED

# --------------------------
# 集群中心
# --------------------------
def compute_swarm_center(drones):
    positions = [d.get_position() for d in drones]
    return np.mean(positions, axis=0)

# --------------------------
# 避障融合 (雷达 + 前摄像头)
# --------------------------
def compute_avoidance_vector(drones):
    total_avoid = np.zeros(3)
    for d in drones:
        local_vectors = []

        for lidar_name in ["LidarFront", "LidarLeft", "LidarRight"]:
            pts = d.get_lidar_points(lidar_name)
            if pts is not None and len(pts) > 0:
                close_pts = pts[np.linalg.norm(pts, axis=1) < AVOID_DISTANCE]
                if len(close_pts) > 0:
                    obstacle_center = np.mean(close_pts, axis=0)
                    diff = d.get_position() - obstacle_center
                    weight = max(0, AVOID_DISTANCE - np.linalg.norm(diff))
                    local_vectors.append(diff / (np.linalg.norm(diff)+1e-3) * weight)

        depth_pts = d.get_front_depth_points()
        if depth_pts is not None and len(depth_pts) > 0:
            close_pts = depth_pts[np.linalg.norm(depth_pts, axis=1) < AVOID_DISTANCE]
            if len(close_pts) > 0:
                obstacle_center = np.mean(close_pts, axis=0)
                diff = d.get_position() - obstacle_center
                weight = max(0, AVOID_DISTANCE - np.linalg.norm(diff))
                local_vectors.append(diff / (np.linalg.norm(diff)+1e-3) * weight)

        if local_vectors:
            local_avoid = np.sum(local_vectors, axis=0)
            local_avoid[2] = np.clip(local_avoid[2], -ALTITUDE_MARGIN, ALTITUDE_MARGIN)
            total_avoid += local_avoid

    norm = np.linalg.norm(total_avoid)
    if norm < 1e-3:
        return np.zeros(3)
    return total_avoid / norm

# --------------------------
# Boids速度融合
# --------------------------
def compute_boids_velocity(drone, drones, target_velocity, v_avoid):
    pos_i = drone.get_position()
    cohesion = np.zeros(3)
    separation = np.zeros(3)
    count = 0

    for other in drones:
        if other.name == drone.name:
            continue
        pos_j = other.get_position()
        diff = pos_j - pos_i
        dist = np.linalg.norm(diff)

        if dist < 1e-3:
            continue
        if dist > DESIRED_DISTANCE:
            cohesion += diff
            count += 1
        if dist < MIN_DISTANCE:
            separation -= diff / dist
    if count > 0:
        cohesion /= count

    v = (
        K_TARGET     * target_velocity +
        K_COHESION   * cohesion +
        K_SEPARATION * separation +
        K_AVOID      * v_avoid
    )

    if np.linalg.norm(v_avoid[:2]) > 0:
        z_correction = TARGET_ALTITUDE + v_avoid[2]
    else:
        z_correction = TARGET_ALTITUDE
    delta_z = z_correction - pos_i[2]
    v[2] = np.clip(delta_z, -MAX_Z_SPEED, MAX_Z_SPEED)

    speed = np.linalg.norm(v)
    if speed > MAX_SPEED:
        v = v / speed * MAX_SPEED
    return v

# --------------------------
# 绕原点圆周速度
# --------------------------
def compute_circular_velocity(pos, center=np.array([0,0,TARGET_ALTITUDE]), speed=CIRCLE_SPEED):
    vec = pos - center
    vec[2] = 0
    dist = np.linalg.norm(vec)
    if dist < 1e-3:
        tangent = np.array([0, speed, 0])
    else:
        tangent = np.array([-vec[1], vec[0], 0])
        tangent = tangent / np.linalg.norm(tangent) * speed
    return tangent
