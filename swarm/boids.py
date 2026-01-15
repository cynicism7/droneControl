import numpy as np
import math
from config.swarm_config import *

# 工具函数
def limit_norm(v, max_norm):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return v
    if n > max_norm:
        return v / n * max_norm
    return v


# Leader 控制
def leader_velocity(leader):
    pos = leader.get_position()
    x, y = pos[0], pos[1]

    r = math.hypot(x, y)
    # 1. 径向控制（拉到圆上）
    r_error = TARGET_RADIUS - r

    if r < 1e-3:
        radial = np.array([1.0, 0.0])
    else:
        radial = np.array([x / r, y / r])
    v_radial = KP_RADIUS * r_error * radial
    # 2. 切向控制（绕圈）
    if r < 1e-3:
        tangent = np.array([0.0, 0.0])
    else:
        tangent = np.array([-y / r, x / r])

    v_tangent = CIRCLE_SPEED * tangent

    v = v_radial + v_tangent
    # 3. 限速
    v_xy = limit_norm(v[:2], MAX_LEADER_SPEED)

    return np.array([v_xy[0], v_xy[1], 0.0], dtype=float)


# Follower 控制
def follower_velocity(leader, follower, lateral_offset):
    p_l = np.array(leader.get_position())
    p_f = np.array(follower.get_position())
    # 由几何关系确定 leader 朝向
    # leader 绕圆：切向方向 = (-y, x)
    x, y = p_l[0], p_l[1]
    r = math.hypot(x, y)

    if r < 1e-3:
        heading = np.array([1.0, 0.0])
    else:
        heading = np.array([-y / r, x / r])

    normal = np.array([-heading[1], heading[0]])
    # 期望编队位置
    target_pos = np.array([
        p_l[0] + lateral_offset * normal[0],
        p_l[1] + lateral_offset * normal[1],
        -TARGET_ALTITUDE
    ])

    error = target_pos - p_f
    # 水平控制
    v_xy = KP_POSITION * error[:2]
    v_xy = limit_norm(v_xy, CIRCLE_SPEED)
    # 高度独立回归
    z_err = (-TARGET_ALTITUDE) - p_f[2]
    v_z = KP_ALTITUDE * z_err
    v_z = np.clip(v_z, -MAX_Z_SPEED, MAX_Z_SPEED)

    return np.array([v_xy[0], v_xy[1], v_z], dtype=float)
