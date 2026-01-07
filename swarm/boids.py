import numpy as np
from config.swarm_config import *

def compute_circle_velocity(pos):
    center = np.array([0.0, 0.0, pos[2]])
    vec = pos - center
    vec[2] = 0.0

    r = np.linalg.norm(vec)
    if r < 1e-3:
        return np.array([0.0, CIRCLE_SPEED, 0.0])

    r_hat = vec / r

    # 切向单位向量（3D）
    t_hat = np.array([-r_hat[1], r_hat[0], 0.0])

    v_tangent = t_hat * CIRCLE_SPEED
    v_radial = (CIRCLE_RADIUS - r) * r_hat * 0.8

    return v_tangent + v_radial


def leader_velocity(leader):
    pos = leader.get_position()
    return compute_circle_velocity(pos)


def follower_velocity(leader, follower, offset_body):
    # 领机圆周速度
    v_leader = compute_circle_velocity(leader.get_position())

    speed_xy = np.linalg.norm(v_leader[:2])
    if speed_xy < 1e-3:
        return np.zeros(3)

    # === 关键修复：heading/right 全部用 3 维 ===
    heading = np.array([
        v_leader[0] / speed_xy,
        v_leader[1] / speed_xy,
        0.0
    ])

    right = np.array([
        -heading[1],
         heading[0],
         0.0
    ])

    leader_pos = leader.get_position()
    follower_pos = follower.get_position()

    # 期望位置（世界坐标系）
    desired_pos = (
        leader_pos
        - heading * FORMATION_DISTANCE
        + right * offset_body
    )

    error = desired_pos - follower_pos

    v_form = K_FORMATION * error
    v = v_leader + v_form

    # 不由队形控制高度
    v[2] = 0.0

    speed = np.linalg.norm(v)
    if speed > MAX_SPEED:
        v = v / speed * MAX_SPEED

    return v
